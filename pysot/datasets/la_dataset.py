from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
import math
from torch.utils.data import Dataset
# online crop
from pysot.utils.img_crop import crop_temp, crop_search, crop_tar
from pysot.utils.bbox import center2corner, Center
# heat map generation
from pysot.utils.image import get_affine_transform, affine_transform
from pysot.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from pysot.utils.image import draw_dense_reg
# augmentation and label
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, latency, num_frame, jitter, pre_target, start_idx):
        self.name = name
        self.root = root
        self.anno = anno
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        self.latency = latency
        self.input_jit = jitter
        self.pred_target = pre_target
        self.num_frame = num_frame
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        if self.name == 'VID':
            self.path_format = '{}.JPEG'
        else:
            self.path_format = '{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        self.mini_lenth = self.num_frame*(self.latency+1+self.input_jit)+1+self.pred_target # 1 5 9 13
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            print(video, trk, frm)
                            break
                    new_frames[frm] = bbox
                if len(new_frames) > self.mini_lenth:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
            # if debug and len(meta_data_new)>100:
                # break
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        if self.name == 'VID':
            frame = "{:06d}".format(frame)
        if self.name == 'LaSOT' or self.name == 'GOT':
            frame = "{:08d}".format(frame)
        elif self.name == 'DTB':
            frame = "{:05d}".format(frame)

        if frame not in self.labels[video][track]:
            print(video, track, frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame))
        # if frame not in self.labels[video][track]:
        # print(video, track, frame)
        assert frame in self.labels[video][track]
        image_anno = self.labels[video][track][frame]
        # img_sz = self.labels[video][track]['frame_sz']
        return image_path, image_anno

    def get_positive_pair(self, index):
        # find seq
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        # find template
        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        # determine jittered latency
        use_latency = (self.latency + 1 + np.random.randint(-self.input_jit, self.input_jit+1)) if self.input_jit else (self.latency+1)
        # determine search range
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        # Find search frames
        # first define mini lenth of a batch seq
        mini_lenth = self.num_frame*use_latency+1+self.pred_target
        assert len(search_range) > mini_lenth
        # # For short seq, skip < latency, average sampling
        # if len(search_range)<mini_lenth:
        #     step = np.floor(len(search_range)/(self.num_frame+2))
        #     assert step>0
        #     input_search = np.arange(search_range[0], search_range[0]+(self.num_frame+1)*step, step)
        #     tar_search = np.append(input_search[-1],search_range[-step:])
        # else:
        # For long seq, skip = latency, first define start of mini head randomly, and define mini batch
        start_search = np.random.choice(search_range[0:len(search_range)-mini_lenth]) # suppose 1
        input_search = np.arange(start_search, start_search+(self.num_frame+1)*use_latency, use_latency) # suppose 1 4 7 10 and 1 is for delta calculation
        tar_search = np.arange(input_search[-1],input_search[-1]+1+self.pred_target) # 10 11 12 13 14 15 ... and 10 is for delta calculationS
        input_search.sort()
        tar_search.sort()
        return self.get_image_anno(video_name, track, template_frame), \
            [self.get_image_anno(video_name, track, s_frame) for s_frame in input_search], \
            [self.get_image_anno(video_name, track, t_frame) for t_frame in tar_search], \
            [float(use_latency), self.pred_target]

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,cfg):
        super(TrkDataset, self).__init__()

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target
        self.anchor_target = AnchorTarget()
        self.latency = cfg.TRAIN.LATENCY
        self.frame_num = cfg.TRAIN.NUM_FRAME
        logger.info("Dataset latency set: " + str(self.latency))

        # create sub dataset
        self.use_img = cfg.DATASET.USE_IMG
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.O_ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    self.latency,
                    self.frame_num,
                    cfg.TRAIN.JITTER,
                    cfg.TRAIN.PRE_TARGET,
                    start,
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        # if neg:
        #     template = dataset.get_random_target(index)
        #     search = np.random.choice(self.all_dataset).get_random_target()
        # else:
        template, in_search, ta_search, param = dataset.get_positive_pair(index)

        # get image
        # may not use image
        template_image = None
        search_images = None
        tar_images = None
        # frame_sz = template[-1]
        if self.use_img:
            template_image = cv2.imread(template[0])
            search_images = [cv2.imread(s[0]) for s in in_search]
            tar_images = [cv2.imread(t[0]) for t in ta_search]

        # get cropped patch and location delta on the patch
        # template only need image patch
        tem_patch = crop_temp(template_image, template[1], exemplar_size=cfg.TRAIN.EXEMPLAR_SIZE, instanc_size=511)
        current_templatebox = None
        if tem_patch is not None:
            current_templatebox = self._get_bbox(tem_patch, template[1])#for template augmentation
        # in_search need biased image patch and corresponding delta
        s_patch, delta_in, search_box = crop_search(search_images, [s[1] for s in in_search], exemplar_size=cfg.TRAIN.EXEMPLAR_SIZE, instanc_size=cfg.TRAIN.SEARCH_SIZE)
        # tar_search only need corresponding delta
        # out_box= [[cx1,cy1,w1,h1],[cx2,cy2,w2,h2]], []
        tar_patch, delta_out, out_box = crop_tar(tar_images, [t[1] for t in ta_search], exemplar_size=cfg.TRAIN.EXEMPLAR_SIZE, instanc_size=cfg.TRAIN.SEARCH_SIZE)

        if cfg.DATASET.HM:
            # generate pred heat map
            ct = np.array([tar_patch[0].shape[1] / 2., tar_patch[0].shape[0] / 2.], dtype=np.float32)
            sz = max(tar_patch[0].shape[0], tar_patch[0].shape[1]) * 1.0
            output_h = cfg.PRED.OUTPUT_SZ
            output_w = cfg.PRED.OUTPUT_SZ
            # affine transform for gaussian generation
            trans_output = get_affine_transform(ct, sz, 0, [output_w, output_h])
            # init pred label
            hm = np.zeros((len(tar_patch), output_h, output_w), dtype=np.float32)
            wh = np.zeros((len(tar_patch), 2), dtype=np.float32)
            reg = np.zeros((len(tar_patch), 2), dtype=np.float32)
            ind = np.ones((1, len(tar_patch)), dtype=np.int64)
            reg_mask = np.ones((1, len(tar_patch)), dtype=np.uint8)
            draw_gaussian = draw_msra_gaussian if cfg.PRED.MSE_LOSS else \
                    draw_umich_gaussian
            # start generating label for centerpred
            for k in range(len(tar_patch)):
                # box in 255 image patch
                # from [cx, cy, w, h] to [x1, y1, x2, y2]
                bbox = np.array(center2corner(Center(out_box[k][1][0], out_box[k][1][1], out_box[k][1][2], out_box[k][1][3]))) 
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
                # box in output map
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(hm[k], ct_int, radius)
                    wh[k] = 1. * w/cfg.PRED.OUTPUT_SZ, 1. * h / cfg.PRED.OUTPUT_SZ
                    # ind[:,k] = ct_int[1] * output_w + ct_int[0]
                    reg[k] = ct - ct_int

        # for motion based predictor
        d_in = np.array(delta_in).astype(np.float32)
        d_out = np.array(delta_out).astype(np.float32)
        box_out = np.array(out_box).astype(np.float32)
        box_in = np.array(search_box).astype(np.float32)

        if tem_patch is not None:
            # use image
            template, _ = self.template_aug(tem_patch,
                                        current_templatebox,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)
            # search also need augmentation
            bboxs = [center2corner(Center(s_box[1][0], s_box[1][1], s_box[1][2], s_box[1][3])) for s_box in search_box]
            # get labels
            clss = []
            deltas = []
            delta_weights = []
            bboxes = []
            search = []
            for i in range(len(bboxs)):
                # color augmentation, box will not change
                s, b = self.search_aug(s_patch[i],
                                       bboxs[i],
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
                c, d, dw, _ = self.anchor_target(b, cfg.TRAIN.OUTPUT_SIZE, neg)
                clss.append(c)
                deltas.append(d)
                delta_weights.append(dw)
                search.append(s)
                bboxes.append(np.array(b))
                
            template = template.transpose((2, 0, 1)).astype(np.float32)
            cls = np.array(clss).astype(np.int64)
            delta = np.array(deltas).astype(np.float32)
            delta_weight = np.array(delta_weights).astype(np.float32)
            bbox = np.array(bboxes).astype(np.float32)
            search = np.array(search).astype(np.float32)
            search = search.transpose((0, 3, 1, 2)).astype(np.float32)
            # with visual input
            if cfg.DATASET.HM:
                return {
                    # for tracker
                    'template': template,
                    'search': search,
                    'label_cls': cls,
                    'label_loc': delta,
                    'label_loc_weight': delta_weight,
                    'bbox': box_in,
                    # for motion predictor
                    'pre_loc': d_out,
                    'input_loc': d_in,
                    'pred_box': box_out,
                    'latency_para': param,
                    # for center predictor
                    'pred_hm': hm, # center heatmap on 64X64
                    'pred_wh': wh, # wh for each pred frame
                    'ind': ind, # peak index in 64X64
                    'reg': reg, # regression delta based on int center
                    'reg_mask': reg_mask,
                    }
            return {
                    # for tracker
                    'template': template,
                    'search': search,
                    'label_cls': cls,
                    'label_loc': delta,
                    'label_loc_weight': delta_weight,
                    'bbox': box_in,
                    # for predictor
                    'pre_loc': d_out,
                    'input_loc': d_in,
                    'pred_box': box_out,
                    'latency_para': param
                    }
        # w/o visual input
        return {
                    'pre_loc': d_out,
                    'input_loc': d_in,
                    'pred_box': box_out,
                    'latency_para': param
                    }

