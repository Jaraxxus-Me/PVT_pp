from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch as t

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.bbox import cxy_wh_2_rect, rect_2_cxy_wh
# from torchvision.models import resnet50
from thop import profile


class SiamRPNTracker_ntr(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker_ntr, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.input_len = cfg.TRAIN.NUM_FRAME
        self.mode = cfg.PRED.MODE
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _update_delta(self, dx, dy, dw, dh, s=None):
        # actual delta in origin image
        last_w = self.size[0] - dw
        last_h = self.size[1] - dh
        # actual last w h in origin image
        # don't need to conver to 255, the factor will be self-divided
        delta_x = dx/last_w
        delta_y = dy/last_h
        delta_w = np.log(self.size[0] / last_w)
        delta_h = np.log(self.size[1] / last_h)

        self.traject['delta'][str(self.traject['fidx'][-1])] = np.array([delta_x, delta_y, delta_w, delta_h])
        if s != None:
            self.traject['search'][str(self.traject['fidx'][-1])] = s
        if len(self.traject['delta'])>self.input_len:
            del self.traject['delta'][str(self.traject['fidx'][-self.input_len-1])]
            if s != None:
                del self.traject['search'][str(self.traject['fidx'][-self.input_len-1])]

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (l, t, w, h) bbox 
        """
        # cx, cy, w, h
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        # init
        # all boxes: [cx, cy, w, h]
        # results given by tracker module, input of predictor
        self.traject = {'boxes': {}, 'delta': {}, 'search': {}, 'fidx': []}
        # results given by predictor module, crop region for tracker, use for evaluation
        # {'boxes_eva': {'fidx1': box1, 'fidx2': box2, ...}, # for evaluation
        # 'time': {'fidx1': t1, 'fidx2': t2, ...}, # for double check evaluation time
        # 'boxes_co': {'fidx1': box1, 'fidx2': box2, ...}, # for correcting tracker, can be later
        self.pred_boxes = {'boxes_eva': {}, 'time': {}, 'boxes_co': {}}
        # first frame
        self.traject['boxes']['0'] = np.hstack((self.center_pos,self.size))
        # self.traject['delta']['0'] = np.array([0.0, 0.0, 0.0, 0.0])
        self.traject['fidx'].append(0)

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)
        self.traject['template'] = self.model.zf if (cfg.MASK.MASK or ('alexnet' in cfg.BACKBONE.TYPE)) else self.model.zf[-1]
        self.model.predictor.init(bbox, img)

    def track(self, img, fidx):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        # search region predict should happen before track
        # use boxes_co, which is free of time limit
        fidx = str(fidx)
        # do not correct
        # if fidx in self.pred_boxes['boxes_co']:
        #     self.size = np.array([self.pred_boxes['boxes_co'][fidx][2],self.pred_boxes['boxes_co'][fidx][3]])
        #     self.center_pos = np.array([self.pred_boxes['boxes_co'][fidx][0],self.pred_boxes['boxes_co'][fidx][1]])
        
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)

        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / self.scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # real dw, dh
        dw_r = width-self.size[0]
        dh_r = height-self.size[1]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        # store tracker output for predictor input and update 
        self.traject['boxes'][fidx] = np.hstack((self.center_pos,self.size))
        self.traject['fidx'].append(int(fidx))
        if self.mode == 'AB':
            self._update_delta(bbox[0], bbox[1], dw_r, dh_r, outputs['s'])
        else:
            self._update_delta(bbox[0], bbox[1], dw_r, dh_r)
        # assert (len(self.traject['delta'])+1)==len(self.traject['fidx'])
        assert len(self.traject['boxes'])==len(self.traject['fidx'])
        # tracker current output
        # [l, t, w, h]
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }

    def predict(self, latest_fid, curr_fid, latency):
        # latest: latest fid with tracker result
        # curr_fid: start of prediction
        # latency: latest skipped frame num, for prediction range
        if cfg.PRED.TYPE=='KF':
            # For KF, no limit
            target_delta_t = np.arange(curr_fid-latest_fid, curr_fid-latest_fid+latency+1, dtype=np.float32)
        else:
            # For LB, there is maximum limit
            target_delta_t = np.arange(curr_fid-latest_fid, min(curr_fid-latest_fid+latency,self.model.predictor.output_num)+1)
        pred_boxes, pred_fidx = self.model.predictor.predict(latest_fid, self.traject, target_delta_t)
        # input_delta = t.randn(1, 3, 8).cuda()
        # input_latency = t.randn(1,).cuda()
        # input_t = t.randn(1, 256, 7, 7).cuda()
        # input_s = t.randn(1, 3, 256, 31, 31).cuda()
        # clip boundary
        # out_boxes = self._bbox_clip(out_boxes, data['search'].shape[-2:])
        pred_boxes = list(pred_boxes)
        pred_fidx = list(pred_fidx)

        return pred_boxes, pred_fidx
    
    def update_pred(self, boxes, next_fid, fidx, time):
        # next_fid is for search region update, no need to be earlier than frame rate 
        for i in range(len(fidx)):
            # do not correct
            # if fidx[i] == next_fid:
            #     self.pred_boxes['boxes_co'][str(int(fidx[i]))] = boxes[i]
            actual_time = fidx[i]/30
            # pred results is earlier than time stamp
            if time < actual_time:
                if cfg.PRED.TYPE=='KF':
                    # For KF, output is already [l, t, w, h]
                    self.pred_boxes['boxes_eva'][str(int(fidx[i]))] = boxes[i]
                else:
                    # else, convert [cx, cy, w, h] to [l, t, w, h]
                    self.pred_boxes['boxes_eva'][str(int(fidx[i]))] = cxy_wh_2_rect([boxes[i][0], boxes[i][1]], [boxes[i][2], boxes[i][3]])
                self.pred_boxes['time'][str(int(fidx[i]))] = time
        return self.pred_boxes

    
