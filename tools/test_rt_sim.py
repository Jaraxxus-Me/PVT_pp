from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse, pickle
import os
import sys
import cv2
import torch
import numpy as np
from time import perf_counter

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
#from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='UAV20',type=str,
        help='datasets')
parser.add_argument('--datasetroot', default='./testing_dataset/UAV123_20L/',type=str,
        help='datasetsroot')
parser.add_argument('--sim_info', default='./testing_dataset/DTB70_trainval/DTB70_SiamMask_sim.pkl',type=str,
        help='datasetsroot')
parser.add_argument('--fps', default=30,type=int,
        help='input frame rate')
parser.add_argument('--config', default='experiments/siammask_r50_l3/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='pretrained/Mask_R50.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False,action='store_true',
        help='whether visualzie result')
parser.add_argument('--overwrite', default=True,action='store_true',
        help='whether to overwrite existing results')
args = parser.parse_args()

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)
    UAVDTdataset = args.datasetroot
    dataset_root = os.path.join(UAVDTdataset)
    with open(args.sim_info, 'rb') as f_sim:
        sim_info = pickle.load(f_sim)
    
    # use DTB70 latency to run UAV123, UAVDT, UAV20L
    if args.dataset != args.sim_info.split('/')[-1].split('_')[0]:
        seqs = sim_info.keys()
        avg_init = 0
        avg_run = 0
        for seq in seqs:
            avg_init += sim_info[seq]['init_time']
            avg_run += sim_info[seq]['running_time']
        avg_init /= len(seqs)
        avg_run /= len(seqs)
    
    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    torch.cuda.synchronize()

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        o_path=os.path.join('results_rt_raw', args.dataset, model_name)
        if not os.path.isdir(o_path):
            os.makedirs(o_path)
        out_path = os.path.join('results_rt_raw', args.dataset, model_name, video.name + '.pkl')
        if os.path.isfile(out_path):
            print('Sikpping Video: {}'.format(video.name))
            continue
        video.load_img()
        init_time = sim_info[video.name]['init_time'] if args.dataset == args.sim_info.split('/')[-1].split('_')[0] else avg_init
        run_time = sim_info[video.name]['running_time'] if args.dataset == args.sim_info.split('/')[-1].split('_')[0] else avg_run
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        input_fidx = []
        runtime = []
        timestamps = []
        last_fidx = None
        n_frame=len(video)
        t_total = n_frame/args.fps
        t_start = 0
        t_elapsed=0
        while 1:
            if t_elapsed>t_total:
                break
            # identify latest available frame
            fidx_continous = t_elapsed*args.fps
            fidx = int(np.floor(fidx_continous))
            #if the tracker finishes current frame before next frame comes, continue
            if fidx == last_fidx:
                continue
            last_fidx=fidx
            (img,gt_bbox)=video[fidx]
            if fidx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                torch.cuda.synchronize()
                t2 = init_time
                t_elapsed=t2-t_start
                timestamps.append(t_elapsed)
                runtime.append(init_time)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
                input_fidx.append(fidx)
            else:
                outputs = tracker.track(img)
                torch.cuda.synchronize()
                t2 = t2 + run_time
                t_elapsed=t2-t_start
                timestamps.append(t_elapsed)
                runtime.append(run_time)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
                input_fidx.append(fidx)
            if t_elapsed>t_total:
                break

        #save results and run time
        if args.overwrite or not os.path.isfile(out_path):
            pickle.dump({
                'results_raw': pred_bboxes,
                'timestamps': timestamps,
                'input_fidx': input_fidx,
                'runtime': runtime,
            }, open(out_path, 'wb'))
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, len(runtime)/sum(runtime)))
        del video

if __name__ == '__main__':
    main()
