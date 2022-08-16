from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse, pickle
import os
import cv2
import torch
import numpy as np
from time import perf_counter

from pysot.core.config import cfg
from pysot.models import get_modelbuilder
from pysot.tracker.tracker_builder import build_tracker_f
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
#from toolkit.utils.region import vot_overlap, vot_float2str


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='DTB70',type=str,
        help='datasets')
parser.add_argument('--datasetroot', default='./testing_dataset/DTB70/',type=str,
        help='datasetsroot')
parser.add_argument('--fps', default=30,type=int,
        help='input frame rate')
# frames to predict
parser.add_argument('--eta', default=2,type=int,
        help='forward predict frames every tracked frame')
# fix or dynamic
parser.add_argument('--dynamic', default=True,
        help='predict range = latest miss + eta')
parser.add_argument('--config', default='experiments/siammask_r50_l3/pre_lb_config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='', type=str,
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
    
        # create model
    model = get_modelbuilder(cfg.PRED.MODE)

        # build tracker
    tracker = build_tracker_f(model)

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    torch.cuda.synchronize()

    # OPE tracking
    for v_idx, video in enumerate(dataset):
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
        # result path
        o_path=os.path.join('results_rt_raw', args.dataset, model_name)
        if not os.path.isdir(o_path):
            os.makedirs(o_path)
        out_path = os.path.join(o_path, video.name + '.pkl')
        if os.path.isfile(out_path):
            print('({:3d}) Video: {:12s} already done!'.format(
            v_idx+1, video.name))
            continue
        video.load_img()
        t_start = perf_counter()
        t_elapsed = 0
        while 1:
            t1 = perf_counter()
            t_elapsed=t1-t_start
            if t_elapsed>t_total:
                break
            # identify latest available frame
            fidx_continous = t_elapsed*args.fps
            fidx = int(np.floor(fidx_continous))
            #if the tracker finishes current frame before next frame comes, continue
            if fidx == last_fidx:
                continue
            tic = cv2.getTickCount()
            (img,gt_bbox)=video[fidx]
            if fidx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                torch.cuda.synchronize()
                t2 = perf_counter()
                t_elapsed=t2-t_start
                timestamps.append(t_elapsed)
                runtime.append(t2-t1)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
                input_fidx.append(fidx)
                last_fidx=fidx
            else:
                # first do prediction
                # range should be slightly bigger than latest skipped frames
                if args.dynamic:
                    latest_mismatch = fidx - last_fidx + args.eta
                else:
                    latest_mismatch = args.eta
                condition = len(tracker.traject['fidx'])>1 if cfg.PRED.TYPE=='KF' else (len(tracker.traject['fidx'])>cfg.TRAIN.NUM_FRAME and (fidx - last_fidx)<cfg.TRAIN.PRE_TARGET)
                if condition:
                    # predictor latest output for future frame
                    pred_boxes, pred_ids = tracker.predict(last_fidx, fidx, latest_mismatch)
                    # predictor latency for updating evaluation results
                    t_elapsed = perf_counter()-t_start
                    # the predictor results is used for correction and evaluation
                    predictor_outputs = tracker.update_pred(pred_boxes, fidx, pred_ids, t_elapsed)
                # tracker output for frame: fidx
                tracker_outputs = tracker.track(img, fidx)
                torch.cuda.synchronize()
                t2 = perf_counter()
                t_elapsed=t2-t_start
                timestamps.append(t_elapsed)
                runtime.append(t2-t1)
                pred_bbox = tracker_outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(tracker_outputs['best_score'])
                input_fidx.append(fidx)
                last_fidx=fidx
            if t_elapsed>t_total:
                break
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

        #save results and run time
        if args.overwrite or not os.path.isfile(out_path):
            # tracker: [l, t, w, h]
            # predictor: [cx, cy, w, h]
            pickle.dump({
                'results_raw_t': pred_bboxes,
                'results_raw_p': predictor_outputs,
                'eta': args.eta,
                'timestamps_t': timestamps,
                'input_fidx_t': input_fidx,
                'runtime_all': runtime,
            }, open(out_path, 'wb'))
        toc /= cv2.getTickFrequency()
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, len(runtime) / sum(runtime)))


if __name__ == '__main__':
    main()
