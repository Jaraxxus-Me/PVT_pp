'''
Streaming evaluation
Given real-time tracking outputs,
it pairs them with the ground truth.

Note that this script does not need to run in real-time
'''

import argparse, pickle
from os.path import join, isfile
import numpy as np
import sys
import os
from tqdm import tqdm

# the line below is for running in both the current directory 
# and the repo's root directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='./results_rt_raw/UAV20/Raw_pred_sim',type=str,
        help='raw result root')
    parser.add_argument('--tar_root', default='./results_rt/UAV20',type=str,
        help='target result root')
    parser.add_argument('--gtroot',default='./testing_dataset/UAV123_20L/anno', type=str)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    trackers=os.listdir(args.raw_root)
    gt_path=args.gtroot
    if 'DTB70' in gt_path:
        seqs = os.listdir(gt_path)
        gt_list=[]
        for seq in seqs:
            gt_list.append(os.path.join(gt_path, seq, 'groundtruth_rect.txt'))
    else:
        gt_list=os.listdir(gt_path)
        gt_list = [os.path.join(gt_path, i) for i in os.listdir(gt_path) if i.endswith('.txt')]
    for tracker in tqdm(trackers):        
        ra_path=join(args.raw_root,tracker)
        ou_path=join(args.tar_root,tracker)
        if os.path.isdir(ou_path):
            continue
        mismatch = 0
        fps_a=[]
    
        for gt_idx, video in enumerate(gt_list):
            name=video.split('/')[-1][0:-4]
            name_rt=name
            # name=video
            if 'DTB70' in gt_path:
                name=video.split('/')[-2]
                name_rt=name
            if 'UAVDT' in gt_path:
                name_rt=name[0:-3]
            # print('Pairing {:s} output with the ground truth ({:d}/{:d}): {:s}'.format(tracker,len(gt_list),gt_idx,name))
            results = pickle.load(open(join(ra_path, name_rt + '.pkl'), 'rb'))
            gtlen = len(open(join(video)).readlines())
            # use raw results when possible in case we change class subset during evaluation
            tra_results_raw = results.get('results_raw_t', None)
            tra_timestamps = results['timestamps_t']
            pre_results = results.get('results_raw_p', None)
            # assume the init box don't need time to process
            tra_timestamps[0]=0

            run_time = results['runtime_all']
            fps_a.append(len(run_time)/sum(run_time))
            tidx_p1 = 0
            pred_bboxes=[]
            
            for idx in range(gtlen):
                # input frame time, i.e., [0, 0.03, 0.06, 0.09, ...]
                t = (idx - args.eta)/args.fps
                # Can predictor give results?
                if ('boxes_eva' in pre_results.keys()) and (str(idx) in pre_results['boxes_eva'].keys()) and pre_results['time'][str(idx)]<=t:
                    # print('Frame {} use predictor results'.format(idx))
                    pred_bboxes.append(pre_results['boxes_eva'][str(idx)])
                    continue
                else:
                    # which is the tracker's latest result?
                    while tidx_p1 < len(tra_timestamps) and tra_timestamps[tidx_p1] <= t:
                        tidx_p1 += 1
                    # there exists at least one result for eva, i.e., the init box, 0
                    
                    # the latest result given is tidx
                    tidx = tidx_p1 - 1
                    
                    # print('GT time is {:3f}, latest tracker time is {:3f}, matching GT id {:3d} with tracker result'.format(t, tra_timestamps[tidx], idx))
                    pred_bboxes.append(tra_results_raw[tidx])
                
            if not os.path.isdir(ou_path):
                os.makedirs(ou_path)
            result_path = join(ou_path, '{}.txt'.format(name_rt))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
        fps_path = join(ou_path, '{}.txt'.format('Speed'))
        with open(fps_path, 'w') as f:
            f.write(str(sum(fps_a)/len(fps_a)))

if __name__ == '__main__':
    main()
