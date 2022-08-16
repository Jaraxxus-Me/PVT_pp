'''
**New** Streaming evaluation
Given real-time tracking outputs,
it pairs them with the ground truth with different permitted latency.
The output will be results from different permitted latency

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
    parser.add_argument('--raw_root', default='results_rt_raw/DTB70',type=str,
        help='raw result root')
    parser.add_argument('--tar_root', default='results_rt_real/new',type=str,
        help='target result root')
    parser.add_argument('--gtroot',default='testing_dataset/DTB70', type=str)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=1.0, help='eta is the max permitted latency, (fidx + eta)/30 = check time, if tracker_results_time <= checktime, use this to match gt')
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
    for i, tracker in enumerate(trackers):     
        print("Matching tracker: {}, {}/{}".format(tracker, i+1 ,len(trackers)))   
        for eta in tqdm(np.arange(0.0, args.eta+0.01, 0.02)):
            ra_path=join(args.raw_root,tracker)
            ou_path=join(args.tar_root,tracker,str(round(eta, 2)))
            if os.path.isdir(ou_path):
                continue
            if not os.path.isdir(ou_path):
                os.makedirs(ou_path)
            mismatch = 0
            fps_a=[]
            fps_path = join(args.tar_root,tracker, '{}.txt'.format('Speed'))
            if not os.path.isdir(fps_path):
                speed_f = open(fps_path,'w')
                need_fps = True
        
            for gt_idx, video in enumerate(gt_list):
                name=video.split('/')[-1][0:-4]
                # name=video
                name_rt=name[0:-3]
                if 'DTB70' in gt_path:
                    name=video.split('/')[-2]
                    name_rt=name
                # print('Pairing {:s} output with the ground truth ({:d}/{:d}): {:s}'.format(tracker,len(gt_list),gt_idx,name))
                results = pickle.load(open(join(ra_path, name + '.pkl'), 'rb'))
                gtlen = len(open(video).readlines())
                # use raw results when possible in case we change class subset during evaluation
                results_raw = results.get('results_raw', None)
                timestamps = results['timestamps']
                # assume the init box don't need time to process
                timestamps[0]=0
                input_fidx = results['input_fidx']
                run_time = results['runtime']
                if need_fps:
                    speed = 'Seq: {}; Speed: {} \n'.format(name, len(run_time)/sum(run_time))
                    speed_f.write(speed)
                    fps_a.append(len(run_time)/sum(run_time))
                tidx_p1 = 0
                pred_bboxes=[]
                
                for idx in range(gtlen):
                    # input frame time, i.e., [0, 0.03, 0.06, 0.09, ...]
                    # w/ permited latency, input frame time will be: [0+eta, 0.03+eta]
                    t = (idx + eta)/args.fps
                    # which is the latest result?
                    while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t:
                        tidx_p1 += 1
                    # there exists at least one result for eva, i.e., the init box, 0
                    
                    # if tidx_p1 == 0:
                    #     # no output
                    #     miss += 1
                    #     bboxes, scores, labels  = [], [], []
                    #     masks, tracks = None, None
                    
                    # the latest result given is tidx
                    tidx = tidx_p1 - 1
                    
                    # compute gt idx and the fidx where the result comes to obtain mismatch
                    ifidx = input_fidx[tidx]
                    mismatch += idx - ifidx
                    # print('GT time is {:3f}, latest tracker time is {:3f}, matching GT id {:3d} with precessed frame {:3d}'.format(t, timestamps[tidx],idx,ifidx))
                    pred_bboxes.append(results_raw[tidx])

                result_path = join(ou_path, '{}.txt'.format(name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
        speed_f.write("Avg speed: {}".format(sum(fps_a)/len(fps_a)))
        speed_f.close()

if __name__ == '__main__':
    main()
