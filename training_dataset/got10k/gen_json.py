from __future__ import unicode_literals
import json
from os.path import join, exists
import os
import pandas as pd
from tqdm import tqdm

dataset_path = 'data'
train_sets = ['GOT-10k_Train_split_01','GOT-10k_Train_split_02','GOT-10k_Train_split_03','GOT-10k_Train_split_04',
            'GOT-10k_Train_split_05','GOT-10k_Train_split_06','GOT-10k_Train_split_07','GOT-10k_Train_split_08',
            'GOT-10k_Train_split_09','GOT-10k_Train_split_10','GOT-10k_Train_split_11','GOT-10k_Train_split_12',
            'GOT-10k_Train_split_13','GOT-10k_Train_split_14','GOT-10k_Train_split_15','GOT-10k_Train_split_16',
            'GOT-10k_Train_split_17','GOT-10k_Train_split_18','GOT-10k_Train_split_19']
val_set = ['val']
d_sets = {'videos_val':val_set,'videos_train':train_sets}


def parse_and_sched(dl_dir='.'):
    js = {}
    videos = os.listdir(dataset_path)
    for video in tqdm(videos):
        if video == 'list.txt':
            continue
        gt_path = join(dataset_path, video, 'groundtruth.txt')
        f = open(gt_path, 'r')
        groundtruth = f.readlines()
        f.close()
        for idx, gt_line in enumerate(groundtruth):
            gt_image = gt_line.strip().split(',')
            frame = '%08d' % (int(idx+1))
            obj = '%02d' % (int(0))
            bbox = [int(float(gt_image[0])), int(float(gt_image[1])),
                    int(float(gt_image[0])) + int(float(gt_image[2])),
                    int(float(gt_image[1])) + int(float(gt_image[3]))]  # xmin,ymin,xmax,ymax
            
            x1 = bbox[0]
            y1 = bbox[1]
            w = bbox[2] - x1
            h = bbox[3] - y1
            if x1 < 0 or y1 < 0 or w <= 0 or h <= 0:
                break

            if video not in js:
                js[video] = {}
            if obj not in js[video]:
                js[video][obj] = {}
            js[video][obj][frame] = bbox
    json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)

    # print(d_set+': All videos downloaded' )


if __name__ == '__main__':
    parse_and_sched()