import json
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video


def ca():
    
    path='./testing_dataset/DTB70/'
    
    name_list=os.listdir(path)
    name_list.sort()
    
    b=[]
    for i in range(len(name_list)):
        b.append(name_list[i])
    c=[]
    
    for jj in range(len(name_list)):
        imgs=path+str(name_list[jj])+'/img/'
        txt=path+str(name_list[jj])+'/groundtruth_rect.txt'
        bbox=[]
        f = open(txt)               # 返回一个文件对象
        file= f.readlines()
        li=os.listdir(imgs)
        li.sort()
        for ii in range(len(file)):
            li[ii]=name_list[jj]+'/img/'+li[ii]
    
            line = file[ii].strip('\n').split(',')
            
            try:
                line[0]=int(line[0])
            except:
                line[0]=float(line[0])
            try:
                line[1]=int(line[1])
            except:
                line[1]=float(line[1])
            try:
                line[2]=int(line[2])
            except:
                line[2]=float(line[2])
            try:
                line[3]=int(line[3])
            except:
                line[3]=float(line[3])
            bbox.append(line)
            
        if len(bbox)!=len(li):
            print (jj)
        f.close()
        c.append({'attr':[],'gt_rect':bbox,'img_names':li,'init_rect':bbox[0],'video_dir':name_list[jj]})
        
    d=dict(zip(b,c))

    return d

class UAVVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(UAVVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)


class DTB70Dataset(Dataset):
    """
    Args:
        name: dataset name, should be 'UAV123', 'UAV20L'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(DTB70Dataset, self).__init__(name, dataset_root)
        meta_data = ca()

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = UAVVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])


