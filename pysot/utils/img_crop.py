from os.path import join, isdir
from os import listdir, mkdir, makedirs
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from concurrent import futures
import sys
import time

VID_base_path = './ILSVRC2015'
ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):

    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0), pix=None):
    # bbox x1, y1, x2, y2
    # 
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = None
    # for pure loc input, img is none
    if image is not None:
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    if pix is not None:
        B = np.mat([[c],
                    [d]]).astype(np.float)
        A = np.mat([[a, 0],
                    [0, b]]).astype(np.float)
        # x1, x2, x1', x2'
        # y1, y2, y1', y2'
        pixels = np.mat([[pix[0][0], pix[0][2],pix[1][0], pix[1][2]],
                            [pix[0][1], pix[0][3],pix[1][1], pix[1][3]]]).astype(np.float)
        # To new axis
        affined_pixels = (np.dot(A, pixels)+B).tolist()
        # calculate delta in affined image patch
        new_x = (affined_pixels[0][3]+affined_pixels[0][2])/2
        new_y = (affined_pixels[1][3]+affined_pixels[1][2])/2
        new_w = affined_pixels[0][3]-affined_pixels[0][2]
        new_h = affined_pixels[1][3]-affined_pixels[1][2]

        pre_x = (affined_pixels[0][1]+affined_pixels[0][0])/2
        pre_y = (affined_pixels[1][1]+affined_pixels[1][0])/2
        pre_w = affined_pixels[0][1]-affined_pixels[0][0]
        pre_h = affined_pixels[1][1]-affined_pixels[1][0]
        # for smooth L1 loss
        dcx = (new_x-pre_x)/pre_w
        dcy = (new_y-pre_y)/pre_h
        dw = np.log(new_w / pre_w)
        dh = np.log(new_h / pre_h)
        delta = [dcx, dcy, dw, dh]
        # affined
        # cx, cy, w, h
        # cx', cy', w', h'
        new_box = [[pre_x, pre_y, pre_w, pre_h], [new_x, new_y, new_w, new_h]]
        return crop, delta, new_box
    return crop

def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x

def crop_temp(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255):
    # Template is a single image, crop center
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    avg_chans = None
    avg_chans = None if (image is None) else np.mean(image, axis=(0, 1))
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, avg_chans)
    return x

def crop_search(images, bboxs, context_amount=0.5, exemplar_size=127, instanc_size=511):
    # Search is consecutive image stream, crop last center
    # Center pixels
    x=[]
    delta=[]
    s_box=[]
    for i in range(len(bboxs)-1):
        c_box = bboxs[i] # crop box location
        actual_box = bboxs[i+1] # actual box location
        image = None if (images is None) else images[i+1]
        # crop using c_box
        target_pos = [(c_box[2]+c_box[0])/2., (c_box[3]+c_box[1])/2.]
        target_size = [c_box[2]-c_box[0], c_box[3]-c_box[1]]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        avg_chans = None if (images is None) else np.mean(image, axis=(0, 1))
        s_x = s_z + 2 * pad
        cur_x, cur_delta, cur_box = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, avg_chans, [c_box, actual_box])
        x.append(cur_x)
        delta.append(cur_delta)
        s_box.append(cur_box)
    return x, delta, s_box

def crop_tar(images, bboxs, context_amount=0.5, exemplar_size=127, instanc_size=511):
    # Search is consecutive image stream, crop last center
    # Center pixels
    x=[]
    delta=[]
    outbox=[]
    for i in range(1, len(bboxs)):
        c_box = bboxs[0] # crop box location
        actual_box = bboxs[i] # actual box location
        image = None if (images is None) else images[i]
        # crop using c_box
        target_pos = [(c_box[2]+c_box[0])/2., (c_box[3]+c_box[1])/2.]
        target_size = [c_box[2]-c_box[0], c_box[3]-c_box[1]]
        wc_z = target_size[1] + context_amount * sum(target_size)
        hc_z = target_size[0] + context_amount * sum(target_size)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        d_search = (instanc_size - exemplar_size) / 2
        pad = d_search / scale_z
        avg_chans = None
        avg_chans = None if (images is None) else np.mean(image, axis=(0, 1))   
        s_x = s_z + 2 * pad
        cur_x, cur_delta, cur_box = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, avg_chans, [c_box, actual_box])
        x.append(cur_x)
        delta.append(cur_delta)
        outbox.append(cur_box)
    return x, delta, outbox


def crop_video(sub_set, video, crop_path, instanc_size):
    video_crop_base_path = join(crop_path, sub_set, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    sub_set_base_path = join(ann_base_path, sub_set)
    xmls = sorted(glob.glob(join(sub_set_base_path, video, '*.xml')))
    for xml in xmls:
        xmltree = ET.parse(xml)
        # size = xmltree.findall('size')[0]
        # frame_sz = [int(it.text) for it in size]
        objects = xmltree.findall('object')
        objs = []
        filename = xmltree.findall('filename')[0].text

        im = cv2.imread(xml.replace('xml', 'JPEG').replace('Annotations', 'Data'))
        avg_chans = np.mean(im, axis=(0, 1))
        for object_iter in objects:
            trackid = int(object_iter.find('trackid').text)
            # name = (object_iter.find('name')).text
            bndbox = object_iter.find('bndbox')
            # occluded = int(object_iter.find('occluded').text)

            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(filename), trackid)), z)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)


def main(instanc_size=511, num_threads=24):
    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)

    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        n_videos = len(videos)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, sub_set, video, crop_path, instanc_size) for video in videos]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)


if __name__ == '__main__':
    since = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
