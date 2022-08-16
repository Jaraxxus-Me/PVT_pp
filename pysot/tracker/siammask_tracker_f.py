from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.siamrpn_tracker_f import SiamRPNTracker_f


class SiamMaskTracker_f(SiamRPNTracker_f):
    def __init__(self, model):
        super(SiamMaskTracker_f, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
            "SiamMaskTracker must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img

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
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
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
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        # store tracker output for predictor input and update 
        self.traject['boxes'][fidx] = np.hstack((self.center_pos,self.size))
        self.traject['fidx'].append(int(fidx))
        self._update_delta()
        assert (len(self.traject['delta'])+1)==len(self.traject['fidx'])
        assert len(self.traject['boxes'])==len(self.traject['fidx'])
        # tracker current output
        # [l, t, w, h]
        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()
        return {
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon,
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

    
