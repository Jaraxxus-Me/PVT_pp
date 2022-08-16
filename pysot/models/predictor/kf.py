from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torch.nn.functional as F
from pysot.models.predictor.base_predictor import BasePredictor
from pysot.utils.bbox import cxy_wh_2_rect, rect_2_cxy_wh
import pycocotools.mask as maskUtils


class KalmanF(BasePredictor):
    def __init__(self):
        '''
        KalmanF takes box [l, t, w, h] as input and output!!
        '''
        super(KalmanF, self).__init__()
        self.kf_P = torch.empty((0, 8, 8), dtype=torch.float64)
        self.kf_F = torch.eye(8, dtype=torch.float64)
        self.kf_Q = torch.eye(8, dtype=torch.float64)
        self.kf_R = 10*torch.eye(4, dtype=torch.float64)
        self.kf_P_init = 100*torch.eye(8, dtype=torch.float64).unsqueeze(0)
        self.tidx = 0
        self.updated=False
        self.tracks = np.arange(self.tidx, self.tidx+1, dtype=np.uint32)
        self.tidx+=1
        self.t=0
    
    def init(self, bbox, img):
        bbox_init = np.array([bbox])
        self.kf_x = self.bbox2x(bbox_init)
        self.kf_P = self.kf_P_init.expand(len(bbox_init), -1, -1)
        self.w_img, self.h_img = img.shape[1], img.shape[0]

    def predict(self, fidx_curr, tra_data, delta_t):
        # latest tracker results
        assert fidx_curr == tra_data['fidx'][-1]
        box_tra = tra_data['boxes'][str(fidx_curr)]
        pre_fidx = [int(fidx_curr + d) for d in delta_t]
        # [cx, cy, w, h]
        center_loc = box_tra[0:2]
        size = box_tra[2:]
        # 2 [l, t, w, h]
        box_curr = np.array([cxy_wh_2_rect(center_loc, size)])
        # find latest result to update KF
        dt = fidx_curr-self.t
        dt=int(dt)
        self.kf_F = self.make_F(self.kf_F, dt)
        self.kf_Q = self.make_Q(self.kf_Q, dt)
        self.kf_x, self.kf_P = self.batch_kf_predict(self.kf_F, self.kf_x, self.kf_P, self.kf_Q)
        self.bboxes_f = self.x2bbox(self.kf_x) #bboxes_f is the forecast result
        # update KF time
        self.t = fidx_curr
        self.bboxes_t2 = box_curr
        self.updated=False
        #association based on IoU match
        order1, order2, n_matched12, self.tracks, self.tkidx = self.iou_assoc(
                                self.bboxes_f, self.tracks, self.tidx,
                                self.bboxes_t2, 0.3,
                                no_unmatched1=True,
                            )
        # If match, update x, P in KF with box_curr
        if n_matched12:
            self.kf_x = self.kf_x[order1]
            self.kf_P = self.kf_P[order1]
            self.kf_x, kf_P = self.batch_kf_update(
                self.bbox2z(self.bboxes_t2[order2[:n_matched12]]),
                self.kf_x,
                self.kf_P,
                self.kf_R,
            )
    
            kf_x_new = self.bbox2x(self.bboxes_t2[order2[n_matched12:]])
            n_unmatched2 = len(self.bboxes_t2) - n_matched12
            kf_P_new = self.kf_P_init.expand(n_unmatched2, -1, -1)
            self.kf_x = torch.cat((self.kf_x, kf_x_new))
            self.kf_P = torch.cat((self.kf_P, kf_P_new))
            self.updated = True
        if not self.updated:
            self.kf_x = self.bbox2x(self.bboxes_t2)
            self.kf_P = self.kf_P_init.expand(len(self.bboxes_t2), -1, -1)
            self.tracks = np.arange(self.tidx, self.tidx + 1, dtype=np.uint32)
            self.tkidx += 1
        # Forecast to fidx_t with updated KF
        bbox=[]
        for dt in delta_t:
            # PyTorch small matrix multiplication is slow
            # use numpy instead
            kf_x_np = self.kf_x[:, :, 0].numpy()
            bboxes_t3 = kf_x_np[:n_matched12, :4] + dt*kf_x_np[:n_matched12, 4:]
            if n_matched12 < len(self.kf_x):
                bboxes_t3 = np.concatenate((bboxes_t3, kf_x_np[n_matched12:, :4]))
            
            bboxes_t3, keep = self.extrap_clean_up(bboxes_t3, self.w_img, self.h_img, lt=True)
            if len(bboxes_t3):
                bbox.append(bboxes_t3[0])
            else :
                bbox.append(box_curr[0])
            # Predictor should output [l, t, w, h]
        return bbox, pre_fidx
        
    
    def bbox2z(self,bboxes):
        return torch.from_numpy(bboxes).unsqueeze_(2)

    def bbox2x(self,bboxes):
        x = torch.cat((torch.from_numpy(bboxes), torch.zeros(bboxes.shape, dtype=torch.float64)), dim=1)
        return x.unsqueeze_(2)
    
    def x2bbox(self,x):
        return x[:, :4, 0].numpy()
    
    def make_F(self,F, dt):
        F[[0, 1, 2, 3], [4, 5, 6, 7]] = dt
        return F.double()
    
    def make_Q(self,Q, dt):
        # assume the base Q is identity
        Q[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]] = dt*dt
        return Q.double()
    
    def batch_kf_predict_only(self,F, x):
        return F @ x
    
    def batch_kf_predict(self,F, x, P, Q):
        x = F @ x
        P = F @ P.double() @ F.t() + Q
        return x.double(), P.double()
    
    def batch_kf_update(self,z, x, P, R):
        # assume H is just slicing operation
        # y = z - Hx
        y = z - x[:, :4]
    
        # S = HPH' + R
        S = P[:, :4, :4] + R
    
        # K = PH'S^(-1)
        K = P[:, :, :4] @ S.inverse()
    
        # x = x + Ky
        x += K @ y
    
        # P = (I - KH)P
        P -= K @ P[:, :4]
        return x.double(), P.double()
    
    def iou_assoc(self,bboxes1, tracks1, tkidx, bboxes2, match_iou_th, no_unmatched1=False):
        # iou-based association
        # shuffle all elements so that matched stays in the front
        # bboxes are in the form of a list of [l, t, w, h]
        m, n = len(bboxes1), len(bboxes2)
            
        _ = n*[0]
        ious = maskUtils.iou(bboxes1, bboxes2, _)
    
        match_fwd = m*[None]
        matched1 = []
        matched2 = []
        unmatched2 = []
    
        for j in range(n):
            best_iou = match_iou_th
            match_i = None
            for i in range(m):
                if match_fwd[i] is not None or ious[i, j] < best_iou:
                    # or labels1[i] != labels2[j] \
                    continue
                best_iou = ious[i, j]
                match_i = i
            if match_i is None:
                unmatched2.append(j)
            else:
                matched1.append(match_i)
                matched2.append(j)
                match_fwd[match_i] = j
    
        if no_unmatched1:
            order1 = matched1
        else:
            unmatched1 = list(set(range(m)) - set(matched1))
            order1 = matched1 + unmatched1
        order2 = matched2 + unmatched2
    
        n_matched = len(matched2)
        n_unmatched2 = len(unmatched2)
        tracks2 = np.concatenate((tracks1[order1][:n_matched],
            np.arange(tkidx, tkidx + n_unmatched2, dtype=tracks1.dtype)))
        tkidx += n_unmatched2
    
        return order1, order2, n_matched, tracks2, tkidx
    
    def extrap_clean_up(self,bboxes, w_img, h_img, min_size=75, lt=False):
        # bboxes in the format of [cx or l, cy or t, w, h]
        wh_nz = bboxes[:, 2:] > 0
        keep = np.logical_and(wh_nz[:, 0], wh_nz[:, 1])
    
        if lt:
            # convert [l, t, w, h] to [l, t, r, b]
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
        else:
            # convert [cx, cy, w, h] to [l, t, r, b]
            bboxes[:, :2] = bboxes[:, :2] - bboxes[:, 2:]/2
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    
        # clip to the image
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w_img)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h_img)
    
        # convert [l, t, r, b] to [l, t, w, h]
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
    
        # int conversion is neccessary, otherwise, there are very small w, h that round up to 0
        keep = np.logical_and(keep, bboxes[:, 2].astype(np.int)*bboxes[:, 3].astype(np.int) >= min_size)
        bboxes = bboxes[keep]
        return bboxes, keep