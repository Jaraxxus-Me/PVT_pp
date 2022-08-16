from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, l1_loss, CtdetLoss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.predictor import get_predictor
from pysot.utils.bbox import cxy_wh_2_rect, rect_2_cxy_wh

from .centernet.decode import ctdet_decode
from .centernet.post_process import ctdet_post_process

def xy2center(box):
    # x1, y1, x2, y2 to cx, cy, w, h
    return [(box[0]+box[2])/2, (box[1]+box[3])/2, box[2]-box[0], box[3]-box[1]]

def center2xy(box):
    # cx, cy, w, h to x1, y1, x2, y2
    return [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]

class PredModelBuilder(nn.Module):
    def __init__(self):
        super(PredModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build predictor
        self.predictor = get_predictor(cfg.PRED.TYPE, 
                                        **cfg.PRED.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

        if cfg.PRED.TYPE == 'CenterPred':
            self.loss = CtdetLoss(cfg)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                's': xf if (cfg.MASK.MASK or ('alexnet' in cfg.BACKBONE.TYPE)) else xf[-1],
                'mask': mask if cfg.MASK.MASK else None
               }

    def predict(self, data):
        # For training eva, inference use model.predictor.predict
        """ for original train tracker, tracking loss
        """
        template = data['template'].cuda()
        search = data['search'].cuda()

        # template feature
        zft = self.backbone(template)
        if cfg.MASK.MASK:
            zft = zft[-1]
        if cfg.ADJUST.ADJUST:
            zft = self.neck(zft)
        n = search.shape[1]
        xfs = []
        for i in range(n):
            xf = self.backbone(search[:,i])
            if cfg.MASK.MASK:
                self.xf_refine = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
                if cfg.MASK.MASK:
                    xfs.append(xf)
                else:
                    xfs.append(xf[-1])
            else:
                xfs.append(xf)

        """ for predictor, use visual feature from tracker, prediction loss
        """
        t_feat = zft if (cfg.MASK.MASK or ('alexnet' in cfg.BACKBONE.TYPE)) else zft[-1]
        s_feat = torch.stack(xfs,dim=1)
        input_loc = data['input_loc'].cuda()
        input_latency = data['latency_para'][0].float().cuda()
        # dcx/w, dcy/h, log(w'/w), log(h'/h)
        loc_m, loc_v, loc_mv = self.predictor(t_feat, s_feat, input_loc, input_latency)
        # Predict normalized delta
        meta_box = data['pred_box'].cuda()
        meta_box = meta_box[:,0,0,:]
        if cfg.PRED.TYPE != 'CenterPred':
            out_boxes = self._convert_bbox(loc_mv, meta_box)
        else:
            out_boxes = self._convert_bbox_center(loc_mv, meta_box)
        # clip boundary
        # out_boxes = self._bbox_clip(out_boxes, data['search'].shape[-2:])
        return out_boxes


    def _convert_bbox(self, delta, anchor):
        tar_box = anchor.unsqueeze(1).repeat(1,delta.shape[1],1)
        delta = delta.permute(2, 1, 0).contiguous()
        tar_box = tar_box.permute(2, 1, 0).contiguous()
        output = torch.zeros_like(tar_box).cpu()

        output[0, :] = delta[0, :] * tar_box[2, :] + tar_box[0, :]
        output[1, :] = delta[1, :] * tar_box[3, :] + tar_box[1, :]
        output[2, :] = torch.exp(delta[2, :]) * tar_box[2, :]
        output[3, :] = torch.exp(delta[3, :]) * tar_box[3, :]
        return output

    def _convert_bbox_center(self, output, anchor):
        with torch.no_grad():
            pred_len = cfg.TRAIN.PRE_TARGET
            tar_box = anchor.unsqueeze(1).repeat(1,cfg.TRAIN.PRE_TARGET,1)
            out = torch.zeros_like(tar_box).cpu()
            for i in range(pred_len):
                hm = output['hm'][:,i:(i+1)].sigmoid_()
                wh = output['wh'][:,i:(i+1)].sigmoid_()*cfg.PRED.OUTPUT_SZ
                reg = output['reg'][:,i:(i+1)].sigmoid_()
                out[:, i, :] = ctdet_decode(hm, wh, reg=reg, K=1).squeeze()
            out=out.numpy()
            meta = {'c': np.array([cfg.TRAIN.SEARCH_SIZE / 2., cfg.TRAIN.SEARCH_SIZE / 2.], dtype=np.float32), 's': np.array([cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE], dtype=np.float32), 
            'out_height': cfg.PRED.OUTPUT_SZ, 
            'out_width': cfg.PRED.OUTPUT_SZ}
            # BXnX4[x1, y1, x2, y2]
            out_boxes = ctdet_post_process(out, meta['c'], meta['s'], meta['out_height'], meta['out_width'], pred_len)
        out_boxes=torch.FloatTensor(out_boxes)
        out = torch.zeros_like(tar_box).cpu()
        out[:,:,0] = (out_boxes[:,:,0]+out_boxes[:,:,2])/2.0
        out[:,:,1] = (out_boxes[:,:,1]+out_boxes[:,:,3])/2.0
        out[:,:,2] = out_boxes[:,:,2]-out_boxes[:,:,0]
        out[:,:,3] = out_boxes[:,:,3]+out_boxes[:,:,1]
        return out

    def _bbox_clip(self, box, boundary):
        output = torch.zeros_like(box).cpu()
        output[0] = max(0, min(box[0], boundary[0]))
        output[1] = max(0, min(box[1], boundary[1]))
        output[2] = max(10, min(box[2], boundary[0]))
        output[3] = max(10, min(box[3], boundary[1]))
        return output


    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        outputs={}

        """ for original train tracker, tracking loss
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_pre_loc = data['input_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # template feature
        zft = self.backbone(template)
        if cfg.MASK.MASK:
            zft = zft[-1]
        if cfg.ADJUST.ADJUST:
            zft = self.neck(zft)
        n = search.shape[1]
        xfs = []
        cls_loss = torch.FloatTensor([0.0]).cuda()[0]
        loc_loss = torch.FloatTensor([0.0]).cuda()[0]
        loc_pre_loss = torch.FloatTensor([0.0]).cuda()[0]
        for i in range(n):
            zf = zft
            xf = self.backbone(search[:,i])
            if cfg.MASK.MASK:
                self.xf_refine = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
                if cfg.MASK.MASK:
                    xfs.append(xf)
                else:
                    xfs.append(xf[-1])
            else:
                xfs.append(xf)
            cls, loc= self.rpn_head(zf, xf)
            if cfg.MASK.MASK or ('alexnet' in cfg.BACKBONE.TYPE):
                loc_pre = self.predictor.estimate(zf, xf)
            else:
                loc_pre = self.predictor.estimate(zf[-1], xf[-1])

            # get loss
            cls = self.log_softmax(cls)
            cls_loss += select_cross_entropy_loss(cls, label_cls[:,i].contiguous())
            loc_loss += weight_l1_loss(loc, label_loc[:,i].contiguous(), label_loc_weight[:,i].contiguous())
            loc_pre_loss += l1_loss(loc_pre, label_pre_loc[:,i]) # use delta loss

        outputs['cls_loss'] = cls_loss.div(n)
        outputs['loc_loss'] = loc_loss.div(n)
        outputs['loc_loss_pre'] = cfg.TRAIN.LOC_WEIGHT_PRE * loc_pre_loss.div(n)
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * outputs['cls_loss'] + \
            cfg.TRAIN.LOC_WEIGHT * outputs['loc_loss'] + \
            cfg.TRAIN.LOC_WEIGHT_PRE * outputs['loc_loss_pre']

        # if cfg.MASK.MASK:
        #     # TODO
        #     mask, self.mask_corr_feature = self.mask_head(zf, xf)
        #     mask_loss = None
        #     outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
        #     outputs['mask_loss'] = mask_loss

        """ for predictor, use visual feature from tracker, prediction loss
        """
        t_feat = zft if (cfg.MASK.MASK or ('alexnet' in cfg.BACKBONE.TYPE)) else zft[-1]
        s_feat = torch.stack(xfs,dim=1)
        input_loc = data['input_loc'].cuda()
        input_latency = data['latency_para'][0].float().cuda()
        # input
        label_loc = data['pre_loc'].cuda()
        # dcx/w, dcy/h, log(w'/w), log(h'/h)
        loc_m, loc_v, loc_mv = self.predictor(t_feat, s_feat, input_loc, input_latency)
        # loc_loss = norm_l1_loss(loc, label_loc, dc, ds)
        if cfg.PRED.TYPE != 'CenterPred':
            m_loss = l1_loss(loc_m, label_loc) # use delta loss
            v_loss = l1_loss(loc_v, label_loc) # use delta loss
            mv_loss = l1_loss(loc_mv, label_loc) # use delta loss
            outputs['pred_loss_m'] = m_loss
            outputs['pred_loss_v'] = v_loss
            outputs['pred_loss_mv'] = mv_loss
            outputs['pred_loss'] = cfg.PRED.M_WEIGHT*m_loss+cfg.PRED.V_WEIGHT*v_loss+cfg.PRED.MV_WEIGHT*mv_loss
        else:
            joint_loss, pred_loss_dic = self.loss(loc, data)
            outputs.update(pred_loss_dic)
        outputs['total_loss'] += outputs['pred_loss']
        return outputs
