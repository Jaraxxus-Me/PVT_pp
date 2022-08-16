from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, l1_loss, norm_l1_loss, trend_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.predictor import get_predictor

def xy2center(box):
    # x1, y1, x2, y2 to cx, cy, w, h
    return [(box[0]+box[2])/2, (box[1]+box[3])/2, box[2]-box[0], box[3]-box[1]]

def center2xy(box):
    # cx, cy, w, h to x1, y1, x2, y2
    return [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

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

        # build anchor
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchors = self.generate_anchor(self.score_size)
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)

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
                'mask': mask if cfg.MASK.MASK else None
               }

    def predict(self, data):
        # For training eva, inference use model.predictor.predict
        if 'template' in data:
            # use img
            template = data['template'].cuda()
            search = data['search'].cuda()
        input_loc = data['input_loc'].cuda()
        input_latency = data['latency_para'][0].float().cuda()
        # input
        label_loc = data['pre_loc'].cuda()
        input_latency = data['latency_para'][0].float().cuda()
        # dcx/w, dcy/h, log(w'/w), log(h'/h)
        norm_loc = self.predictor(input_loc, input_latency)
        # Predict normalized delta
        meta_box = data['pred_box'].cuda()
        meta_box = meta_box[:,0,0,:]

        out_boxes = self._convert_bbox(norm_loc, meta_box)
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
        if 'template' not in data:
            """ for predictive fine-tune, learning baseline
            """
            input_loc = data['input_loc'].cuda()
            input_latency = data['latency_para'][0].float().cuda()
            # input
            label_loc = data['pre_loc'].cuda()
            # dcx/w, dcy/h, log(w'/w), log(h'/h)
            loc = self.predictor(input_loc, input_latency)
            # loc_loss = norm_l1_loss(loc, label_loc, dc, ds)
            pre_loss = l1_loss(loc, label_loc)
            outputs['pre_loss'] = pre_loss
            outputs['total_loss'] = pre_loss
            return outputs

        """ for original train tracker
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        search_box = data['bbox'].cuda()
        center_boxes = search_box[:,:,0,:]

        zft = self.backbone(template)
        if cfg.MASK.MASK:
            zft = zft[-1]
        if cfg.ADJUST.ADJUST:
            zft = self.neck(zft)
        n = search.shape[1]
        cls_loss = torch.FloatTensor([0.0]).cuda()[0]
        loc_loss = torch.FloatTensor([0.0]).cuda()[0]
        tracked_delta = torch.zeros_like(search_box[:,:,0,:])
        for i in range(n):
            xf = self.backbone(search[:,i])
            if cfg.MASK.MASK:
                zf = zf[-1]
                self.xf_refine = xf[:-1]
                xf = xf[-1]
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
            cls, loc = self.rpn_head(zft, xf)

            # get loss
            soft_cls = self.log_softmax(cls)
            cls_loss += select_cross_entropy_loss(soft_cls, label_cls[:,i].contiguous())
            loc_loss += weight_l1_loss(loc, label_loc[:,i].contiguous(), label_loc_weight[:,i].contiguous())
            outputs['cls_loss'] = cls_loss.div(n)
            outputs['loc_loss'] = loc_loss.div(n)

            B = cls.shape[0]

        """ for predictive fine-tune, learning baseline
        """
        input_loc = data['input_loc'].cuda()
        input_latency = data['latency_para'][0].float().cuda()
        # input
        label_loc = data['pre_loc'].cuda()
        # dcx/w, dcy/h, log(w'/w), log(h'/h)
        loc = self.predictor(input_loc, input_latency)
        # loc_loss = norm_l1_loss(loc, label_loc, dc, ds)
        pre_loss = l1_loss(loc, label_loc)
        outputs['pre_loss'] = pre_loss
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss.div(n) + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss.div(n) + outputs['pre_loss']
        return outputs

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return torch.FloatTensor(anchor).cuda()

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _convert_anchor(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = torch.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = torch.exp(delta[3, :]) * anchor[:, 3]
        return delta
