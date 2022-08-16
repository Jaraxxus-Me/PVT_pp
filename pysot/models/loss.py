from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
from .centernet.losses import FocalLoss, RegL1Loss

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)

def l1_loss(pred_loc, label_loc):
    b = pred_loc.shape[0]
    # pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    loss = diff.sum()
    return loss.sum().div(b)

def trend_l1_loss(pred_loc, label_loc):
    b, _, _ = pred_loc.size()
    d = torch.FloatTensor([0.0476, 0.0476, 0.1429, 0.381, 0.381]).cuda()
    diff = (pred_loc - label_loc).abs()
    loss = diff.sum(dim=[0,2])*d*5
    return loss.sum().div(b)

def norm_l1_loss(pred_loc, label_loc, norm_wc, norm_ws):
    b, _, _ = pred_loc.size()
    # pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    norm_loc = torch.zeros_like(label_loc)
    # x y use norm
    norm_loc[:,:,0:2] = label_loc[:,:,0:2].div(norm_wc.unsqueeze(-1).unsqueeze(-1))
    # w h use origin
    norm_loc[:,:,2:4] = label_loc[:,:,2:4].div(norm_ws.unsqueeze(-1).unsqueeze(-1))
    diff = (pred_loc - norm_loc).abs()
    # diff = diff.div(norm_w.unsqueeze(-1).unsqueeze(-1))
    loss = diff.sum()
    return loss.sum().div(b)

class CtdetLoss(torch.nn.Module):
  def __init__(self, cfg):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if cfg.PRED.MSE_LOSS else FocalLoss()
    self.crit_reg = l1_loss
    self.crit_wh = l1_loss
    self.cfg = cfg

  def forward(self, output, batch):
    cfg = self.cfg
    # hm_loss, wh_loss, off_loss = 0, 0, 0
    if not cfg.PRED.MSE_LOSS:
        output['hm'] = _sigmoid(output['hm'])
        hm_loss = self.crit(output['hm'], batch['pred_hm'].cuda())
        wh_loss = self.crit_wh(output['wh'], batch['pred_wh'].cuda())
        off_loss = self.crit_reg(output['reg'], batch['reg'].cuda())
        # for s in range(cfg.TRAIN.PRE_TARGET):
        #     hm_loss += self.crit(output['hm'][:,s:(s+1)], batch['pred_hm'][:,s:(s+1)].cuda())
        #     wh_loss += self.crit_wh(output['wh'][:,2*s:2*(s+1)], batch['reg_mask'][:,:,s].cuda(), batch['ind'][:,:,s].cuda(), batch['pred_wh'][:,:,s].cuda())
        #     off_loss += self.crit_reg(output['reg'][:,2*s:2*(s+1)], batch['reg_mask'][:,:,s].cuda(), batch['ind'][:,:,s].cuda(), batch['reg'][:,:,s].cuda())
        
    loss = cfg.PRED.HM_W * hm_loss + cfg.PRED.WH_W * wh_loss + \
           cfg.PRED.REG_W * off_loss
    loss_stats = {'pred_loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats
