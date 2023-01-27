from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from re import U
from matplotlib import use

import copy
import torch as t
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from pysot.models.predictor.base_predictor import BasePredictor
from pysot.utils.bbox import cxy_wh_2_rect, rect_2_cxy_wh
import time


class LearnBaseV5(BasePredictor):
    def __init__(self, hidden_1, hidden_2, hidden_3, num_input, num_output):
        super(LearnBaseV5, self).__init__()
        self.name = 'learning_baseline5'
        self.num_input = num_input
        self.output_num = num_output
        self.hidden_3 = hidden_3
        self.input_encode = nn.Sequential(
                            nn.Linear(8, hidden_1),
                            nn.LayerNorm(hidden_1),
                            nn.ReLU())        # delta t does not need first BN
        self.temporal = nn.Sequential(
                            nn.Conv1d(hidden_1, hidden_1, 3, padding=1),
                            nn.BatchNorm1d(hidden_1),
                            nn.ReLU())        # delta t does not need first BN
        self.mid_hidden = nn.Sequential(
                            nn.Linear(hidden_1, hidden_2),
                            nn.BatchNorm1d(hidden_2),
                            nn.ReLU())
        self.head = nn.Linear(hidden_2, hidden_3)
        self.pred_head = _get_clones(self.head, num_output)
        self.out_decode = nn.Sequential(
                            nn.LayerNorm(self.hidden_3),
                            nn.ReLU(),
                            nn.Linear(self.hidden_3, 4))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                m.bias.data.zero_()

    def _convert_bbox_pred(self, delta, anchor):
        tar_box = anchor.unsqueeze(1).repeat(1,delta.shape[1],1)
        delta = delta.permute(2, 1, 0).contiguous()
        tar_box = tar_box.permute(2, 1, 0).contiguous()
        output = t.zeros_like(tar_box).cpu()

        output[0, :] = delta[0, :] * tar_box[2, :] + tar_box[0, :]
        output[1, :] = delta[1, :] * tar_box[3, :] + tar_box[1, :]
        output[2, :] = t.exp(delta[2, :]) * tar_box[2, :]
        output[3, :] = t.exp(delta[3, :]) * tar_box[3, :]
        return output.permute(2,1,0)

    def init(self,bbox_init,img):
        self.init_box = rect_2_cxy_wh(bbox_init)
        self.w_img, self.h_img = img.shape[1], img.shape[0]

    def forward(self, in_delta, latency):
        # Calculate input normalize factor
        dm = t.abs(in_delta).div(latency.unsqueeze(-1).unsqueeze(-1)).mean(dim=[1],keepdim=True)

        B, n, _ = in_delta.shape
        speed = in_delta/latency.unsqueeze(1).unsqueeze(1)
        in_delta = t.cat((in_delta, speed), dim=2)
        in_data = self.input_encode(in_delta)
        motion_cue = self.mid_hidden(self.temporal(in_data.permute(0,2,1)).mean(dim=2))
        output = []
        for h in self.pred_head:
            output.append(h(motion_cue).unsqueeze(1))
        out = self.out_decode(t.cat(output, dim=1))
        # Predict normalized delta
        norm_out = out*dm.repeat(1, out.shape[1], 1)
        return norm_out

    @t.no_grad()
    def predict(self, current_fid, tra_data, delta_t):
        in_delta = t.zeros(1, self.num_input, 8).cuda()
        in_dt = t.ones(1, self.num_input, 1).cuda()
        if len(tra_data['delta'])>=self.num_input:
            # last num_frame loc_deltas from tracker
            in_delta[:,:,:4] = t.FloatTensor([tra_data['delta'][str(idx)] for idx in tra_data['fidx'][-(self.num_input):]]).cuda()
            # delta_t for the loc_delta
            in_dt[:,:,:] = t.FloatTensor([[-tra_data['fidx'][-i-1]+tra_data['fidx'][-i]] for i in range(self.num_input,0,-1)]).cuda()
            in_delta[:,:,4:] = in_delta[:,:,:4]/in_dt
        else:
            in_delta[:,-len(tra_data['delta']):,:4] = t.FloatTensor([tra_data['delta'][str(idx)] for idx in tra_data['fidx'][1:]]).cuda()
            in_dt[:,-len(tra_data['delta']):,:] = t.FloatTensor([[-tra_data['fidx'][-i-1]+tra_data['fidx'][-i]] for i in range(len(tra_data['delta']),0,-1)]).cuda()
            in_delta[:,:,4:] = in_delta[:,:,:4]/in_dt
            in_delta[:,:-len(tra_data['delta']),:] = in_delta[:,-len(tra_data['delta']),:].unsqueeze(1).repeat(1,(self.num_input - len(tra_data['delta'])),1)

        # same as forward
        t.cuda.synchronize()
        s = time.time()
        in_data = self.input_encode(in_delta)
        motion_cue = self.mid_hidden(self.temporal(in_data.permute(0,2,1)).mean(dim=2))
        output = []
        for h in self.pred_head[delta_t[0]-1:delta_t[-1]]:
            output.append(h(motion_cue).unsqueeze(1))
        out = self.out_decode(t.cat(output, dim=1))
        # Predict normalized delta
        # Calculate input normalize factor
        dm = t.abs(in_delta[:,:,4:]).mean(dim=[1],keepdim=True)
        norm_loc = out*dm.repeat(1, out.shape[1], 1)
        t.cuda.synchronize()
        e = time.time()
        print('Latency: {}'.format(e-s))
        # input
        # Predict normalized delta
        meta_box = t.from_numpy(tra_data['boxes'][str(current_fid)]).cuda().unsqueeze(0)
        bbox = self._convert_bbox_pred(norm_loc, meta_box)
        # convert dt to future id
        pre_fidx = [(current_fid+int(d)) for d in delta_t]
        return bbox.squeeze(0).numpy(), pre_fidx

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])