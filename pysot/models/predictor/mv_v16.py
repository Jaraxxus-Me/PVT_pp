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
from pysot.core.xcorr import xcorr_fast, xcorr_depthwise


class MVV16(BasePredictor):
    def __init__(self, num_input, num_output, dwconv_k, dwconv_id, dwconv_hd, hidden_1, hidden_2, hidden_3):
        super(MVV16, self).__init__()
        # general setting
        self.name = 'MVV16'
        self.num_input = num_input
        self.output_num = num_output
        self.hidden_3 = hidden_3
        # visual encoding layer
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(dwconv_id, dwconv_hd, kernel_size=dwconv_k, bias=False),
                nn.BatchNorm2d(dwconv_hd),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(dwconv_id, dwconv_hd, kernel_size=dwconv_k, bias=False),
                nn.BatchNorm2d(dwconv_hd),
                nn.ReLU(inplace=True),
                )
        self.hidden_conv = nn.Sequential(
                nn.Conv2d(dwconv_hd, dwconv_hd, kernel_size=1, bias=False),
                nn.BatchNorm2d(dwconv_hd),
                nn.ReLU(inplace=True),
                )
        # motion encoding layer
        self.input_encode = nn.Sequential(
                            nn.Linear(8, hidden_1),
                            nn.LayerNorm(hidden_1),
                            nn.ReLU())              
        # temporal interaction layer
        self.interact_m = nn.Sequential(
                            nn.Conv1d(hidden_1, hidden_1, 3, padding=1),
                            nn.BatchNorm1d(hidden_1),
                            nn.ReLU())  
        self.interact_v = nn.Sequential(
                            nn.Conv3d(dwconv_hd, dwconv_hd, (3,3,3), padding=(1, 1, 1)),
                            nn.BatchNorm3d(dwconv_hd),
                            nn.ReLU())
        # predictive decode layer
        self.mid_hidden_m = nn.Sequential(
                            nn.Linear(hidden_1, hidden_2),
                            nn.BatchNorm1d(hidden_2),
                            nn.ReLU())
        self.mid_hidden_v = nn.Sequential(
                            nn.Linear(dwconv_hd, hidden_2),
                            nn.BatchNorm1d(hidden_2),
                            nn.ReLU())
        self.mid_hidden_mv = nn.Sequential(
                            nn.Linear(hidden_1+dwconv_hd, hidden_2),
                            nn.BatchNorm1d(hidden_2),
                            nn.ReLU())
        self.head = nn.Linear(hidden_2, hidden_3)
        self.pred_head_v = _get_clones(self.head, num_output)
        self.pred_head_m = _get_clones(self.head, num_output)
        self.pred_head_mv = _get_clones(self.head, num_output)
        self.out_decode_v = nn.Sequential(
                            nn.LayerNorm(self.hidden_3),
                            nn.ReLU(),
                            nn.Linear(self.hidden_3, 4))
        self.out_decode_m = nn.Sequential(
                            nn.LayerNorm(self.hidden_3),
                            nn.ReLU(),
                            nn.Linear(self.hidden_3, 4))
        self.out_decode_mv = nn.Sequential(
                            nn.LayerNorm(self.hidden_3),
                            nn.ReLU(),
                            nn.Linear(self.hidden_3, 4))
        # localization head
        self.pred_loc_head = nn.Sequential(
                nn.Conv2d(dwconv_hd, dwconv_hd, kernel_size=1, bias=False),
                nn.BatchNorm2d(dwconv_hd),
                nn.ReLU(inplace=True),
                nn.Conv2d(dwconv_hd, 4, kernel_size=1)
                )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

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

    def estimate(self, t_feat, s_feat):
        t_feat = self.conv_kernel(t_feat)
        s_feat = self.conv_search(s_feat)
        similarity = xcorr_depthwise(s_feat, t_feat)
        loc_map = self.pred_loc_head(similarity)
        return loc_map.mean(dim=[2,3])

    def forward(self, t_feat, s_feat, in_delta, latency):
        # Calculate input normalize factor
        dm = t.abs(in_delta).div(latency.unsqueeze(-1).unsqueeze(-1)).mean(dim=[1],keepdim=True)
        B, n, _, _, _ = s_feat.shape
        # visual branch
        t_feat = self.conv_kernel(t_feat)
        t_feat = t_feat.unsqueeze(1).repeat(1,n,1,1,1)
        t_feat = t_feat.flatten(start_dim=0, end_dim=1)
        s_feat = self.conv_search(s_feat.flatten(start_dim=0, end_dim=1))
        similarity = xcorr_depthwise(s_feat, t_feat)
        similarity = self.hidden_conv(similarity)
        similarity = similarity.view(B, n, similarity.shape[1], similarity.shape[2], similarity.shape[3])
        # motion branch
        speed = in_delta/latency.unsqueeze(1).unsqueeze(1)
        in_delta = t.cat((in_delta, speed), dim=2)
        in_data = self.input_encode(in_delta)
        # temporal interaction
        visual_cue = self.interact_v(similarity.permute(0,2,1,3,4)).mean(dim=[2,3,4])
        motion_cue = self.interact_m(in_data.permute(0,2,1)).permute(0,2,1).mean(dim=[1])
        # m decoding
        m_cue = self.mid_hidden_m(motion_cue)
        output_m = []
        for h in self.pred_head_m:
            output_m.append(h(m_cue).unsqueeze(1))
        out_m = self.out_decode_m(t.cat(output_m, dim=1))
        norm_out_m = out_m*dm.repeat(1, out_m.shape[1], 1)
        # v decoding
        v_cue = self.mid_hidden_v(visual_cue)
        output_v = []
        for h in self.pred_head_v:
            output_v.append(h(v_cue).unsqueeze(1))
        out_v = self.out_decode_v(t.cat(output_v, dim=1))
        norm_out_v = out_v*dm.repeat(1, out_v.shape[1], 1)
        # joint decoding
        mv_cue = self.mid_hidden_mv(t.cat([visual_cue, motion_cue], dim=1))
        output_mv = []
        for h in self.pred_head_mv:
            output_mv.append(h(mv_cue).unsqueeze(1))
        out_mv = self.out_decode_mv(t.cat(output_mv, dim=1))
        norm_out_mv = out_mv*dm.repeat(1, out_mv.shape[1], 1)
        return norm_out_m, norm_out_v, norm_out_mv

    @t.no_grad()
    def predict(self, current_fid, tra_data, delta_t):
        # prepare tracker's data for predictor
        s_key = str(tra_data['fidx'][-1])
        t_feat = tra_data['template']
        s_feat = t.zeros(1, self.num_input, tra_data['search'][s_key].shape[1], tra_data['search'][s_key].shape[2], tra_data['search'][s_key].shape[3]).cuda()
        in_delta = t.zeros(1, self.num_input, 8).cuda()
        in_dt = t.ones(1, self.num_input, 1).cuda()
        if len(tra_data['delta'])>=self.num_input:
            # last num_frame loc_deltas from tracker
            in_delta[:,:,:4] = t.FloatTensor([tra_data['delta'][str(idx)] for idx in tra_data['fidx'][-(self.num_input):]]).cuda()
            # last num_frame search feature from tracker
            s_feat = t.stack([tra_data['search'][str(idx)] for idx in tra_data['fidx'][-(self.num_input):]], dim=1)
            # delta_t for the loc_delta
            in_dt[:,:,:] = t.FloatTensor([[-tra_data['fidx'][-i-1]+tra_data['fidx'][-i]] for i in range(self.num_input,0,-1)]).cuda()
            in_delta[:,:,4:] = in_delta[:,:,:4]/in_dt
        else:
            # use all availble delta and search
            in_delta[:,-len(tra_data['delta']):,:4] = t.FloatTensor([tra_data['delta'][str(idx)] for idx in tra_data['fidx'][1:]]).cuda()
            s_feat[:,-len(tra_data['search']):] = t.stack([tra_data['search'][str(idx)] for idx in tra_data['fidx'][1:]],dim=1)
            # dt
            in_dt[:,-len(tra_data['delta']):,:] = t.FloatTensor([[-tra_data['fidx'][-i-1]+tra_data['fidx'][-i]] for i in range(len(tra_data['delta']),0,-1)]).cuda()
            in_delta[:,:,4:] = in_delta[:,:,:4]/in_dt
            # stamp
            # in_delta[:,:-len(tra_data['delta']),:] = in_delta[:,-len(tra_data['delta']),:].unsqueeze(1).repeat(1,(self.num_input - len(tra_data['delta'])),1)
            # s_feat[:,:-len(tra_data['search'])] = s_feat[:,-len(tra_data['search'])].unsqueeze(1).repeat(1,(self.num_input - len(tra_data['search'])),1,1,1)

        # Calculate input normalize factor
        dm = t.abs(in_delta[:,:,4:]).mean(dim=[1],keepdim=True)
        # visual branch
        t_feat = self.conv_kernel(t_feat)
        t_feat = t_feat.unsqueeze(1).repeat(1,self.num_input,1,1,1)
        t_feat = t_feat.flatten(start_dim=0, end_dim=1)
        s_feat = self.conv_search(s_feat.flatten(start_dim=0, end_dim=1))
        similarity = xcorr_depthwise(s_feat, t_feat)
        similarity = self.hidden_conv(similarity)
        similarity = similarity.view(1, self.num_input, similarity.shape[1], similarity.shape[2], similarity.shape[3])
        # motion branch
        in_data = self.input_encode(in_delta)
        # temporal interaction
        visual_cue = self.interact_v(similarity.permute(0,2,1,3,4)).mean(dim=[2,3,4])
        motion_cue = self.interact_m(in_data.permute(0,2,1)).permute(0,2,1).mean(dim=[1])
        # joint decoding
        mv_cue = self.mid_hidden_mv(t.cat([visual_cue, motion_cue], dim=1))
        output_mv = []
        for h in self.pred_head_mv[delta_t[0]-1:delta_t[-1]]:
            output_mv.append(h(mv_cue).unsqueeze(1))
        out_mv = self.out_decode_mv(t.cat(output_mv, dim=1))
        # Predict normalized delta
        norm_loc = out_mv*dm.repeat(1, out_mv.shape[1], 1)
        # convert delta to box
        # input
        meta_box = t.from_numpy(tra_data['boxes'][str(current_fid)]).cuda().unsqueeze(0)
        bbox = self._convert_bbox_pred(norm_loc, meta_box)
        # convert dt to future id
        pre_fidx = [(current_fid+int(d)) for d in delta_t]
        return bbox.squeeze(0).numpy(), pre_fidx

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])