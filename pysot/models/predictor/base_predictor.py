from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from pysot.core.config import cfg
import torch
import torch.nn as nn


class BasePredictor(nn.Module):
    """ Base predictor for prediction
    """
    def init(self, box_init, img_0):
        """
        args:
            box_init(np.ndarray): [l, t, w, h]
            img_0(np.ndarray): BGR image
        """
        raise NotImplementedError

    def predict(self, curr_fid, data, delta_t):
        """
        args:
            curr_fid(int): latest processed frame (base frame)
            data(dict): output of tracker
            delta_t(list/ndarray): target delta_t for prediction (target frame)
        return:
            bbox(list/ndarray): predicted boxes [[cx, cy, w, h]_1, [cx, cy, w, h]_2, ...]
            pre_fidx(list/ndarray): future frame id for predicted boxes [fidx_1, fidx_2, ...]
        """
        raise NotImplementedError