from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.predictor.base_predictor import BasePredictor
# motion predictor
from pysot.models.predictor.kf import KalmanF
from pysot.models.predictor.lb_5 import LearnBaseV5
# visual predictor
from pysot.models.predictor.lbv_5 import VisualBaseV5
# joint predictor
from pysot.models.predictor.mv_v16 import MVV16
from pysot.models.predictor.mv_v16_add import MVV16Add
from pysot.models.predictor.mv_v16_attn import MVV16Attn

Predictors = {
        'KF': KalmanF,
        'LB_v5': LearnBaseV5,
        'LBv_v5':VisualBaseV5,
        'MV_v16': MVV16,
        'MV_v16_add': MVV16Add,
        'MV_v16_attn': MVV16Attn
       }


def get_predictor(name, **kwargs):
    return Predictors[name](**kwargs)

