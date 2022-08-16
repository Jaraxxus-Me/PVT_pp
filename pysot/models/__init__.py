from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.model_builder import ModelBuilder #tracker + predictor
from pysot.models.pred_model_builder import PredModelBuilder # predictive tracker


Builders = {
        'A+B': ModelBuilder,
        'AB': PredModelBuilder,
       }


def get_modelbuilder(name, **kwargs):
    return Builders[name](**kwargs)

