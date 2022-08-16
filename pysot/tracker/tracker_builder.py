from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siamrpn_tracker_f import SiamRPNTracker_f
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siammask_tracker_f import SiamMaskTracker_f
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }

TRACKSF = {
          'SiamRPNTracker': SiamRPNTracker_f,
          'SiamMaskTracker': SiamRPNTracker_f,
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

def build_tracker_f(model):
    return TRACKSF[cfg.TRACK.TYPE](model)
