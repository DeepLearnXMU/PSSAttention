# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.BL_MN
import thumt.models.FINAL_BL_MN

def get_model(name):
    if name == "BL_MN":
        return thumt.models.BL_MN.BL_MN
    elif name == "FINAL_BL_MN":
        return thumt.models.FINAL_BL_MN.FINAL_BL_MN
    else:
        raise LookupError("Unknown model %s" % name)
