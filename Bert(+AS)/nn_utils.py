# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# 为擦除训练集时候，和测试集读取数据
# 读取输入，仅需要属性train_id, x_ind, x_seg, xt_ind, xt_seg, mask, y
def get_batch_input(dataset, bs, idx):
    batch_input = dataset[idx*bs:(idx+1)*bs]
    batch_data = pd.DataFrame.from_dict(batch_input)
    target_fields = ['sid', 'wids', 'wseg', 'input_mask', 'mask','tmask','y']
    batch_input_var = []
    for key in target_fields:
        data = list(batch_data[key].values)
        if key in ['input_mask','mask','tmask']:
            batch_input_var.append(np.array(data, dtype='float32'))
        else:
            batch_input_var.append(np.array(data, dtype='int32'))
    return batch_input_var


# 为最终训练集读取数据
# 读取输入，仅需要属性test_id, x_ind, x_seg, xt_ind, xt_seg, mask, y, amask, avalue
def get_batch_input_final(dataset, bs, idx):
    batch_input = dataset[idx*bs:(idx+1)*bs]
    batch_data = pd.DataFrame.from_dict(batch_input)
    target_fields = ['sid', 'wids', 'wseg', 'input_mask',  'mask','tmask', 'y', 'amask', 'avalue']
    batch_input_var = []
    for key in target_fields:
        data = list(batch_data[key].values)
        if key in ['input_mask', 'mask', 'tmask','amask', 'avalue']:
            batch_input_var.append(np.array(data, dtype='float32'))
        else:
            batch_input_var.append(np.array(data, dtype='int32'))
    return batch_input_var


# 为samme读取数据
# 读取输入，仅需要属性'sid', 'wids', 'tids', 'y', 'pw', 'mask', 'observation_weight'
def get_batch_input_samme(dataset, bs, idx):
    batch_input = dataset[idx*bs:(idx+1)*bs]
    batch_data = pd.DataFrame.from_dict(batch_input)
    target_fields = ['sid', 'wids', 'wseg', 'input_mask', 'mask','tmask','y', 'observation_weight']
    batch_input_var = []
    for key in target_fields:
        data = list(batch_data[key].values)
        if key in ['input_mask','mask','tmask', 'observation_weight']:
            batch_input_var.append(np.array(data, dtype='float32'))
        else:
            batch_input_var.append(np.array(data, dtype='int32'))
    return batch_input_var
