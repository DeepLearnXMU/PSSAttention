# -*- coding: utf-8 -*-
import argparse
import random

import math
import time
import os
from ori_samme import ModelTrain
from utils import *
from nn_utils import *
from evals import *
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def find_best_index(result_list):
    best = 0
    best_index = -1
    for i, this_result in enumerate(result_list):
        if this_result[0] + this_result[1] > best:
            best = this_result[0] + this_result[1]
            best_index = i
    return best_index

def train_samme(model_idex, observation_weights, args):
    set_seed(args.seed)
    # 数据集（训练集，测试集）， 词向量， 功能词向量（暂时没用）， 训练集长度（padding后）， 测试集长度（padding后）
    dataset,n_train, n_test = build_dataset(data_rate=args.data_rate,erase=args.erase,ds_name=args.ds_name, bs=args.bs, vocab_path=args.pretrained_path,
            a1_name=None, a2_name=None, a3_name=None, a4_name=None, a5_name=None, erasing_or_final=True)

    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    args.is_final=0
    train_set, test_set = dataset
    args.num_example=len(train_set)
    n_train_batches = math.ceil(n_train / args.bs) # 训练batch个数
    n_test_batches = math.ceil(n_test / args.bs) # 验证或者测试batch个数
    cur_model_name = "BERT-%s" % model_idex
    model = ModelTrain(args=args) # 初始化模型

    if observation_weights == None:
        for train_instance in train_set:
            train_instance['observation_weight'] = 1.0
    else:
        for train_instance in train_set:
            train_instance['observation_weight'] = observation_weights[train_instance['sid']]

    result_strings = [] # 存储日志
    result_store_train = [] # 存储训练集记录
    result_store_test = [] # 存储测试集记录
    global_step=0
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        # ---------------training----------------
        # 训练集
        np.random.shuffle(train_set)
        train_y_pred, train_y_gold = [], []
        train_losses = []
        train_ids = []
        train_observation_weights = []
        for j in range(n_train_batches):
            train_id, x_ind, x_seg, input_mask, mask,tmask, y, train_observation_weight = get_batch_input_samme(dataset=train_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, alpha = model.train(x_ind, x_seg, input_mask, mask,tmask, y, train_observation_weight)
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
            # 存储训练集id和alpha
            train_ids.extend(train_id)
            train_observation_weights.extend(train_observation_weight)
            global_step+=1
            if global_step % args.eval_step ==0:
                # ---------------prediction----------------
                # val data or test data
                test_y_pred, test_y_gold = [], []
                test_ids = []
                test_alphas = []
                test_y_softmax = []
                for b in range(n_test_batches):
                    test_id, x_ind, x_seg, input_mask, mask, tmask, y = get_batch_input(dataset=test_set,
                            bs=args.bs, idx=b)
                    y_pred, y_softmax, y_gold, loss, alpha = model.test(x_ind, x_seg, input_mask, mask, tmask, y,  0)
                    test_y_pred.extend(y_pred)
                    test_y_gold.extend(y_gold)
                    test_y_softmax.extend(y_softmax)
                    # 存储验证或测试集id和alpha
                    test_ids.extend(test_id)
                    test_alphas.extend(alpha)
                acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
                result_store_test.append((acc * 100, f * 100, test_ids, test_alphas, test_y_pred, test_y_gold, test_y_softmax))
                end = time.time()
                result_strings.append("In Epoch %s Step %s: accuracy: %.2f, macro-f1: %.2f, cost %f s\n" % (i,global_step, acc * 100, f * 100, end - beg))
                print("In Epoch %s Step %s: accuracy: %.2f, macro-f1: %.2f, cost %f s" % (i,global_step, acc * 100, f * 100, end - beg))
                beg = time.time()
        acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        result_store_train.append((acc * 100, f * 100, train_ids, train_observation_weights, train_y_pred, train_y_gold))

    # best_index_test = result_store_test.index(max(result_store_test))
    best_index_test = find_best_index(result_store_test)
    best_result_test = result_store_test[best_index_test]
    result_strings.append("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f\n" % (best_index_test+1, best_result_test[0], best_result_test[1]))
    # best_index_train = result_store_train.index(max(result_store_train))
    best_index_train = find_best_index(result_store_train)
    best_result_train = result_store_train[best_index_train]
    result_strings.append("Best model in Epoch %s: train accuracy: %.2f, macro-f1: %.2f\n" % (best_index_train+1, best_result_train[0], best_result_train[1]))

    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log'):
        os.mkdir('log')

    print ("".join(result_logs))

    # re-order
    test_y_pred = [0] * len(test_set)
    test_y_gold = [0] * len(test_set)
    test_y_softmax = [0] * len(test_set)
    for (id, pred, gold, y_softmax) in zip(best_result_test[2], best_result_test[4], best_result_test[5], best_result_test[6]):
        test_y_pred[id] = pred
        test_y_gold[id] = gold
        test_y_softmax[id] = y_softmax

    # best_index_test = result_store_test.index(max(result_store_test))
    best_index_test = find_best_index(result_store_test)
    best_result_test = result_store_test[best_index_test]
    result_strings.append("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f\n" % (best_index_test+1, best_result_test[0], best_result_test[1]))
    # best_index_train = result_store_train.index(max(result_store_train))
    best_index_train = find_best_index(result_store_train)
    best_result_train = result_store_train[best_index_train]
    result_strings.append("Best model in Epoch %s: train accuracy: %.2f, macro-f1: %.2f\n" % (best_index_train+1, best_result_train[0], best_result_train[1]))

    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log'):
        os.mkdir('log')
    print ("".join(result_logs))

    # store observation_weights
    best_index_test = int(best_index_test / n_train_batches * args.eval_step)
    print(best_index_test)
    err_m = 0.
    for p, g, ow in zip(result_store_train[best_index_test][4], result_store_train[best_index_test][5], result_store_train[best_index_test][3]):
        if p != g:
            err_m += ow
    err_m /= len(train_set)
    alpha_m = math.log((1 - err_m) / err_m) + math.log(2)

    # re-order and update
    new_observation_weights = [0] * len(train_set)
    for id, p, g, ow in zip(result_store_train[best_index_test][2], result_store_train[best_index_test][4], result_store_train[best_index_test][5], result_store_train[best_index_test][3]):
        if p != g:
            new_observation_weights[id] = ow * math.exp(alpha_m)
        else:
            new_observation_weights[id] = ow
    new_observation_weights = new_observation_weights[:n_train]

    # re-normalize
    new_observation_weights = np.array(new_observation_weights)
    new_observation_weights = list(new_observation_weights / new_observation_weights.sum() * n_train)
    del model
    import gc
    gc.collect()
    return alpha_m, new_observation_weights, test_y_pred[:n_test], test_y_gold[:n_test], test_y_softmax[:n_test]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TNet settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest", help="dataset name") # 数据名：14semeval_laptop，14semeval_rest，Twitter
    parser.add_argument("-bs", type=int, default=16, help="batch size") # batch size大小
    parser.add_argument("-dropout_rate", type=float, default=0.1, help="dropout rate for sentimental features") # dropout率
    parser.add_argument("-dim_h", type=int, default=1024, help="dimension of hidden state") # 隐藏层纬度
    parser.add_argument("-n_epoch", type=int, default=6, help="number of training epoch") # 训练轮数
    parser.add_argument("-warmup_rate", type=float, default=0.1, help="warm up rate for bert warm up") # warmup率
    parser.add_argument("-dim_y", type=int, default=3, help="dimension of label space") # 3个类别
    parser.add_argument("-connection_type", type=str, default="AS", help="connection type, only AS and LF are valid") # transformation类别：AS，LF（我们用AS）
    parser.add_argument("-log_name", type=str, default="14semeval_rest", help="dataset name")
    parser.add_argument("-store_model", type=bool, default=False, help="store model")
    # parser.add_argument("-pretrained_path", type=str, default='uncased_L-12_H-768_A-12', help="pretrained BERT base path")
    parser.add_argument("-pretrained_path", type=str, default='uncased_L-24_H-1024_A-16', help="pretrained BERT large path")
    parser.add_argument("-gradient", type=int, default=0, help="0:don't ust 1:gradient 2:smooth gradient")
    parser.add_argument("-device", default='cuda:0', type=str, help="e.g. cuda:0")

    args = parser.parse_args()
    args.model_file = None
    args.seed = 123456
    args.config_path = os.path.join(args.pretrained_path, 'bert_config.json')
    args.checkpoint_path = os.path.join(args.pretrained_path, 'bert_model.ckpt')
    args.vocab_path = os.path.join(args.pretrained_path, 'vocab.txt')
    args.eval_step=30
    args.lamda = 0.1
    args.data_rate= 1
    if args.ds_name == 'Twitter' or args.ds_name == 'Twitter_val':
        args.eval_step=50
    args.erase = 4

    for j in range(1):

        for i in range(2):
            args.seed = i
            alpha_1, observation_weights_1, test_y_pred_1, test_y_gold_1, test_y_softmax_1 = train_samme(1, None, args)

            alpha_2, observation_weights_2, test_y_pred_2, test_y_gold_2, test_y_softmax_2 = train_samme(2,
                    observation_weights_1,
                    args)

            alpha_3, observation_weights_3, test_y_pred_3, test_y_gold_3, test_y_softmax_3 = train_samme(3,
                    observation_weights_2,
                    args)

            alpha_4, observation_weights_4, test_y_pred_4, test_y_gold_4, test_y_softmax_4 = train_samme(4,
                    observation_weights_3,
                    args)

            alpha_5, observation_weights_5, test_y_pred_5, test_y_gold_5, test_y_softmax_5 = train_samme(5,
                    observation_weights_4,
                    args)

            acc, f, _, _ = evaluate(pred=test_y_pred_1, gold=test_y_gold_1)  # check
            print(
                    '--------------------------------------------------------------------------------------------------------')
            print(alpha_1)
            print(acc)
            print(f)
            print(
                    '--------------------------------------------------------------------------------------------------------')

            acc, f, _, _ = evaluate(pred=test_y_pred_2, gold=test_y_gold_2)  # check
            print(
                    '--------------------------------------------------------------------------------------------------------')
            print(alpha_2)
            print(acc)
            print(f)
            print(
                    '--------------------------------------------------------------------------------------------------------')

            acc, f, _, _ = evaluate(pred=test_y_pred_3, gold=test_y_gold_3)  # check
            print(
                    '--------------------------------------------------------------------------------------------------------')
            print(alpha_3)
            print(acc)
            print(f)
            print(
                    '--------------------------------------------------------------------------------------------------------')

            acc, f, _, _ = evaluate(pred=test_y_pred_4, gold=test_y_gold_4)  # check
            print(
                    '--------------------------------------------------------------------------------------------------------')
            print(alpha_4)
            print(acc)
            print(f)
            print(
                    '--------------------------------------------------------------------------------------------------------')

            acc, f, _, _ = evaluate(pred=test_y_pred_5, gold=test_y_gold_5)  # check
            print(
                    '--------------------------------------------------------------------------------------------------------')
            print(alpha_5)
            print(acc)
            print(f)
            print(
                    '--------------------------------------------------------------------------------------------------------')

            test_y_pred = []
            for p1, p2, p3, p4, p5 in zip(test_y_softmax_1, test_y_softmax_2, test_y_softmax_3, test_y_softmax_4,
                    test_y_softmax_5):
                p = np.argmax(alpha_1 * p1 + alpha_2 * p2 + alpha_3 * p3 + alpha_4 * p4 + alpha_5 * p5)
                test_y_pred.append(p)
            test_y_gold = test_y_gold_5

            acc, f, _, _ = evaluate(pred=test_y_pred, gold=test_y_gold)
            print(
                    '--------------------------------------------------------------------------------------------------------')
            print(acc)
            print(f)
            print(
                    '--------------------------------------------------------------------------------------------------------')
