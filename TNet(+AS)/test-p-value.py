# -*- coding: utf-8 -*-
import argparse
import math
import time
import os
from layer import TNet
from utils import *
from nn_utils import *
from evals import *
import random

def test_final(a1_name, a2_name, a3_name, a4_name, a5_name, erasing, args):
    # 数据集（训练集，测试集）， 词向量， 功能词向量（暂时没用）， 训练集长度（padding后）， 测试集长度（padding后）
    dataset, embeddings, n_train, n_test = build_dataset(ds_name=args.ds_name, bs=args.bs, dim_w=args.dim_w, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name, erasing_or_final=False)
    
    args.dim_w = len(embeddings[1])
    args.embeddings = embeddings
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    
    n_train_batches = math.ceil(n_train / args.bs) # 训练batch个数
    n_test_batches = math.ceil(n_test / args.bs) # 验证或测试batch个数
    train_set, test_set = dataset
    
    cur_model_name = "TNet-ATT-FINAL-%s" % erasing
    model = TNet(args=args) # 初始化模型
    with open('log/%s/best_model_%s_%s' % (args.log_name, args.ds_name, cur_model_name), 'rb') as f:
        model.load_model(f)

    test_y_pred, test_y_gold = [], []
    test_ids = []
    test_alphas = []
    for j in range(n_test_batches):
        test_id, test_x, test_xt, test_y, test_pw, test_mask = get_batch_input(dataset=test_set, bs=args.bs, idx=j)
        y_pred, y_gold, loss, alpha = model.test(test_x, test_xt, test_y, test_pw, test_mask, np.int32(0))
        test_y_pred.extend(y_pred)
        test_y_gold.extend(y_gold)
        # 存储训练集id和alpha
        test_ids.extend(test_id)
        test_alphas.extend(alpha)
    acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
    print ("Accuracy: %.2f, Macro-f1: %.2f" % (acc * 100, f * 100))
    with open("log/%s/test_set_result_%s_%s" % (args.log_name, args.ds_name, cur_model_name) ,"w") as f:
        for y_pred, y_gold in zip(test_y_pred[:n_test], test_y_gold[:n_test]):
            f.write(str(y_pred) + ' ' + str(y_gold) + '\n')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TNet settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest", help="dataset name") # 数据名：14semeval_laptop，14semeval_rest，Twitter
    parser.add_argument("-bs", type=int, default=64, help="batch size") # batch size大小
    parser.add_argument("-dim_w", type=int, default=300, help="dimension of word embeddings") # 词向量纬度
    parser.add_argument("-dropout_rate", type=float, default=0.3, help="dropout rate for sentimental features") # dropout率
    parser.add_argument("-dim_h", type=int, default=50, help="dimension of hidden state") # 隐藏层纬度
    parser.add_argument("-n_epoch", type=int, default=50, help="number of training epoch") # 训练轮数
    parser.add_argument("-dim_y", type=int, default=3, help="dimension of label space") # 3个类别
    parser.add_argument("-connection_type", type=str, default="AS", help="connection type, only AS and LF are valid") # transformation类别：AS，LF（我们用AS）
    parser.add_argument("-log_name", type=str, default="14semeval_rest", help="dataset name")
    
    args = parser.parse_args()
    args.lamda = 0.1
    
    # 论文里提到的rest数据集batchsize设置为25
    if args.ds_name == '14semeval_rest' or args.ds_name == '14semeval_rest_val':
        args.lamda = 0.5
        args.bs = 25
    '''
    a1_name = None
    a2_name = None
    a3_name = None
    a4_name = None
    a5_name = None

    # baseline
    test_final(a1_name, a2_name, a3_name, a4_name, a5_name, 0, args)

    a1_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-1")
    a2_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-2")
    #a3_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-3")
    #a4_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-4")
    #a5_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-5")

    # our best model
    test_final(a1_name, a2_name, a3_name, a4_name, a5_name, 2, args)
    '''
    # 读取输出文件
    baseline_ouput = open("log/%s/test_set_result_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-FINAL-0") ,"r")
    ourmodel_output = open("log/%s/test_set_result_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-FINAL-2") ,"r")
    baseline_y_pred, baseline_y_gold = [], []
    ourmodel_y_pred, ourmodel_y_gold = [], []
    n_test = 0
    for baseline_line, ourmodel_line in zip(baseline_ouput, ourmodel_output):
        n_test += 1
        baseline_line = baseline_line.strip().split()
        baseline_y_pred.append(int(baseline_line[0]))
        baseline_y_gold.append(int(baseline_line[1]))

        ourmodel_line = ourmodel_line.strip().split()
        ourmodel_y_pred.append(int(ourmodel_line[0]))
        ourmodel_y_gold.append(int(ourmodel_line[1]))

    print ("Number of Test Instances: %d" % n_test)
    # 计算p-value，采样1000次
    total_hit_acc = 0
    total_hit_f = 0
    for i in range(1000):
        this_baseline_y_pred, this_baseline_y_gold = [], []
        this_ourmodel_y_pred, this_ourmodel_y_gold = [], []
        for j in range(n_test):
            n = random.randint(0, n_test - 1)
            this_baseline_y_pred.append(baseline_y_pred[n])
            this_baseline_y_gold.append(baseline_y_gold[n])

            this_ourmodel_y_pred.append(ourmodel_y_pred[n])
            this_ourmodel_y_gold.append(ourmodel_y_gold[n])

        this_baseline_acc, this_baseline_f, _, _ = evaluate(pred=this_baseline_y_pred, gold=this_baseline_y_gold)
        this_ourmodel_acc, this_ourmodel_f, _, _ = evaluate(pred=this_ourmodel_y_pred, gold=this_ourmodel_y_gold)

        if this_ourmodel_acc > this_baseline_acc:
            total_hit_acc += 1
        if this_ourmodel_f > this_baseline_f:
            total_hit_f += 1
    print ("Accurancy p-value: %.4f" % (1 - total_hit_acc / 1000.0))
    print ("Macro-f1 p-value: %.4f" % (1- total_hit_f  / 1000.0))

