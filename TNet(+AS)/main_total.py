# -*- coding: utf-8 -*-
import argparse
import math
import time
import os
from layer import TNet
from utils import *
from nn_utils import *
from evals import *

def train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, erasing, args):
    # 数据集（训练集，测试集）， 词向量， 功能词向量（暂时没用）， 训练集长度（padding后）， 测试集长度（padding后）
    dataset, embeddings, n_train, n_test = build_dataset(ds_name=args.ds_name, bs=args.bs, dim_w=args.dim_w, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name, erasing_or_final=True)

    args.dim_w = len(embeddings[1])
    args.embeddings = embeddings
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])

    n_train_batches = math.ceil(n_train / args.bs) # 训练batch个数
    n_test_batches = math.ceil(n_test / args.bs) # 验证或者测试batch个数
    train_set, test_set = dataset
    
    cur_model_name = "TNet-ATT-%s" % erasing
    model = TNet(args=args) # 初始化模型

    result_strings = [] # 存储日志
    result_store_train = [] # 存储训练集记录
    result_store_test = [] # 存储测试集记录
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        # ---------------training----------------
        # 训练集
        np.random.shuffle(train_set)
        train_y_pred, train_y_gold = [], []
        train_losses = []
        train_ids = []
        train_alphas = []
        for j in range(n_train_batches):
            train_id, train_x, train_xt, train_y, train_pw, train_mask = get_batch_input(dataset=train_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, alpha = model.train(train_x, train_xt, train_y, train_pw, train_mask, np.int32(1))
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
            # 存储训练集id和alpha
            train_ids.extend(train_id)
            train_alphas.extend(alpha)
        acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        result_store_train.append((acc * 100, f * 100, train_ids, train_alphas, train_y_pred, train_y_gold))
        
        # ---------------prediction----------------
        # val data or test data
        test_y_pred, test_y_gold = [], []
        test_ids = []
        test_alphas = []
        for j in range(n_test_batches):
            test_id, test_x, test_xt, test_y, test_pw, test_mask = get_batch_input(dataset=test_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, alpha = model.test(test_x, test_xt, test_y, test_pw, test_mask, np.int32(0))
            test_y_pred.extend(y_pred)
            test_y_gold.extend(y_gold)
            # 存储验证或测试集id和alpha
            test_ids.extend(test_id)
            test_alphas.extend(alpha)
        acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
        result_store_test.append((acc * 100, f * 100, test_ids, test_alphas))
        end = time.time()
        result_strings.append("In Epoch %s: accuracy: %.2f, macro-f1: %.2f, cost %f s\n" % (i, acc * 100, f * 100, end - beg))
        # store model
        #if args.store_model and acc * 100 == max(result_store_test)[0] and f * 100 == max(result_store_test)[1]:
        #    with open('log/%s/best_model_%s_%s' % (args.log_name, args.ds_name, cur_model_name), 'wb') as f:
        #        model.save_model(f)
        #    result_strings.append("Store model In Epoch %s\n" % i)
    
    best_index_test = result_store_test.index(max(result_store_test))
    best_result_test = result_store_test[best_index_test]
    result_strings.append("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f\n" % (best_index_test+1, best_result_test[0], best_result_test[1]))
    best_index_train = result_store_train.index(max(result_store_train))
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

    
    # store train alpha
    write_alpha_to_file = np.zeros((len(train_set), args.sent_len))
    for (id, alpha, pred, gold) in zip(best_result_train[2], best_result_train[3], best_result_train[4], best_result_train[5]):
        if int(pred) == int(gold):
            write_alpha_to_file[id] = alpha
        else:
            write_alpha_to_file[id] = -alpha
    np.savetxt("log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, cur_model_name), write_alpha_to_file)


def train_final(a1_name, a2_name, a3_name, a4_name, a5_name, erasing, args):
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
    
    result_strings = []
    result_store_test = []
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        # ---------------training----------------
        np.random.shuffle(train_set)
        train_y_pred, train_y_gold = [], []
        train_losses = []
        train_ids = []
        train_alosses = []
        for j in range(n_train_batches):
            train_id, train_x, train_xt, train_y, train_pw, train_mask, train_amask, train_avalue = get_batch_input_final(dataset=train_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, aloss = model.train_final(train_x, train_xt, train_y, train_pw, train_mask, train_amask, train_avalue, np.int32(1))
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
            train_alosses.append(aloss)
        acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        # ---------------prediction----------------
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
        result_store_test.append((acc * 100, f * 100, test_ids, test_alphas))
        end = time.time()
        result_strings.append("In Epoch %s: accuracy: %.2f, macro-f1: %.2f, cost %f s\n" % (i, acc * 100, f * 100, end - beg))
        # store model
        #if args.store_model and acc * 100 == max(result_store_test)[0] and f * 100 == max(result_store_test)[1]:
        #    with open('log/%s/best_model_%s_%s' % (args.log_name, args.ds_name, cur_model_name), 'wb') as f:
        #        model.save_model(f)
        #    result_strings.append("Store model In Epoch %s\n" % i)

    best_index_test = result_store_test.index(max(result_store_test))
    best_result_test = result_store_test[best_index_test]
    result_strings.append("Best model in Epoch %s: test accuracy: %.2f, macro-f1: %.2f\n" % (best_index_test+1, best_result_test[0], best_result_test[1]))

    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log'):
        os.mkdir('log')
    print ("".join(result_logs))

    # store alpha
    write_alpha_to_file = np.zeros((len(test_set), len(test_set[0]['wids'])))
    for (id, alpha) in zip(best_result_test[2], best_result_test[3]):
        write_alpha_to_file[id] = alpha
    np.savetxt("log/%s/test_set_alpha_%s_%s" % (args.log_name, args.ds_name, cur_model_name), write_alpha_to_file)


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
    parser.add_argument("-store_model", type=bool, default=False, help="store model")


    args = parser.parse_args()
    args.lamda = 0.1
    
    # 论文里提到的rest数据集batchsize设置为25
    if args.ds_name == '14semeval_rest' or args.ds_name == '14semeval_rest_val':
        args.lamda = 0.5
        args.bs = 25
    
    a1_name = None
    a2_name = None
    a3_name = None
    a4_name = None
    a5_name = None

    
    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 1, args)
    a1_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-1")
    
    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 2, args)
    a2_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-2")
    
    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 3, args)
    a3_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-3")

    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 4, args)
    a4_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-4")

    train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 5, args)
    a5_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-5")
    
    
    a1_name = None
    a2_name = None
    a3_name = None
    a4_name = None
    a5_name = None

    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 0, args)

    a1_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-1")
    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 1, args)

    a2_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-2")
    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 2, args)

    a3_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-3")
    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 3, args)

    a4_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-4")
    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 4, args)

    a5_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-5")
    train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 5, args)
    





