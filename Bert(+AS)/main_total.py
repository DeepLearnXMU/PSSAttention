# -*- coding: utf-8 -*-
import argparse
import random

import math
import time
import os
from bert_model import ModelTrain
from utils import *
from nn_utils import *
from evals import *
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, erasing, args):
    set_seed(args.seed)
    # 数据集（训练集，测试集）， 词向量， 功能词向量（暂时没用）， 训练集长度（padding后）， 测试集长度（padding后）
    dataset,n_train, n_test = build_dataset(data_rate=args.data_rate,erase=args.erase,ds_name=args.ds_name, bs=args.bs, vocab_path=args.pretrained_path, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name, erasing_or_final=True)

    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    args.is_final=0
    train_set, test_set = dataset
    args.num_example=len(train_set)
    n_train_batches = math.ceil(n_train / args.bs) # 训练batch个数
    n_test_batches = math.ceil(n_test / args.bs) # 验证或者测试batch个数
    cur_model_name = "TNet-ATT-%s" % erasing


    model = ModelTrain(args=args) # 初始化模型

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
        train_alphas = []
        for j in range(n_train_batches):
            train_id, x_ind, x_seg, input_mask, mask,tmask, y = get_batch_input(dataset=train_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, alpha = model.train(x_ind, x_seg, input_mask, mask,tmask, y)

            # get smooth grad
            if args.gradient == 1:
                alpha_sum=[]
                for k in range(10):
                    # 2表示测试阶段且需要添加噪声
                    _, _, _, alphai=model.test(x_ind, x_seg, input_mask, mask,tmask, y, 1)
                    alpha_sum.append(alphai.tolist())
                grad=np.mean(np.array(alpha_sum),axis=0)
                grad=(grad-np.min(grad,axis=-1,keepdims=True))*(mask-tmask)
                alpha=np.nan_to_num(grad / np.sum(grad, axis=-1, keepdims=True))

            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
            # 存储训练集id和alpha
            train_ids.extend(train_id)
            train_alphas.extend(alpha)
            global_step+=1
            if global_step % args.eval_step ==0:
                # ---------------prediction----------------
                # val data or test data
                test_y_pred, test_y_gold = [], []
                test_ids = []
                test_alphas = []
                beg=time.time()
                for b in range(n_test_batches):
                    test_id, x_ind, x_seg, input_mask, mask, tmask, y = get_batch_input(dataset=test_set,
                                                                                            bs=args.bs, idx=b)
                    y_pred, y_gold, loss, alpha = model.test(x_ind, x_seg, input_mask, mask, tmask, y,  0)
                    test_y_pred.extend(y_pred)
                    test_y_gold.extend(y_gold)
                    # 存储验证或测试集id和alpha
                    test_ids.extend(test_id)
                    test_alphas.extend(alpha)
                acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
                result_store_test.append((acc * 100, f * 100, test_ids, test_alphas))
                end = time.time()
                result_strings.append("In Epoch %s Step %s: accuracy: %.2f, macro-f1: %.2f, cost %f s\n" % (i,global_step, acc * 100, f * 100, end - beg))
                print("In Epoch %s Step %s: accuracy: %.2f, macro-f1: %.2f, cost %f s" % (i,global_step, acc * 100, f * 100, end - beg))
                beg = time.time()


        acc, f, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        result_store_train.append((acc * 100, f * 100, train_ids, train_alphas, train_y_pred, train_y_gold))
        
        # # ---------------prediction----------------
        # # val data or test data
        # test_y_pred, test_y_gold = [], []
        # test_ids = []
        # test_alphas = []
        # for j in range(n_test_batches):
        #     test_id, x_ind, x_seg, xt_ind, xt_seg, mask,tmask, y = get_batch_input(dataset=test_set, bs=args.bs, idx=j)
        #     y_pred, y_gold, loss, alpha = model.test(x_ind, x_seg, xt_ind, xt_seg, mask,tmask, y, np.int32(0))
        #     test_y_pred.extend(y_pred)
        #     test_y_gold.extend(y_gold)
        #     # 存储验证或测试集id和alpha
        #     test_ids.extend(test_id)
        #     test_alphas.extend(alpha)
        # acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
        # result_store_test.append((acc * 100, f * 100, test_ids, test_alphas))
        # end = time.time()
        # result_strings.append("In Epoch %s: accuracy: %.2f, macro-f1: %.2f, cost %f s\n" % (i, acc * 100, f * 100, end - beg))
        # print("In Epoch %s: accuracy: %.2f, macro-f1: %.2f, cost %f s\n" % (i, acc * 100, f * 100, end - beg))




    max_=0
    best_result_test = result_store_test[0]
    for result in result_store_test:
        if result[0]>max_:
            max_=result[0]
            best_result_test = result

    result_strings.append("Best model : test accuracy: %.2f, macro-f1: %.2f\n" % (best_result_test[0], best_result_test[1]))

    max_ = 0
    best_result_train = result_store_train[0]
    for result in result_store_train:
        if result[0] > max_:
            max_ = result[0]
            best_result_train = result
    # best_result_train = result_store_train[-1]
    result_strings.append("Best model : train accuracy: %.2f, macro-f1: %.2f\n" % (best_result_train[0], best_result_train[1]))

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
        # alpha = alpha.numpy()
        # valid_index = np.where(alpha != 0)[0]
        # if valid_index.size != 0:
        #     index_random = np.random.choice(valid_index)
        #     index_max = alpha.argmax()
        #     alpha[index_random], alpha[index_max] = alpha[index_max], alpha[index_random]
        if int(pred) == int(gold):
            write_alpha_to_file[id] = alpha
        else:
            write_alpha_to_file[id] = -alpha
    np.savetxt("log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, cur_model_name), write_alpha_to_file)

    del model
    import gc
    gc.collect()


def train_final(a1_name, a2_name, a3_name, a4_name, a5_name, erasing, args):
    set_seed(args.seed)
    # 数据集（训练集，测试集）， 词向量， 功能词向量（暂时没用）， 训练集长度（padding后）， 测试集长度（padding后）
    dataset, n_train, n_test = build_dataset(data_rate=args.data_rate,erase=args.erase,ds_name=args.ds_name, bs=args.bs, vocab_path=args.vocab_path, a1_name=a1_name, a2_name=a2_name, a3_name=a3_name, a4_name=a4_name, a5_name=a5_name, erasing_or_final=False)

    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    args.is_final=1

    train_set, test_set = dataset
    args.num_example=len(train_set)
    n_train_batches = math.ceil(n_train / args.bs) # 训练batch个数
    n_test_batches = math.ceil(n_test / args.bs) # 验证或测试batch个数
    
    cur_model_name = "TNet-ATT-FINAL-%s" % erasing

    model = ModelTrain(args=args) # 初始化模型
    
    result_strings = []
    result_store_test = []
    global_step=0
    for i in range(1, args.n_epoch + 1):
        beg = time.time()
        # ---------------training----------------
        np.random.shuffle(train_set)
        train_y_pred, train_y_gold = [], []
        train_losses = []
        train_ids = []
        train_alosses = []
        for j in range(n_train_batches):
            train_id, x_ind, x_seg, input_mask, mask, tmask, y, amask, avalue= get_batch_input_final(dataset=train_set, bs=args.bs, idx=j)
            y_pred, y_gold, loss, aloss = model.train_final(x_ind, x_seg, input_mask, mask,tmask, y, amask, avalue)
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_losses.append(loss)
            train_alosses.append(aloss)
            global_step+=1
            if global_step % args.eval_step==0:
                # ---------------prediction----------------
                test_y_pred, test_y_gold = [], []
                test_ids = []
                test_alphas = []
                for b in range(n_test_batches):
                    test_id, x_ind, x_seg, input_mask, mask, tmask, y = get_batch_input(dataset=test_set,
                                                                                            bs=args.bs, idx=b)
                    y_pred, y_gold, loss, alpha = model.test(x_ind, x_seg, input_mask, mask, tmask, y, 0)
                    test_y_pred.extend(y_pred)
                    test_y_gold.extend(y_gold)
                    # 存储训练集id和alpha
                    test_ids.extend(test_id)
                    test_alphas.extend(alpha)
                acc, f, _, _ = evaluate(pred=test_y_pred[:n_test], gold=test_y_gold[:n_test])
                result_store_test.append((acc * 100, f * 100, test_ids, test_alphas, test_y_pred[:n_test], test_y_gold[:n_test]))
                end = time.time()
                result_strings.append("In Epoch %s Step %s: accuracy: %.2f, macro-f1: %.2f, cost %f s\n" % (i,global_step, acc * 100, f * 100,end - beg))
                beg = time.time()
                # store model
                if args.store_model and acc * 100 == max(result_store_test)[0] and f * 100 == max(result_store_test)[1]:
                    path='log/%s/best_model_%s_%s' % (args.log_name, args.ds_name, cur_model_name)
                    model.save_model(path)
                    result_strings.append("Store model In Epoch %s\n" % i)
                    print("Store model")



    max_ = 0
    for result in result_store_test:
        if result[0] > max_:
            max_ = result[0]
            best_result_test = result
    # best_result_test = result_store_test[-1]
    result_strings.append("Best model: test accuracy: %.2f, macro-f1: %.2f\n" % (best_result_test[0], best_result_test[1]))

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
    for (id, alpha, pred, gold) in zip(best_result_test[2], best_result_test[3], best_result_test[4], best_result_test[5]):
        if int(pred) == int(gold):
            write_alpha_to_file[id] = alpha
        else:
            write_alpha_to_file[id] = -alpha
    np.savetxt("log/%s/test_set_alpha_%s_%s" % (args.log_name, args.ds_name, cur_model_name), write_alpha_to_file)

    del model
    import gc
    gc.collect()

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
    parser.add_argument("-pretrained_path", type=str, default='uncased_L-24_H-1024_A-16', help="pretrained BERT large path")
    parser.add_argument("-gradient", type=int, default=0, help="0:don't ust 1:gradient 2:smooth gradient")
    parser.add_argument("-device", default='cuda:0', type=str, help="e.g. cuda:0")

    args = parser.parse_args()
    args.seed = 123456
    args.config_path = os.path.join(args.pretrained_path, 'bert_config.json')
    args.checkpoint_path = os.path.join(args.pretrained_path, 'bert_model.ckpt')
    args.vocab_path = os.path.join(args.pretrained_path, 'vocab.txt')
    args.eval_step=30
    args.lamda = 0.1
    args.data_rate=1
    if args.ds_name == 'Twitter' or args.ds_name == 'Twitter_val':
        args.eval_step=50


    args.erase = 4

    a1_name = None
    a2_name = None
    a3_name = None
    a4_name = None
    a5_name = None

    for i in range(3):
        args.seed = i
        beg_time = time.time()
        train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 1, args)
        a1_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-1")
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 2, args)
        a2_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-2")
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 3, args)
        a3_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-3")
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 4, args)
        a4_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-4")
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        train_erasing(a1_name, a2_name, a3_name, a4_name, a5_name, 5, args)
        a5_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-5")
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time



        a1_name = None
        a2_name = None
        a3_name = None
        a4_name = None
        a5_name = None

        train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 0, args)
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        a1_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-1")
        train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 1, args)
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        a2_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-2")
        train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 2, args)
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        a3_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-3")
        train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 3, args)
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        a4_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-4")
        train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 4, args)
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time

        a5_name = "log/%s/train_set_alpha_%s_%s" % (args.log_name, args.ds_name, "TNet-ATT-5")
        train_final(a1_name, a2_name, a3_name, a4_name, a5_name, 5, args)
        end_time = time.time()
        print('spend time:%f s' % (end_time - beg_time))
        beg_time = end_time
