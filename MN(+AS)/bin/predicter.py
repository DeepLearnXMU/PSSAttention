# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import six

import numpy as np
import tensorflow as tf
import thumt.data.cache as cache
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.hooks as hooks
import thumt.utils.inference as inference
import thumt.utils.optimize as optimize
import thumt.utils.parallel as parallel
import itertools

from decimal import *

def float2int(num):
    num = str(Decimal(str(num)).quantize(Decimal('0.0000')))
    return "{name: >10s}".format(name=num)

def int2int(num):
    num = str(int(num))
    return "{name: >10s}".format(name=num)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Using Recurrent Neural Networks Approach for Estimating the Quality of Machine Translation to predict translations' scores",
        usage="predicter.py [<args>] [-h | --help]"
    )

    parser.add_argument("--input", type=str, required=True, nargs=3,
                        help="Path of input source, target corpus files")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, required=True,
                        help="Path of source and target vocabulary")

    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        hops=1,
        mapping=None,
        append_eos=False,
        device_list=[4],
        num_threads=1,
        predict_batch_size = 32
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    if model_name.startswith("experimental_"):
        model_name = model_name[13:]

    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)
    
    print (args.vocabulary)

    params.vocabulary = vocabulary.load_vocabulary(args.vocabulary)

    params.vocabulary = vocabulary.process_vocabulary(params.vocabulary, params)

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = vocabulary.get_control_mapping(params.vocabulary, control_symbols)

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix, feed_dict):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                placeholder = tf.placeholder(tf.float32,
                                             name="placeholder/" + var_name)
                with tf.device("/cpu:0"):
                    op = tf.assign(var, placeholder)
                    ops.append(op)
                feed_dict[placeholder] = value_dict[name]
                break

    return ops


def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]

        if batch < num_shards:
            feed_dict[placeholders[0][name]] = feat
            n = 1
        else:
            shard_size = (batch + num_shards - 1) // num_shards

            for i in range(num_shards):
                shard_feat = feat[i * shard_size:(i + 1) * shard_size]
                feed_dict[placeholders[i][name]] = shard_feat
                n = num_shards

    if isinstance(predictions, (list, tuple)):
        predictions = [item[:n] for item in predictions]

    return predictions, feed_dict


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]
    
    with tf.Graph().as_default():
        model_var_lists = []

        for i, checkpoint in enumerate(args.checkpoints):
            tf.logging.info("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)

        model_list = []

        for i in range(len(args.checkpoints)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_list.append(model)

        params = params_list[0]
        params.initializer_gain = 1.0

        sorted_keys, sorted_inputs = dataset.read_eval_input_file(args.input)
        
        features = dataset.get_predict_input(sorted_inputs, params)
        
        placeholders = []

        for i in range(len(params.device_list)):
            placeholders.append({
                "text": tf.placeholder(tf.int32, [None, None],
                                         "text_%d" % i),
                "text_length": tf.placeholder(tf.int32, [None],
                                                "text_length_%d" % i),
                "aspect": tf.placeholder(tf.int32, [None, None],
                                         "aspect_%d" % i),
                "aspect_length": tf.placeholder(tf.int32, [None],
                                                "aspect_length_%d" % i),
                "polarity": tf.placeholder(tf.int32, [None, None],
                                        "polarity_%d" % i)
            })

        predict_fn = inference.create_predict_graph

        predictions = parallel.data_parallelism(
            params.device_list, lambda f: predict_fn(model_list, f, params),
            placeholders)

        assign_ops = []
        feed_dict = {}

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i, feed_dict)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)
        init_op = tf.tables_initializer()
        results = []

        with tf.Session(config=session_config(params)) as sess:
            sess.run(assign_op, feed_dict=feed_dict)
            sess.run(init_op)

            while True:
                try:
                    feats = sess.run(features)
                    op, feed_dict = shard_features(feats, placeholders,
                                                   predictions)
                    results.append(sess.run(op, feed_dict=feed_dict))
                    message = "Finished batch %d" % len(results)
                    tf.logging.log(tf.logging.INFO, message)
                except tf.errors.OutOfRangeError:
                    break

        input_features = []
        scores1 = []
        scores2 = []
        output_alphas = []
        for result in results:
            for item in result[0]:
                input_features.append(item.tolist())
            for item in result[1]:
                scores1.append(item.tolist())
            for item in result[2]:
                scores2.append(item.tolist())
            for item in result[3]:
                output_alphas.append(item.tolist())
        
        scores1 = list(itertools.chain(*scores1))
        scores2 = list(itertools.chain(*scores2))
        output_alphas = list(itertools.chain(*output_alphas))

        restored_scores1 = []
        restored_scores2 = []
        restored_output_alphas = []
        restored_inputs_text = []
        restored_inputs_aspect = []
        restored_inputs_score = []

        for index in range(len(sorted_inputs[0])):
            restored_scores1.append(scores1[sorted_keys[index]][0])
            restored_scores2.append(scores2[sorted_keys[index]])
            restored_output_alphas.append(output_alphas[sorted_keys[index]])
            
            restored_inputs_text.append(sorted_inputs[0][sorted_keys[index]])
            restored_inputs_aspect.append(sorted_inputs[1][sorted_keys[index]])
            restored_inputs_score.append(sorted_inputs[2][sorted_keys[index]])

        class3_bad_TP = 0.0
        class3_bad_FP = 0.0
        class3_bad_FN = 0.0
        
        class3_mid_TP = 0.0
        class3_mid_FP = 0.0
        class3_mid_FN = 0.0
        
        class3_good_TP = 0.0
        class3_good_FP = 0.0
        class3_good_FN = 0.0

    
        with open(args.output, "w") as outfile:
            
            for score1, score2, score3, alphas, text, aspect in zip(restored_scores1, restored_scores2, restored_inputs_score, restored_output_alphas, restored_inputs_text, restored_inputs_aspect):
                score1 = str(score1)
                outfile.write("###########################\n")
                pattern = "%s|||%f,%f,%f|||%s\n"
                values = (score1, score2[0], score2[1], score2[2], score3)
                outfile.write(pattern % values)
                outfile.write(aspect + "\n")
                for (word, alpha) in zip(text.split(),alphas):
                    outfile.write(word + " " + str(alpha) + "\t")
                outfile.write("\n")
                
                if score1 == '0' and score3 == '0':
                    class3_bad_TP += 1.0
                if score1 == '1' and score3 == '1':
                    class3_mid_TP += 1.0
                if score1 == '2' and score3 == '2':
                    class3_good_TP += 1.0
            
                if score1 == '0' and score3 != '0':
                    class3_bad_FP += 1.0
                if score1 == '1' and score3 != '1':
                    class3_mid_FP += 1.0
                if score1 == '2' and score3 != '2':
                    class3_good_FP += 1.0

                if score1 != '0' and score3 == '0':
                    class3_bad_FN += 1.0
                if score1 != '1' and score3 == '1':
                    class3_mid_FN += 1.0
                if score1 != '2' and score3 == '2':
                    class3_good_FN += 1.0

            outfile.write("\n")
            outfile.write("Class 3:\n")
            outfile.write("Confusion Matrix:\n")
            outfile.write("\t" + "{name: >10s}".format(name="positive") + "\t" +"{name: >10s}".format(name="neural") + "\t" + "{name: >10s}".format(name="negative") + "\n")
            outfile.write("TP\t" + int2int(class3_bad_TP) + "\t" + int2int(class3_mid_TP) + "\t" + int2int(class3_good_TP) + "\n")
            outfile.write("FP\t" + int2int(class3_bad_FP) + "\t" + int2int(class3_mid_FP) + "\t" + int2int(class3_good_FP) + "\n")
            outfile.write("FN\t" + int2int(class3_bad_FN) + "\t" + int2int(class3_mid_FN) + "\t" + int2int(class3_good_FN) + "\n")
            outfile.write("P\t" + float2int(class3_bad_TP/(class3_bad_TP + class3_bad_FP + 0.000001)) + "\t"
                    + float2int(class3_mid_TP/(class3_mid_TP + class3_mid_FP + 0.000001)) + "\t"
                    + float2int(class3_good_TP/(class3_good_TP + class3_good_FP + 0.000001)) + "\n")
            outfile.write("R\t" + float2int(class3_bad_TP/(class3_bad_TP + class3_bad_FN + 0.000001)) + "\t"
                    + float2int(class3_mid_TP/(class3_mid_TP + class3_mid_FN + 0.000001)) + "\t"
                    + float2int(class3_good_TP/(class3_good_TP + class3_good_FN + 0.000001)) + "\n")
            outfile.write("F1\t" + float2int(class3_bad_TP * 2/(class3_bad_TP * 2 + class3_bad_FP + class3_bad_FN + 0.000001)) + "\t"
                    + float2int(class3_mid_TP * 2/(class3_mid_TP * 2 + class3_mid_FP + class3_mid_FN + 0.000001)) + "\t"
                    + float2int(class3_good_TP * 2/(class3_good_TP * 2 + class3_good_FP + class3_good_FN + 0.000001)) + "\n")
            outfile.write("F1-Micro:\t" + float2int((class3_bad_TP + class3_mid_TP + class3_good_TP) * 2/((class3_bad_TP + class3_mid_TP + class3_good_TP) * 2 + (class3_bad_FP + class3_mid_FP + class3_good_FP) + (class3_bad_FN + class3_mid_FN + class3_good_FN) + 0.000001)) + "\n")
            outfile.write("F1-Macro:\t" + float2int((class3_bad_TP * 2/(class3_bad_TP * 2 + class3_bad_FP + class3_bad_FN + 0.000001)
                                               + class3_mid_TP * 2/(class3_mid_TP * 2 + class3_mid_FP + class3_mid_FN + 0.000001)
                                               + class3_good_TP * 2/(class3_good_TP * 2 + class3_good_FP + class3_good_FN + 0.000001))/3.0) + "\n")



if __name__ == "__main__":
    main(parse_args())
