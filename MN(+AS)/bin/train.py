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

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training Recurrent Neural Networks Approach for Estimating the Quality of Machine Translation",
        usage="qe_main.py [<args>] [-h | --help]"
    )

    parser.add_argument("--input", type=str, nargs=3,
                        help="Path of text, aspect and polarity files")
    parser.add_argument("--output", type=str, default="qe",
                        help="Path to saved models")
    parser.add_argument("--vocabulary", type=str,
                        help="Path of vocabulary")
    parser.add_argument("--validation", type=str, nargs=3,
                        help="Path of validation text, aspect and polarity files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--pretrained_embedding", type=str, required=True,
                    help="Name of the pretrained_embedding")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    
    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=["", "", ""],
        output="",
        model="BL_MN",
        pretrained_embedding="",
        vocab="",
        constant_batch_size=False,
        batch_size=128,
        initializer_gain=0.05,
        clip_grad_norm=50.0,
        learning_rate=0.01,
        optimizer="SGD",
        initializer="uniform",
        hops=1,
        
        learning_rate_boundaries=[1000],
        learning_rate_values=[0.001, 0.0005],
        learning_rate_decay="none",
                                
        num_threads=6,
        max_length=256,
        length_multiplier=1,
        mantissa_bits=2,
        warmup_steps=0,
        train_steps=20000,
        device_list=[4],
        update_cycle=1,
        scale_l1=0.0,
        scale_l2=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        keep_checkpoint_max=10,
        keep_top_checkpoint_max=5,
        buffer_size=10000,
        eval_steps=20,
        eval_secs=0,
        validation=["", "", ""],
        save_checkpoint_secs=0,
        save_checkpoint_steps=20,
        predict_batch_size=32,
        only_save_trainable=False
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
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    params.model = args.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.pretrained_embedding = args.pretrained_embedding or params.pretrained_embedding
    params.validation = args.validation or params.validation
    params.parse(args.parameters)

    params.vocabulary = vocabulary.load_vocabulary(params.vocab)
    
    params.vocabulary = vocabulary.process_vocabulary(
        params.vocabulary, params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]
    params.mapping = vocabulary.get_control_mapping(params.vocabulary, control_symbols)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in params.values().iterkeys():
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def restore_variables(checkpoint):
    if not checkpoint:
        return tf.no_op("restore_op")

    tf.logging.info("Loading %s" % checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))

    return tf.group(*ops, name="restore_op")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    config.gpu_options.per_process_gpu_memory_fraction=0.4
    
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)

    params = default_parameters()

    params = merge_parameters(params, model_cls.get_parameters())

    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    export_params(params.output, "params.json", params)
    export_params(
        params.output,
        "%s.json" % args.model,
        collect_params(params, model_cls.get_parameters())
    )

    with tf.Graph().as_default():
        features = dataset.get_training_input(params.input, params)

        update_cycle = params.update_cycle
        features, init_op = cache.cache_features(features, update_cycle)

        initializer = get_initializer(params)
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params.scale_l1, scale_l2=params.scale_l2)
        model = model_cls(params)
        global_step = tf.train.get_or_create_global_step()

        sharded_losses = parallel.parallel_model(
            model.get_training_func(initializer, regularizer),
            features,
            params.device_list
        )

        loss = tf.add_n(sharded_losses) / len(sharded_losses)
        loss = loss + tf.losses.get_regularization_loss()

        all_weights = {v.name: v for v in tf.trainable_variables()}
        total_size = 0

        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                            str(v.shape).ljust(20))
            v_size = np.prod(np.array(v.shape.as_list())).tolist()
            total_size += v_size
        tf.logging.info("Total trainable variables size: %d", total_size)

        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
        tf.summary.scalar("learning_rate", learning_rate)
        
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        elif params.optimizer == "SGD":
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        
        loss, ops = optimize.create_train_op(loss, opt, global_step, params)
        restore_op = restore_variables(args.checkpoint)

        if params.validation:
            eval_sorted_keys , eval_inputs = dataset.read_eval_input_file(params.validation)
            eval_input_fn = dataset.get_predict_input
        else:
            eval_input_fn = None

        save_vars = tf.trainable_variables() + [global_step]
        saver = tf.train.Saver(
            var_list=save_vars if params.only_save_trainable else None,
            max_to_keep=params.keep_checkpoint_max,
            sharded=False
        )
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)

        multiplier = tf.convert_to_tensor([update_cycle, 1])

        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "loss": loss,
                    "text": tf.shape(features["text"]) * multiplier,
                    "aspect": tf.shape(features["aspect"]) * multiplier,
                    "polarity": tf.shape(features["polarity"]) * multiplier
                },
                every_n_iter=1
            ),
            tf.train.CheckpointSaverHook(
                checkpoint_dir=params.output,
                save_secs=params.save_checkpoint_secs or None,
                save_steps=params.save_checkpoint_steps or None,
                saver=saver
            )
        ]

        config = session_config(params)

        if eval_input_fn is not None:
            train_hooks.append(
                hooks.EvaluationHook(
                    lambda f: inference.create_predict_graph(
                        [model], f, params
                    ),
                    lambda: eval_input_fn(eval_inputs, params),
                    params.output,
                    config,
                    params.keep_top_checkpoint_max,
                    eval_secs=params.eval_secs,
                    eval_steps=params.eval_steps
                )
            )

        def restore_fn(step_context):
            step_context.session.run(restore_op)

        def step_fn(step_context):
            step_context.session.run([init_op, ops["zero_op"]])
            for i in range(update_cycle - 1):
                step_context.session.run(ops["collect_op"])

            return step_context.run_with_hooks(ops["train_op"])

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=params.output, hooks=train_hooks,
                save_checkpoint_secs=None, config=config) as sess:
            sess.run_step_fn(restore_fn)

            while not sess.should_stop():
                sess.run_step_fn(step_fn)



if __name__ == "__main__":
    main(parse_args())
