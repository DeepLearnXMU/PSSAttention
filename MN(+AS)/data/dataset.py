# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator

import numpy as np
import tensorflow as tf


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs


def get_training_input(filenames, params):

    with tf.device("/cpu:0"):
        text_dataset = tf.data.TextLineDataset(filenames[0])
        aspect_dataset = tf.data.TextLineDataset(filenames[1])
        polarity_dataset = tf.data.TextLineDataset(filenames[2])

        dataset = tf.data.Dataset.zip((text_dataset, aspect_dataset, polarity_dataset))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        dataset = dataset.map(
            lambda text, aspect, polarity: (
                tf.string_split([text]).values,
                tf.string_split([aspect]).values,
                tf.string_split([polarity]).values
            ),
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.map(
            lambda text, aspect, polarity: {
                "text": text,
                "aspect": aspect,
                "polarity": tf.string_to_number(polarity),
                "text_length": tf.shape(text),
                "aspect_length": tf.shape(aspect)
            },
            num_parallel_calls=params.num_threads
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary),
            default_value=params.mapping[params.unk]
        )

        features["text"] = table.lookup(features["text"])
        features["aspect"] = table.lookup(features["aspect"])
        
        shard_multiplier = len(params.device_list) * params.update_cycle
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=shard_multiplier,
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        features["text"] = tf.to_int32(features["text"])
        features["aspect"] = tf.to_int32(features["aspect"])
        features["polarity"] = tf.to_int32(features["polarity"])
        features["text_length"] = tf.to_int32(features["text_length"])
        features["aspect_length"] = tf.to_int32(features["aspect_length"])
        features["text_length"] = tf.squeeze(features["text_length"], 1)
        features["aspect_length"] = tf.squeeze(features["aspect_length"], 1)
        
        return features


def get_final_training_input(filenames, params):
    with tf.device("/cpu:0"):
        text_dataset = tf.data.TextLineDataset(filenames[0])
        aspect_dataset = tf.data.TextLineDataset(filenames[1])
        polarity_dataset = tf.data.TextLineDataset(filenames[2])
        attention_value_dataset = tf.data.TextLineDataset(filenames[3])
        attention_mask_dataset = tf.data.TextLineDataset(filenames[4])
        
        dataset = tf.data.Dataset.zip((text_dataset, aspect_dataset, polarity_dataset, attention_value_dataset, attention_mask_dataset))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()
        
        dataset = dataset.map(
            lambda text, aspect, polarity, attention_value, attention_mask: (
                tf.string_split([text]).values,
                tf.string_split([aspect]).values,
                tf.string_split([polarity]).values,
                tf.string_split([attention_value]).values,
                tf.string_split([attention_mask]).values
            ),
            num_parallel_calls=params.num_threads
        )
    
        dataset = dataset.map(
            lambda text, aspect, polarity, attention_value, attention_mask: {
                "text": text,
                "aspect": aspect,
                "polarity": tf.string_to_number(polarity),
                "attention_value": tf.string_to_number(attention_value),
                "attention_mask": tf.string_to_number(attention_mask),
                "text_length": tf.shape(text),
                "aspect_length": tf.shape(aspect)
            },
            num_parallel_calls=params.num_threads
        )
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        
        table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary),
            default_value=params.mapping[params.unk]
        )

        features["text"] = table.lookup(features["text"])
        features["aspect"] = table.lookup(features["aspect"])
    
        shard_multiplier = len(params.device_list) * params.update_cycle
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=shard_multiplier,
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)
            
        features["text"] = tf.to_int32(features["text"])
        features["aspect"] = tf.to_int32(features["aspect"])
        features["polarity"] = tf.to_int32(features["polarity"])
        features["attention_value"] = tf.to_float(features["attention_value"])
        features["attention_mask"] = tf.to_float(features["attention_mask"])
        features["text_length"] = tf.to_int32(features["text_length"])
        features["aspect_length"] = tf.to_int32(features["aspect_length"])
        features["text_length"] = tf.squeeze(features["text_length"], 1)
        features["aspect_length"] = tf.squeeze(features["aspect_length"], 1)
                                  
        return features

def read_eval_input_file(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1), reverse=True)
    sorted_inputs = []
    sorted_keys = {}

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, [list(x) for x in zip(*sorted_inputs)], 


def get_predict_input(inputs, params):
    with tf.device("/cpu:0"):
        datasets = []

        for data in inputs:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        dataset = dataset.map(
            lambda *x: {
                "text": x[0],
                "text_length": tf.shape(x[0])[0],
                "aspect": x[1],
                "aspect_length": tf.shape(x[1])[0],
                "polarity": tf.to_int32(tf.string_to_number(x[2]))
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.predict_batch_size,
            {
                "text": [tf.Dimension(None)],
                "text_length": [],
                "aspect": [tf.Dimension(None)],
                "aspect_length": [],
                "polarity": [tf.Dimension(None)]
            },
            {
                "text": params.pad,
                "text_length": 0,
                "aspect": params.pad,
                "aspect_length": 0,
                "polarity": -1
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary),
            default_value=params.mapping[params.unk]
        )

        features["text"] = table.lookup(features["text"])
        features["aspect"] = table.lookup(features["aspect"])

        features["text"] = tf.to_int32(features["text"])
        features["aspect"] = tf.to_int32(features["aspect"])
        features["text_length"] = tf.to_int32(features["text_length"])
        features["aspect_length"] = tf.to_int32(features["aspect_length"])

    return features
