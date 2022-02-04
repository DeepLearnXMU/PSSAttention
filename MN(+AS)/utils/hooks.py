# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import operator
import os

import tensorflow as tf


def _get_saver():
    collection_key = tf.GraphKeys.SAVERS
    savers = tf.get_collection(collection_key)

    if not savers:
        raise RuntimeError("No items in collection {}. "
                           "Please add a saver to the collection ")
    elif len(savers) > 1:
        raise RuntimeError("More than one item in collection")

    return savers[0]


def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _read_checkpoint_def(filename):
    records = []

    with tf.gfile.GFile(filename) as fd:
        fd.readline()

        for line in fd:
            records.append(line.strip().split(":")[-1].strip()[1:-1].split("/")[-1])

    return records


def _save_checkpoint_def(filename, checkpoint_names):
    keys = []

    for checkpoint_name in checkpoint_names:
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, checkpoint_name))

    sorted_names = sorted(keys, key=operator.itemgetter(0),
                          reverse=True)

    with tf.gfile.GFile(filename, "w") as fd:
        fd.write("model_checkpoint_path: \"%s\"\n" % checkpoint_names[0])

        for checkpoint_name in sorted_names:
            checkpoint_name = checkpoint_name[1]
            fd.write("all_model_checkpoint_paths: \"%s\"\n" % checkpoint_name)


def _read_score_record(filename):
    records = []

    if not tf.gfile.Exists(filename):
        return records

    with tf.gfile.GFile(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1])
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with tf.gfile.GFile(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records


def _evaluate(eval_fn, input_fn, path, config):
    graph = tf.Graph()
    with graph.as_default():
        features = input_fn()
        placeholders = {
            "text": tf.placeholder(tf.int32, [None, None], "text"),
            "text_length": tf.placeholder(tf.int32, [None], "text_length"),
            "aspect": tf.placeholder(tf.int32, [None, None], "aspect"),
            "aspect_length": tf.placeholder(tf.int32, [None], "aspect_length"),
            "polarity": tf.placeholder(tf.int32, [None, None], "polarity")
        }
        predictions = eval_fn(placeholders)
        predictions = predictions[1][:, :1]

        all_refs = []
        all_outputs = []

        sess_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=path,
            config=config
        )

        with tf.train.MonitoredSession(session_creator=sess_creator) as sess:
            while not sess.should_stop():
                feats = sess.run(features)
                outputs = sess.run(predictions, feed_dict={
                    placeholders["text"]: feats["text"],
                    placeholders["text_length"]: feats["text_length"],
                    placeholders["aspect"]: feats["aspect"],
                    placeholders["aspect_length"]: feats["aspect_length"],
                    placeholders["polarity"]: feats["polarity"]
                })
                outputs = outputs.tolist()
                references = feats["polarity"].tolist()

                all_outputs.extend(outputs)
                all_refs.extend(references)
        
        positive_TP = 0.0
        positive_FP = 0.0
        positive_FN = 0.0

        neutral_TP = 0.0
        neutral_FP = 0.0
        neutral_FN = 0.0

        negative_TP = 0.0
        negative_FP = 0.0
        negative_FN = 0.0

        for pred, ref  in zip(all_outputs, all_refs):
            ref = str(ref[0])
            pred = str(pred[0])
            if pred == '0' and ref == '0':
                positive_TP += 1.0
            if pred == '1' and ref == '1':
                neutral_TP += 1.0
            if pred == '2' and ref == '2':
                negative_TP += 1.0

            if pred == '0' and ref != '0':
                positive_FP += 1.0
            if pred == '1' and ref != '1':
                neutral_FP += 1.0
            if pred == '2' and ref != '2':
                negative_FP += 1.0

            if pred != '0' and ref == '0':
                positive_FN += 1.0
            if pred != '1' and ref == '1':
                neutral_FN += 1.0
            if pred != '2' and ref == '2':
                negative_FN += 1.0

        F1_positive = positive_TP * 2/(positive_TP * 2 + positive_FP + positive_FN + 0.000001)
        F1_neutral = neutral_TP * 2/(neutral_TP * 2 + neutral_FP + neutral_FN + 0.000001)
        F1_negative = negative_TP * 2/(negative_TP * 2 + negative_FP + negative_FN + 0.000001)

        Macro = (F1_positive + F1_neutral + F1_negative)/3.0

        return Macro


class EvaluationHook(tf.train.SessionRunHook):

    def __init__(self, eval_fn, eval_input_fn, base_dir,
                 session_config, max_to_keep=5, eval_secs=None,
                 eval_steps=None, metric="Macro"):
        tf.logging.info("Create EvaluationHook.")

        if metric != "Macro":
            raise ValueError("Currently, EvaluationHook only support Macro")

        self._base_dir = base_dir.rstrip("/")
        self._session_config = session_config
        self._save_path = os.path.join(base_dir, "eval")
        self._record_name = os.path.join(self._save_path, "record")
        self._log_name = os.path.join(self._save_path, "log")
        self._eval_fn = eval_fn
        self._eval_input_fn = eval_input_fn
        self._max_to_keep = max_to_keep
        self._metric = metric
        self._global_step = None
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=eval_secs or None, every_steps=eval_steps or None
        )

    def begin(self):
        if self._timer.last_triggered_step() is None:
            self._timer.update_last_triggered_step(0)

        global_step = tf.train.get_global_step()

        if not tf.gfile.Exists(self._save_path):
            tf.logging.info("Making dir: %s" % self._save_path)
            tf.gfile.MakeDirs(self._save_path)

        params_pattern = os.path.join(self._base_dir, "*.json")
        params_files = tf.gfile.Glob(params_pattern)

        for name in params_files:
            tf.logging.info("%s" % name)
            tf.logging.info("%s" % self._base_dir + "/")
            tf.logging.info("%s" % self._save_path + "/")
            new_name = name.replace(self._base_dir + "/", self._save_path + "/")
            tf.logging.info("%s" % new_name)
            tf.gfile.Copy(name, new_name, overwrite=True)

        if global_step is None:
            raise RuntimeError("Global step should be created first")

        self._global_step = global_step

    def before_run(self, run_context):
        args = tf.train.SessionRunArgs(self._global_step)
        return args

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results

        if self._timer.should_trigger_for_step(stale_global_step + 1):
            global_step = run_context.session.run(self._global_step)

            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                save_path = os.path.join(self._base_dir, "model.ckpt")
                saver = _get_saver()
                tf.logging.info("Saving checkpoints for %d into %s." %
                                (global_step, save_path))
                saver.save(run_context.session,
                           save_path,
                           global_step=global_step)
                tf.logging.info("Validating model at step %d" % global_step)
                score = _evaluate(self._eval_fn, self._eval_input_fn,
                                  self._base_dir,
                                  self._session_config)
                tf.logging.info("%s at step %d: %f" %
                                (self._metric, global_step, score))

                _save_log(self._log_name, (self._metric, global_step, score))

                checkpoint_filename = os.path.join(self._base_dir,
                                                   "checkpoint")
                all_checkpoints = _read_checkpoint_def(checkpoint_filename)
                records = _read_score_record(self._record_name)
                latest_checkpoint = all_checkpoints[-1]
                record = [latest_checkpoint, score]
                added, removed, records = _add_to_record(records, record,
                                                         self._max_to_keep)

                if added is not None:
                    tf.logging.info("%s" % (added))
                    old_path = os.path.join(self._base_dir, added)
                    new_path = os.path.join(self._save_path, added)
                    old_files = tf.gfile.Glob(old_path + "*")
                    tf.logging.info("Copying %s to %s" % (old_path, new_path))

                    for o_file in old_files:
                        n_file = o_file.replace(old_path, new_path)
                        tf.gfile.Copy(o_file, n_file, overwrite=True)

                if removed is not None:
                    filename = os.path.join(self._save_path, removed)
                    tf.logging.info("Removing %s" % filename)
                    files = tf.gfile.Glob(filename + "*")

                    for name in files:
                        tf.gfile.Remove(name)

                _save_score_record(self._record_name, records)
                checkpoint_filename = checkpoint_filename.replace(
                    self._base_dir, self._save_path
                )
                _save_checkpoint_def(checkpoint_filename,
                                     [item[0] for item in records])

                best_score = records[0][1]
                tf.logging.info("Best score at step %d: %f" %
                                (global_step, best_score))

    def end(self, session):
        last_step = session.run(self._global_step)

        if last_step != self._timer.last_triggered_step():
            global_step = last_step
            tf.logging.info("Validating model at step %d" % global_step)
            score = _evaluate(self._eval_fn, self._eval_input_fn,
                              self._base_dir,
                              self._session_config)
            tf.logging.info("%s at step %d: %f" %
                            (self._metric, global_step, score))

            checkpoint_filename = os.path.join(self._base_dir,
                                               "checkpoint")
            all_checkpoints = _read_checkpoint_def(checkpoint_filename)
            records = _read_score_record(self._record_name)
            latest_checkpoint = all_checkpoints[-1]
            record = [latest_checkpoint, score]
            added, removed, records = _add_to_record(records, record,
                                                     self._max_to_keep)

            if added is not None:
            	tf.logging.info("%s" % (added))
                old_path = os.path.join(self._base_dir, added)
                new_path = os.path.join(self._save_path, added)
                old_files = tf.gfile.Glob(old_path + "*")
                tf.logging.info("Copying %s to %s" % (old_path, new_path))

                for o_file in old_files:
                    n_file = o_file.replace(old_path, new_path)
                    tf.gfile.Copy(o_file, n_file, overwrite=True)

            if removed is not None:
                filename = os.path.join(self._save_path, removed)
                tf.logging.info("Removing %s" % filename)
                files = tf.gfile.Glob(filename + "*")

                for name in files:
                    tf.gfile.Remove(name)

            _save_score_record(self._record_name, records)
            checkpoint_filename = checkpoint_filename.replace(
                self._base_dir, self._save_path
            )
            _save_checkpoint_def(checkpoint_filename,
                                 [item[0] for item in records])

            best_score = records[0][1]
            tf.logging.info("Best score: %f" % best_score)
