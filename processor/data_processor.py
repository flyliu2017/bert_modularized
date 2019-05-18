# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import re

import tensorflow as tf

from tokenization import tokenization
from model.model_fn import *
from utils.metric_utils import *

flags=tf.flags
FLAGS=flags.FLAGS

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters."""

    def __init__(self, data_dir, tokenizer, max_seq_length):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_list = self.get_labels()
        self.label_map = {}
        for i, label in enumerate(self.label_list):
            self.label_map[label] = i

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @property
    def name_to_features(self):
        """ Name to features dictionary used to parse tfrecord."""
        raise NotImplementedError

    @property
    def padding_input_features(self):
        """Fake input features,only used for TPUEstimator."""
        raise NotImplementedError

    @property
    def create_model(self):
        """Create fine-tuning model."""
        raise NotImplementedError

    @property
    def eval_metric_fn(self):
        raise NotImplementedError

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def convert_single_example(self, ex_index, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        tokens, input_ids, input_mask, segment_ids = self.create_input_features(example)
        label_id = self.create_label_features(example, tokens)

        # start,end=label.split()
        # start=int(start)
        # end=int(end)

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % example.guid)
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: {}".format(label_id))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            is_real_example=True)
        return feature

    def create_label_features(self, example, tokens):
        raise NotImplementedError

    def create_input_features(self, example):
        if isinstance(example, PaddingInputExample):
            return self.padding_input_features

        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        if len(input_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(input_ids)
            input_ids.extend([0] * padding_length)
            input_mask.extend([0] * padding_length)
            segment_ids.extend([0] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return tokens, input_ids, input_mask, segment_ids

    def file_based_convert_examples_to_features(
            self, examples, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""

        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            if feature:
                features = collections.OrderedDict()
                features["input_ids"] = create_int_feature(feature.input_ids)
                features["input_mask"] = create_int_feature(feature.input_mask)
                features["segment_ids"] = create_int_feature(feature.segment_ids)
                features["label_ids"] = create_int_feature(feature.label_id)
                features["is_real_example"] = create_int_feature(
                    [int(feature.is_real_example)])

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
        writer.close()

    def file_based_input_fn_builder(self, input_file, is_training,
                                    drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        def _decode_record(record):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, self.name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

            return d

        return input_fn

    def convert_examples_to_features(self, examples):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example)

            features.append(feature)
        return features

    def post_process(self, output_dir, label_ids, probabilities, input_mask, input_ids,threshold):
        output_predict_file = os.path.join(output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for probability in probabilities:
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probability) + "\n"
                writer.write(output_line)

class SequenceTaggingProcessor(DataProcessor):
    """Base class for data converters for sequence tagging data sets."""

    @property
    def name_to_features(self):
        return {
            "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

    @property
    def padding_input_features(self):
        return InputFeatures(
            input_ids=[0] * self.max_seq_length,
            input_mask=[0] * self.max_seq_length,
            segment_ids=[0] * self.max_seq_length,
            label_id=[0] * self.max_seq_length,
            is_real_example=False)

    @property
    def create_model(self):
        return create_sequence_tagging_model

    @property
    def eval_metric_fn(self):
        return sequence_tagging_metric_fn

    def post_process(self, output_dir, label_ids, probabilities, input_mask, input_ids,threshold):
        predictions = np.argmax(probabilities, axis=-1)
        input_mask = np.array(input_mask)
        predictions = np.where(input_mask == 1, predictions, np.full(predictions.shape, -1))
        with tf.gfile.GFile(os.path.join(output_dir,'predict_result.tsv'), "w") as writer:
            for input_id,prediction in zip(input_ids, predictions):
                for tag_id in set(list(prediction))-{-1}:
                    tag=self.label_list[tag_id]
                    output_tokens = [self.tokenizer.inv_vocab[id] if pred_id == tag_id else ' '
                                     for id, pred_id in zip(input_id, prediction)]
                    phrase=''.join(output_tokens).strip()
                    phrase=re.sub(r' +',' ',phrase)
                    writer.write('{}: {}\t'.format(tag,phrase))
                writer.write('\n')

        report_and_save_metrics(output_dir, label_ids, predictions,labels=list(range(len(self.label_list))))

class SequenceBinaryTaggingProcessor(SequenceTaggingProcessor):
    """Base class for data converters for sequence tagging data sets."""

    @property
    def create_model(self):
        return create_sequence_binary_tagging_model

    @property
    def eval_metric_fn(self):
        return sequence_binary_tagging_metric_fn

    def post_process(self, output_dir, label_ids, probabilities, input_mask, input_ids,threshold):
        probabilities=np.array(probabilities)
        input_mask=np.array(input_mask)
        predictions = np.where(np.logical_and(probabilities >= threshold, input_mask == 1),
                               np.ones(probabilities.shape),
                               tf.zeros(probabilities.shape))
        with tf.gfile.GFile(os.path.join(output_dir, 'predict_result.tsv'), "w") as writer:
            for prediction in predictions:
                output_tokens = [self.tokenizer.inv_vocab[input_id] if pred_id == 1 else ' '
                                 for input_id, pred_id in zip(input_ids, prediction)]
                phrase = ''.join(output_tokens).strip()
                phrase = re.sub(r' +', ' ', phrase)
                writer.write('{}\n'.format(phrase))

        report_and_save_metrics(output_dir, label_ids, predictions)

class SingleLabelClassificationProcessor(DataProcessor):
    """Base processor for the Single Label Classification data set."""

    @property
    def name_to_features(self):
        return {
            "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

    @property
    def padding_input_features(self):
        return InputFeatures(
            input_ids=[0] * self.max_seq_length,
            input_mask=[0] * self.max_seq_length,
            segment_ids=[0] * self.max_seq_length,
            label_id=[0],
            is_real_example=False)

    @property
    def create_model(self):
        return create_classification_model

    @property
    def eval_metric_fn(self):
        return

    def create_label_features(self, example, tokens):
        label = example.label
        return self.label_map[label]

    def post_process(self, output_dir, label_ids, probabilities, input_mask, input_ids,threshold):
        predictions = np.argmax(probabilities, axis=-1)

        with tf.gfile.GFile(os.path.join(output_dir, 'predict_result.tsv'), "w") as writer:
            for prediction in predictions:
                label=self.label_list[prediction]
                writer.write('{}\n'.format(label))

        report_and_save_metrics(output_dir, label_ids, predictions)

class MultiLabelClassificationProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    @property
    def name_to_features(self):
        return {
            "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([len(self.label_list)], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64)
        }

    @property
    def padding_input_features(self):
        return InputFeatures(
            input_ids=[0] * self.max_seq_length,
            input_mask=[0] * self.max_seq_length,
            segment_ids=[0] * self.max_seq_length,
            label_id=[0] * len(self.label_list),
            is_real_example=False)

    @property
    def create_model(self):
        def multi_label_model(*args, **kwargs):
            return create_classification_model(*args, multi_label=True, **kwargs)

        return multi_label_model

    def create_label_features(self, example, tokens):
        labels = example.label
        label_ids = [0] * len(self.label_list)
        for label in labels:
            label_ids[self.label_map[label]] = 1
        return label_ids

    def post_process(self, output_dir, label_ids, probabilities, input_mask, input_ids,threshold):
        probabilities=np.array(probabilities)
        input_mask=np.array(input_mask)
        predictions = np.where(np.logical_and(probabilities >= threshold, input_mask == 1),
                               np.ones(probabilities.shape),
                               tf.zeros(probabilities.shape))
        with tf.gfile.GFile(os.path.join(output_dir, 'predict_result.tsv'), "w") as writer:
            for prediction in predictions:
                output_tokens = [self.label_list[i] for i,pred in enumerate(prediction) if pred==1]
                labels = '\t'.join(output_tokens)
                writer.write('{}\n'.format(labels))

        report_and_save_metrics(output_dir, label_ids, predictions)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn

def report_and_save_metrics(output_dir, label_ids, predictions,labels=None):
    report = report_metrics(label_ids, predictions, labels)
    tf.logging.info(report)
    with tf.gfile.GFile(os.path.join(output_dir, 'test_result.tsv'), "w") as writer:
        writer.write(report)