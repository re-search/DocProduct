
import os
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_generator_for_ffn(
        file_list,
        mode='train'):

    # file_list = glob(os.path.join(data_dir, '*.csv'))

    for full_file_path in file_list:
        # full_file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError("File %s not found" % full_file_path)
        df = pd.read_csv(full_file_path, encoding='utf8')

        # so train test split
        if mode == 'train':
            df, _ = train_test_split(df, test_size=0.2, random_state=SEED)
        else:
            _, df = train_test_split(df, test_size=0.2, random_state=SEED)

        for _, row in df.iterrows():
            q_vectors = np.fromstring(row.question_bert.replace(
                '[[', '').replace(']]', ''), sep=' ')
            a_vectors = np.fromstring(row.answer_bert.replace(
                '[[', '').replace(']]', ''), sep=' ')
            vectors = np.stack([q_vectors, a_vectors], axis=0)
            if mode in ['train', 'eval']:
                yield vectors, 1
            else:
                yield vectors


def ffn_serialize_fn(features):
    features_tuple = {'features': _float_list_feature(
        features[0].flatten()), 'labels': _int64_feature(features[1])}
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features_tuple))
    return example_proto.SerializeToString()


def make_tfrecord(data_dir, generator_fn, serialize_fn, suffix='', **kwargs):
    """Function to make TF Records from csv files
    This function will take all csv files in data_dir, convert them
    to tf example and write to *_{suffix}_train/eval.tfrecord to data_dir.

    Arguments:
        data_dir {str} -- dir that has csv files and store tf record
        generator_fn {fn} -- A function that takes a list of filepath and yield the
        parsed recored from file.
        serialize_fn {fn} -- A function that takes output of generator fn and convert to tf example

    Keyword Arguments:
        suffix {str} -- suffix to add to tf record files (default: {''})
    """
    file_list = glob(os.path.join(data_dir, '*.csv'))
    train_tf_record_file_list = [
        f.replace('.csv', '_{0}_train.tfrecord'.format(suffix)) for f in file_list]
    test_tf_record_file_list = [
        f.replace('.csv', '_{0}_eval.tfrecord'.format(suffix)) for f in file_list]
    for full_file_path, train_tf_record_file_path, test_tf_record_file_path in zip(file_list, train_tf_record_file_list, test_tf_record_file_list):
        print('Converting file {0} to TF Record'.format(full_file_path))
        with tf.io.TFRecordWriter(train_tf_record_file_path) as writer:
            for features in generator_fn([full_file_path], mode='train', **kwargs):
                example = serialize_fn(features)
                writer.write(example)
        with tf.io.TFRecordWriter(test_tf_record_file_path) as writer:
            for features in generator_fn([full_file_path], mode='eval', **kwargs):
                example = serialize_fn(features)
                writer.write(example)


def create_dataset_for_ffn(
        data_dir,
        mode='train',
        hidden_size=768,
        shuffle_buffer=10000,
        prefetch=10000,
        batch_size=32):

    tfrecord_file_list = glob(os.path.join(
        data_dir, '*_FFN_{0}.tfrecord'.format((mode))))
    if not tfrecord_file_list:
        print('TF Record not found')
        make_tfrecord(
            data_dir, create_generator_for_ffn,
            ffn_serialize_fn, 'FFN')

    dataset = tf.data.TFRecordDataset(tfrecord_file_list)

    def _parse_ffn_example(example_proto):
        feature_description = {
            'features': tf.io.FixedLenFeature([2*768], tf.float32),
            'labels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        feature_dict = tf.io.parse_single_example(
            example_proto, feature_description)
        return tf.reshape(feature_dict['features'], (2, 768)), feature_dict['labels']
    dataset = dataset.map(_parse_ffn_example)

    if mode == 'train':
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.prefetch(prefetch)

    dataset = dataset.batch(batch_size)
    return dataset


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


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


def convert_single_example(tokenizer, example, max_seq_length=256, dynamic_padding=False):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    if not dynamic_padding:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256, dynamic_padding=False):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length, dynamic_padding=dynamic_padding
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.squeeze(np.array(input_ids)),
        np.squeeze(np.array(input_masks)),
        np.squeeze(np.array(segment_ids)),
        np.array(labels).reshape(-1, 1),
    )


def convert_text_to_feature(text, tokenizer, max_seq_length, dynamic_padding=False):
    example = InputExample(
        guid=None, text_a=text)
    features = convert_examples_to_features(
        tokenizer, [example], max_seq_length, dynamic_padding=dynamic_padding)
    return features


def create_generator_for_bert(
        file_list,
        tokenizer,
        mode='train',
        max_seq_length=256,
        dynamic_padding=False):
    # file_list = glob(os.path.join(data_dir, '*.csv'))
    for full_file_path in file_list:
        # full_file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError("File %s not found" % full_file_path)

        if os.path.basename(full_file_path) == 'healthtap_data_cleaned.csv':
            df = pd.read_csv(full_file_path, lineterminator='\n')
            df.columns = ['index', 'question', 'answer']
            df.drop(columns=['index'], inplace=True)
        else:
            df = pd.read_csv(full_file_path, lineterminator='\n')

        # so train test split
        if mode == 'train':
            df, _ = train_test_split(df, test_size=0.2, random_state=SEED)
        else:
            _, df = train_test_split(df, test_size=0.2, random_state=SEED)

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Writing to TFRecord'):
            try:
                q_features = convert_text_to_feature(
                    row.question, tokenizer, max_seq_length, dynamic_padding=dynamic_padding)
            except (ValueError, AttributeError):
                continue
            # no labels
            q_features = q_features[:3]
            try:
                a_features = convert_text_to_feature(
                    row.answer, tokenizer, max_seq_length, dynamic_padding=dynamic_padding)
            except (ValueError, AttributeError):
                continue
            a_features = a_features[:3]
            yield (q_features+a_features, 1)


def _qa_ele_to_length(features, labels):
    return tf.shape(features['q_input_ids'])[0] + tf.shape(features['a_input_ids'])[0]


def bert_serialize_fn(features):
    feature, labels = features
    # feature = [_int64_feature(f.flatten()) for f in feature]
    # labels = _int64_feature(labels)
    # features_tuple = (feature, labels)
    features_tuple = {
        'q_input_ids': _int64_list_feature(
            feature[0].flatten()),
        'q_input_masks': _int64_list_feature(
            feature[1].flatten()),
        'q_segment_ids': _int64_list_feature(
            feature[2].flatten()),
        'q_input_shape': _int64_list_feature(
            feature[0].shape),
        'a_input_ids': _int64_list_feature(
            feature[3].flatten()),
        'a_input_masks': _int64_list_feature(
            feature[4].flatten()),
        'a_segment_ids': _int64_list_feature(
            feature[5].flatten()),
        'a_input_shape': _int64_list_feature(
            feature[3].shape),
        'labels': _int64_feature(labels)}
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features_tuple))
    return example_proto.SerializeToString()


def create_dataset_for_bert(
        data_dir,
        tokenizer=None,
        mode='train',
        max_seq_length=256,
        shuffle_buffer=10000,
        prefetch=10000,
        batch_size=32,
        dynamic_padding=False,
        bucket_batch_sizes=[32, 16, 8],
        bucket_boundaries=[64, 128],
        element_length_func=_qa_ele_to_length):

    tfrecord_file_list = glob(os.path.join(
        data_dir, '*_BertFFN_{0}.tfrecord'.format((mode))))
    if not tfrecord_file_list:
        print('TF Record not found')
        make_tfrecord(
            data_dir, create_generator_for_bert,
            bert_serialize_fn, 'BertFFN', tokenizer=tokenizer, dynamic_padding=True, max_seq_length=max_seq_length)
        tfrecord_file_list = glob(os.path.join(
            data_dir, '*_BertFFN_{0}.tfrecord'.format((mode))))

    dataset = tf.data.TFRecordDataset(tfrecord_file_list)

    def _parse_bert_example(example_proto):
        feature_description = {
            'q_input_ids': tf.io.VarLenFeature(tf.int64),
            'q_input_masks': tf.io.VarLenFeature(tf.int64),
            'q_segment_ids': tf.io.VarLenFeature(tf.int64),
            'a_input_ids': tf.io.VarLenFeature(tf.int64),
            'a_input_masks': tf.io.VarLenFeature(tf.int64),
            'a_segment_ids': tf.io.VarLenFeature(tf.int64),
            'labels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }
        feature_dict = tf.io.parse_single_example(
            example_proto, feature_description)
        dense_feature_dict = {k: tf.sparse.to_dense(
            v) for k, v in feature_dict.items() if k != 'labels'}
        dense_feature_dict['labels'] = feature_dict['labels']
        return dense_feature_dict, feature_dict['labels']
    dataset = dataset.map(_parse_bert_example)

    if mode == 'train':
        dataset = dataset.shuffle(shuffle_buffer)
    if dynamic_padding:
        dataset = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=element_length_func,
                bucket_batch_sizes=bucket_batch_sizes,
                bucket_boundaries=bucket_boundaries
            ))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(prefetch)

    return dataset
