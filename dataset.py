import os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42


def create_generator_for_ffn(
        data_dir,
        mode='train'):

    file_list = glob(os.path.join(data_dir, '*.csv'))

    for full_file_path in file_list:
        # full_file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError("File %s not found" % full_file_path)
        for df in pd.read_csv(full_file_path, chunksize=10**5):

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
                if mode in ['train', 'eval']:
                    yield {
                        "q_vectors": q_vectors,
                        "a_vectors": a_vectors
                    }, 1
                else:
                    yield {
                        "q_vectors": q_vectors,
                        "a_vectors": a_vectors,
                    }


def create_dataset_for_ffn(
        data_dir,
        mode='train',
        hidden_size=768,
        shuffle_buffer=10000,
        prefetch=128,
        batch_size=32):

    def gen(): return create_generator_for_ffn(
        data_dir=data_dir,
        mode=mode)

    output_types = {
        'q_vectors': tf.float32,
        'a_vectors': tf.float32
    }

    output_shapes = {
        'q_vectors': [hidden_size],
        'a_vectors': [hidden_size],
    }

    if mode in ['train', 'eval']:
        output_types = (output_types, tf.int32)
        output_shapes = (output_shapes, [])

    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=output_types,
        output_shapes=output_shapes
    )
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
        data_dir,
        tokenizer,
        mode='train',
        max_seq_length=256,
        dynamic_padding=False):
    file_list = glob(os.path.join(data_dir, '*.csv'))
    for full_file_path in file_list:
        # full_file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError("File %s not found" % full_file_path)
        for df in pd.read_csv(full_file_path, chunksize=10**5):

            # so train test split
            if mode == 'train':
                df, _ = train_test_split(df, test_size=0.2, random_state=SEED)
            else:
                _, df = train_test_split(df, test_size=0.2, random_state=SEED)

            for _, row in df.iterrows():
                q_features = convert_text_to_feature(
                    row.question, tokenizer, max_seq_length, dynamic_padding=dynamic_padding)
                a_features = convert_text_to_feature(
                    row.answer, tokenizer, max_seq_length, dynamic_padding=dynamic_padding)
                if mode in ['train', 'eval']:
                    yield {
                        "q_input_ids": q_features[0],
                        "q_input_masks": q_features[1],
                        "q_segment_ids": q_features[2],
                        "a_input_ids": a_features[0],
                        "a_input_masks": a_features[1],
                        "a_segment_ids": a_features[2],
                    }, 1
                else:
                    yield {
                        "q_input_ids": q_features[0],
                        "q_input_masks": q_features[1],
                        "q_segment_ids": q_features[2],
                        "a_input_ids": a_features[0],
                        "a_input_masks": a_features[1],
                        "a_segment_ids": a_features[2],
                    }


def _qa_ele_to_length(yield_dict):
    return tf.shape(yield_dict['q_input_ids'])[0]+tf.shape(yield_dict['a_input_ids'])[0]


def create_dataset_for_bert(
        data_dir,
        tokenizer=None,
        mode='train',
        max_seq_length=256,
        shuffle_buffer=10000,
        prefetch=128,
        batch_size=32,
        dynamic_padding=False,
        bucket_batch_sizes=[64, 32, 16],
        bucket_boundaries=[100, 300],
        element_length_func=_qa_ele_to_length):

    def gen(): return create_generator_for_bert(
        data_dir=data_dir,
        tokenizer=tokenizer,
        mode=mode,
        max_seq_length=max_seq_length,
        dynamic_padding=dynamic_padding)

    output_types = {
        'q_input_ids': tf.int32,
        'q_input_masks': tf.int32,
        'q_segment_ids': tf.int32,
        'a_input_ids': tf.int32,
        'a_input_masks': tf.int32,
        'a_segment_ids': tf.int32
    }
    if dynamic_padding:
        output_shapes = {
            'q_input_ids': [None],
            'q_input_masks': [None],
            'q_segment_ids': [None],
            'a_input_ids': [None],
            'a_input_masks': [None],
            'a_segment_ids': [None]
        }
    else:
        output_shapes = {
            'q_input_ids': [max_seq_length],
            'q_input_masks': [max_seq_length],
            'q_segment_ids': [max_seq_length],
            'a_input_ids': [max_seq_length],
            'a_input_masks': [max_seq_length],
            'a_segment_ids': [max_seq_length]
        }
    if mode in ['train', 'eval']:
        output_types = (output_types, tf.int32)
        output_shapes = (output_shapes, [])

    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=output_types,
        output_shapes=output_shapes
    )
    if mode == 'train':
        dataset = dataset.shuffle(shuffle_buffer)
    if dynamic_padding:
        dataset = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=element_length_func,
                bucket_batch_sizes=bucket_batch_sizes,
                bucket_boundaries=bucket_boundaries,
            ))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(prefetch)

    return dataset
