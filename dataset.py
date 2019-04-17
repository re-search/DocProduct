import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def create_generator_for_ffn(
        data_dir,
        file_list=[
            "ehealthforumQAs.csv",
            "icliniqQAs.csv",
            "questionDoctorQAs.csv",
            "webmdQAs.csv",
            "healthtapQAs.csv"],
        mode='train'):

    for file_name in file_list:
        full_file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(full_file_path):
            raise FileNotFoundError("File %s not found" % full_file_path)
        df = pd.read_csv(full_file_path)

        # so train test split
        if mode == 'train':
            df, _ = train_test_split(df, test_size=0.2)
        else:
            _, df = train_test_split(df, test_size=0.2)

        for _, row in df.iterrows():
            q_vectors = np.fromstring(row.question_bert.replace(
                '[[', '').replace(']]', ''), sep=' ')
            a_vectors = np.fromstring(row.answer_bert.replace(
                '[[', '').replace(']]', ''), sep=' ')
            if mode == 'train':
                yield {
                    "q_vectors": q_vectors,
                    "a_vectors": a_vectors,
                    "labels": 1
                }
            else:
                yield {
                    "q_vectors": q_vectors,
                    "a_vectors": a_vectors,
                }


def create_dataset_for_ffn(
        data_dir,
        file_list=[
            "ehealthforumQAs.csv",
            "icliniqQAs.csv",
            "questionDoctorQAs.csv",
            "webmdQAs.csv",
            "healthtapQAs.csv"],
        mode='train',
        hidden_size=768,
        shuffle_buffer=10000,
        prefetch=128,
        batch_size=32):

    def gen(): return create_generator_for_ffn(
        data_dir=data_dir,
        file_list=file_list,
        mode=mode)

    output_types = {
        'q_vectors': tf.float32,
        'a_vectors': tf.float32
    }

    output_shapes = {
        'q_vectors': [hidden_size],
        'a_vectors': [hidden_size],
    }

    if mode == 'train':
        output_types.update({'labels': tf.int32})
        output_shapes.update({'labels': []})

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
