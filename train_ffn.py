import argparse

import tensorflow as tf
import tensorflow.keras.backend as K

from dataset import create_dataset_for_ffn
from models import MedicalQAModel
from loss import qa_pair_loss


def train_ffn(args):
    d = create_dataset_for_ffn(
        args.data_path, batch_size=args.batch_size, shuffle_buffer=100000)
    eval_d = create_dataset_for_ffn(
        args.data_path, batch_size=args.batch_size, mode='eval')
    medical_qa_model = MedicalQAModel()
    optimizer = tf.keras.optimizers.Adam()
    medical_qa_model.compile(
        optimizer=optimizer, loss=qa_pair_loss)

    epochs = args.num_epochs
    loss_metric = tf.keras.metrics.Mean()

    medical_qa_model.fit(d, epochs=epochs, validation_data=eval_d)
    medical_qa_model.summary()
    K.set_learning_phase(0)
    q_embedding, a_embedding = tf.unstack(
        medical_qa_model(next(iter(eval_d))[0]), axis=1)

    q_embedding = q_embedding / tf.norm(q_embedding, axis=-1, keepdims=True)
    a_embedding = a_embedding / tf.norm(a_embedding, axis=-1, keepdims=True)

    batch_score = tf.reduce_sum(q_embedding*a_embedding, axis=-1)
    baseline_score = tf.reduce_mean(
        tf.matmul(q_embedding, tf.transpose(a_embedding)))

    print('Eval Batch Cos similarity')
    print(tf.reduce_mean(batch_score))
    print('Baseline: {0}'.format(baseline_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    parser.add_argument('--data_path', type=str,
                        default='/content/gdrive/', help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--validation_split', type=float, default=0.2)

    args = parser.parse_args()
    train_ffn(args)
