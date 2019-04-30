import argparse

import tensorflow as tf
import tensorflow.keras.backend as K

from dataset import create_dataset_for_ffn
from models import MedicalQAModel
from loss import qa_pair_loss, qa_pair_cross_entropy_loss

DEVICE = ["/gpu:0", "/gpu:1"]


def multi_gpu_train(args, loss=qa_pair_loss):
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICE[:args.num_gpu])
    global_batch_size = args.batch_size*args.num_gpu
    learning_rate = args.learning_rate*1.5**args.num_gpu
    with mirrored_strategy.scope():
        d = create_dataset_for_ffn(
            args.data_path, batch_size=global_batch_size, shuffle_buffer=100000)

        d_iter = mirrored_strategy.make_dataset_iterator(d)

        medical_qa_model = tf.keras.Sequential()
        medical_qa_model.add(tf.keras.layers.Input((2, 768)))
        medical_qa_model.add(MedicalQAModel())
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        medical_qa_model.compile(
            optimizer=optimizer, loss=loss)

    epochs = args.num_epochs
    loss_metric = tf.keras.metrics.Mean()

    medical_qa_model.fit(d_iter, epochs=epochs)
    medical_qa_model.save_weights(args.model_path)
    return medical_qa_model


def single_gpu_train(args, loss=qa_pair_loss):
    global_batch_size = args.batch_size*args.num_gpu
    learning_rate = args.learning_rate
    d = create_dataset_for_ffn(
        args.data_path, batch_size=global_batch_size, shuffle_buffer=50000)

    medical_qa_model = MedicalQAModel()
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    medical_qa_model.compile(
        optimizer=optimizer, loss=loss)

    epochs = args.num_epochs
    loss_metric = tf.keras.metrics.Mean()

    medical_qa_model.fit(d, epochs=epochs)
    medical_qa_model.save_weights(args.model_path)
    return medical_qa_model


def train_ffn(args):

    if args.loss == 'categorical_crossentropy':
        loss_fn = qa_pair_cross_entropy_loss
    else:
        loss_fn = qa_pair_loss

    if args.num_gpu > 1:
        medical_qa_model = multi_gpu_train(args, loss_fn)
    else:
        medical_qa_model = single_gpu_train(args, loss_fn)

    eval_d = create_dataset_for_ffn(
        args.data_path, batch_size=args.batch_size, mode='eval')

    medical_qa_model.summary()
    medical_qa_model.save_weights(args.model_path, overwrite=True)
    # K.set_learning_phase(0)
    # q_embedding, a_embedding = tf.unstack(
    #     medical_qa_model(next(iter(eval_d))[0]), axis=1)

    # q_embedding = q_embedding / tf.norm(q_embedding, axis=-1, keepdims=True)
    # a_embedding = a_embedding / tf.norm(a_embedding, axis=-1, keepdims=True)

    # batch_score = tf.reduce_sum(q_embedding*a_embedding, axis=-1)
    # baseline_score = tf.reduce_mean(
    #     tf.matmul(q_embedding, tf.transpose(a_embedding)))

    # print('Eval Batch Cos similarity')
    # print(tf.reduce_mean(batch_score))
    # print('Baseline: {0}'.format(baseline_score))

    # medical_qa_model.save_weights(args.model_path, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/ffn_crossentropy/ffn', help='path for saving trained models')
    parser.add_argument('--data_path', type=str,
                        default='data/mqa_csv', help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--loss', type=str,
                        default='categorical_crossentropy')

    args = parser.parse_args()
    train_ffn(args)
