import argparse

import tensorflow as tf
import tensorflow.keras.backend as K

from docproduct.dataset import create_dataset_for_ffn
from docproduct.models import MedicalQAModel
from docproduct.loss import qa_pair_loss, qa_pair_cross_entropy_loss
from docproduct.metrics import qa_pair_batch_accuracy

DEVICE = ["/gpu:0", "/gpu:1"]


def multi_gpu_train(batch_size, num_gpu, data_path, num_epochs, model_path, loss=qa_pair_loss):
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICE[:num_gpu])
    global_batch_size = batch_size*num_gpu
    learning_rate = learning_rate*1.5**num_gpu
    with mirrored_strategy.scope():
        d = create_dataset_for_ffn(
            data_path, batch_size=global_batch_size, shuffle_buffer=100000)

        d_iter = mirrored_strategy.make_dataset_iterator(d)

        medical_qa_model = tf.keras.Sequential()
        medical_qa_model.add(tf.keras.layers.Input((2, 768)))
        medical_qa_model.add(MedicalQAModel())
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        medical_qa_model.compile(
            optimizer=optimizer, loss=loss)

    epochs = num_epochs
    loss_metric = tf.keras.metrics.Mean()

    medical_qa_model.fit(d_iter, epochs=epochs, metrics=[
                         qa_pair_batch_accuracy])
    medical_qa_model.save_weights(model_path)
    return medical_qa_model


def single_gpu_train(batch_size, num_gpu, data_path, num_epochs, model_path, loss=qa_pair_loss):
    global_batch_size = batch_size*num_gpu
    learning_rate = learning_rate
    d = create_dataset_for_ffn(
        data_path, batch_size=global_batch_size, shuffle_buffer=500000)
    eval_d = create_dataset_for_ffn(
        data_path, batch_size=batch_size, mode='eval')

    medical_qa_model = MedicalQAModel()
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    medical_qa_model.compile(
        optimizer=optimizer, loss=loss, metrics=[
            qa_pair_batch_accuracy])

    epochs = num_epochs

    medical_qa_model.fit(d, epochs=epochs, validation_data=eval_d)
    medical_qa_model.save_weights(model_path)
    return medical_qa_model


def train_ffn(model_path='models/ffn_crossentropy/ffn',
              data_path='data/mqa_csv',
              num_epochs=300,
              num_gpu=1,
              batch_size=64,
              learning_rate=0.0001,
              validation_split=0.2,
              loss='categorical_crossentropy'):

    if loss == 'categorical_crossentropy':
        loss_fn = qa_pair_cross_entropy_loss
    else:
        loss_fn = qa_pair_loss
    eval_d = create_dataset_for_ffn(
        data_path, batch_size=batch_size, mode='eval')

    if num_gpu > 1:
        medical_qa_model = multi_gpu_train(
            batch_size, num_gpu, data_path, num_epochs, model_path, loss_fn)
    else:
        medical_qa_model = single_gpu_train(
            batch_size, num_gpu, data_path, num_epochs, model_path, loss_fn)

    medical_qa_model.summary()
    medical_qa_model.save_weights(model_path, overwrite=True)
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

    # medical_qa_model.save_weights(model_path, overwrite=True)


if __name__ == "__main__":

    train_ffn()
