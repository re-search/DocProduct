import argparse
import os

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


from docproduct.dataset import create_dataset_for_bert
from docproduct.models import MedicalQAModelwithBert
from docproduct.loss import qa_pair_loss, qa_pair_cross_entropy_loss
from docproduct.tokenization import FullTokenizer
from docproduct.metrics import qa_pair_batch_accuracy

DEVICE = ["/gpu:0", "/gpu:1"]


def train_bertffn(model_path='models/bertffn_crossentropy/bertffn',
                  data_path='data/mqa_csv',
                  num_epochs=20,
                  num_gpu=1,
                  batch_size=64,
                  learning_rate=2e-5,
                  validation_split=0.2,
                  loss='categorical_crossentropy',
                  pretrained_path='pubmed_pmc_470k/',
                  max_seq_len=256):
    tf.compat.v1.disable_eager_execution()
    if loss == 'categorical_crossentropy':
        loss_fn = qa_pair_cross_entropy_loss
    else:
        loss_fn = qa_pair_loss

    K.set_floatx('float32')
    tokenizer = FullTokenizer(os.path.join(pretrained_path, 'vocab.txt'))
    d = create_dataset_for_bert(
        data_path, tokenizer=tokenizer, batch_size=batch_size,
        shuffle_buffer=500000, dynamic_padding=False, max_seq_length=max_seq_len)
    eval_d = create_dataset_for_bert(
        data_path, tokenizer=tokenizer, batch_size=batch_size,
        mode='eval', dynamic_padding=False, max_seq_length=max_seq_len,
        bucket_batch_sizes=[64, 64, 64])

    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICE[:num_gpu])
    global_batch_size = batch_size*num_gpu
    learning_rate = learning_rate*1.5**num_gpu

    # with mirrored_strategy.scope():
    # d = create_dataset_for_bert(
    #     data_path, batch_size=global_batch_size, shuffle_buffer=100000)

    # d_iter = mirrored_strategy.make_dataset_iterator(d)
    input_layer = {
        'q_input_ids': keras.Input(shape=(None, ), name='q_input_ids'),
        'q_input_masks': keras.Input(shape=(None, ), name='q_input_masks'),
        'q_segment_ids': keras.Input(shape=(None, ), name='q_segment_ids'),
        'a_input_ids': keras.Input(shape=(None, ), name='a_input_ids'),
        'a_input_masks': keras.Input(shape=(None, ), name='a_input_masks'),
        'a_segment_ids': keras.Input(shape=(None, ), name='a_segment_ids'),
    }

    base_model = MedicalQAModelwithBert(config_file=os.path.join(
        pretrained_path, 'bert_config.json'),
        checkpoint_file=os.path.join(pretrained_path, 'biobert_model.ckpt'))
    outputs = base_model(input_layer)

    medical_qa_model = keras.Model(inputs=input_layer, outputs=outputs)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    medical_qa_model.compile(
        optimizer=optimizer, loss=loss)

    config = tf.estimator.RunConfig(
        train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)

    estimator = tf.keras.estimator.model_to_estimator(
        medical_qa_model, model_dir=model_path)

    def train_input_fn():
        return create_dataset_for_bert(
            data_path, tokenizer=tokenizer, batch_size=batch_size,
            shuffle_buffer=500000, dynamic_padding=False, max_seq_length=max_seq_len)
    estimator.train(train_input_fn, steps=100)

    epochs = num_epochs
    loss_metric = tf.keras.metrics.Mean()

    medical_qa_model.fit(d, epochs=epochs)
    medical_qa_model.summary()
    medical_qa_model.save_weights(model_path)
    medical_qa_model.evaluate(eval_d)


if __name__ == "__main__":

    train_bertffn()
