import argparse
import os

import tensorflow as tf
import tensorflow.keras.backend as K

from docproduct.dataset import create_dataset_for_bert
from docproduct.models import MedicalQAModelwithBert
from docproduct.loss import qa_pair_loss, qa_pair_cross_entropy_loss
from docproduct.tokenization import FullTokenizer
from docproduct.metrics import qa_pair_batch_accuracy


def train_bertffn(model_path='models/bertffn_crossentropy/bertffn',
                  data_path='data/mqa_csv',
                  num_epochs=20,
                  num_gpu=1,
                  batch_size=64,
                  learning_rate=2e-5,
                  validation_split=0.2,
                  loss='categorical_crossentropy',
                  pretrained_path='models/pubmed_pmc_470k/',
                  max_seq_len=256):
    """A function to train BertFFNN similarity embedding model.

    Input file format:
        question,answer
        my eyes hurts, go see a doctor

    For more information about training details:
    https://github.com/Santosh-Gupta/DocProduct/blob/master/README.md

    Keyword Arguments:
        model_path {str} -- Path to save embedding model weights, ends with prefix of model files (default: {'models/bertffn_crossentropy/bertffn'})
        data_path {str} -- CSV data path (default: {'data/mqa_csv'})
        num_epochs {int} -- Number of Epochs to train (default: {20})
        num_gpu {int} -- Number of GPU to use(Currently only support single GPU) (default: {1})
        batch_size {int} -- Batch size (default: {64})
        learning_rate {float} -- learning rate (default: {2e-5})
        validation_split {float} -- validation split (default: {0.2})
        loss {str} -- Loss type, either MSE or crossentropy (default: {'categorical_crossentropy'})
        pretrained_path {str} -- Pretrained bioBert model path (default: {'models/pubmed_pmc_470k/'})
        max_seq_len {int} -- Max sequence length of model(No effects if dynamic padding is enabled) (default: {256})
    """
    tf.compat.v1.disable_eager_execution()
    if loss == 'categorical_crossentropy':
        loss_fn = qa_pair_cross_entropy_loss
    else:
        loss_fn = qa_pair_loss
    K.set_floatx('float32')
    tokenizer = FullTokenizer(os.path.join(pretrained_path, 'vocab.txt'))
    d = create_dataset_for_bert(
        data_path, tokenizer=tokenizer, batch_size=batch_size,
        shuffle_buffer=500000, dynamic_padding=True, max_seq_length=max_seq_len)
    eval_d = create_dataset_for_bert(
        data_path, tokenizer=tokenizer, batch_size=batch_size,
        mode='eval', dynamic_padding=True, max_seq_length=max_seq_len,
        bucket_batch_sizes=[64, 64, 64])

    medical_qa_model = MedicalQAModelwithBert(
        config_file=os.path.join(
            pretrained_path, 'bert_config.json'),
        checkpoint_file=os.path.join(pretrained_path, 'biobert_model.ckpt'))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    medical_qa_model.compile(
        optimizer=optimizer, loss=loss_fn, metrics=[qa_pair_batch_accuracy])

    epochs = num_epochs

    callback = tf.keras.callbacks.ModelCheckpoint(
        model_path, verbose=1, save_weights_only=True, save_best_only=False, period=1)

    medical_qa_model.fit(d, epochs=epochs, callbacks=[callback])
    medical_qa_model.summary()
    medical_qa_model.save_weights(model_path)
    medical_qa_model.evaluate(eval_d)


if __name__ == "__main__":

    train_bertffn()
