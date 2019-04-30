import argparse
import os

import tensorflow as tf
import tensorflow.keras.backend as K

from dataset import create_dataset_for_bert
from models import MedicalQAModelwithBert
from loss import qa_pair_loss, qa_pair_cross_entropy_loss
from tokenization import FullTokenizer
from metrics import qa_pair_batch_accuracy

tf.compat.v1.disable_eager_execution()


def train_all(args):
    if args.loss == 'categorical_crossentropy':
        loss_fn = qa_pair_cross_entropy_loss
    else:
        loss_fn = qa_pair_loss
    K.set_floatx('float32')
    tokenizer = FullTokenizer(os.path.join(args.pretrained_path, 'vocab.txt'))
    d = create_dataset_for_bert(
        args.data_path, tokenizer=tokenizer, batch_size=args.batch_size,
        shuffle_buffer=100000, dynamic_padding=True, max_seq_length=args.max_seq_len)
    eval_d = create_dataset_for_bert(
        args.data_path, tokenizer=tokenizer, batch_size=args.batch_size,
        mode='eval', dynamic_padding=True, max_seq_length=args.max_seq_len)
    medical_qa_model = MedicalQAModelwithBert(
        config_file=os.path.join(
            args.pretrained_path, 'bert_config.json'),
        checkpoint_file=os.path.join(args.pretrained_path, 'biobert_model.ckpt'))
    optimizer = tf.keras.optimizers.Adam()
    medical_qa_model.compile(
        optimizer=optimizer, loss=loss_fn, metrics=[qa_pair_batch_accuracy])

    epochs = args.num_epochs
    loss_metric = tf.keras.metrics.Mean()

    medical_qa_model.fit(d, epochs=epochs)
    medical_qa_model.summary()
    medical_qa_model.save_weights(args.model_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/bertffn_crossentropy/bertffn', help='path for saving trained models')
    parser.add_argument('--data_path', type=str,
                        default='data/mqa_csv', help='path for tfrecords data')
    parser.add_argument('--pretrained_path', type=str,
                        default='pubmed_pmc_470k/', help='pretrained model path')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--loss', type=str,
                        default='categorical_crossentropy')

    args = parser.parse_args()
    train_all(args)
