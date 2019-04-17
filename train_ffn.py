import argparse

import tensorflow as tf
import tensorflow.keras.backend as K

from dataset import create_dataset_for_ffn
from models import MedicalQAModel, qa_pair_loss


def train_ffn(args):
    d = create_dataset_for_ffn(
        args.data_path, batch_size=args.batch_size, shuffle_buffer=100000)
    medical_qa_model = MedicalQAModel()
    optimizer = tf.keras.optimizers.Adam()
    medical_qa_model.compile(optimizer=optimizer)

    epochs = args.num_epochs
    loss_metric = tf.keras.metrics.Mean()
    K.set_learning_phase(1)

    # Iterate over epochs.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(d):
            with tf.GradientTape() as tape:
                q_embedding, a_embedding = medical_qa_model(x_batch_train)
                loss = qa_pair_loss(q_embedding, a_embedding)

            grads = tape.gradient(loss, medical_qa_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, medical_qa_model.trainable_variables))

            loss_metric(loss)

            if step % 100 == 0:
                print('step %s: mean loss = %s' %
                      (step, loss_metric.result()))

    tf.keras.models.save_model(
        medical_qa_model,
        args.model_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/', help='path for saving trained models')
    parser.add_argument('--data_path', type=str,
                        default='/content/gdrive/', help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--validation_split', type=float, default=0.2)

    args = parser.parse_args()
    train_ffn(args)
