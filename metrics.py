import tensorflow as tf


def qa_pair_batch_accuracy(y_true, y_pred):
    y_true = tf.eye(tf.shape(y_pred)[0])
    q_embedding, a_embedding = tf.unstack(y_pred, axis=1)
    similarity_matrix = tf.matmul(
        q_embedding, a_embedding, transpose_b=True)
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(similarity_matrix, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    return acc
