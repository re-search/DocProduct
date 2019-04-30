import tensorflow as tf


def qa_pair_cross_entropy_loss(y_true, y_pred):
    y_true = tf.eye(tf.shape(y_pred)[0])
    q_embedding, a_embedding = tf.unstack(y_pred, axis=1)
    similarity_matrix = tf.matmul(
        q_embedding, a_embedding, transpose_b=True)
    y_true = tf.argmax(y_true, axis=0)
    y_pred = tf.argmax(similarity_matrix, axis=0)
    return tf.keras.losses.categorical_crossentropy(y_true, similarity_matrix)
