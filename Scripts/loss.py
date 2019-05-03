import tensorflow as tf


def qa_pair_loss(y_true, y_pred):
    y_true = tf.eye(tf.shape(y_pred)[0])*2-1
    q_embedding, a_embedding = tf.unstack(y_pred, axis=1)
    q_embedding = q_embedding / \
        tf.norm(q_embedding, axis=-1, keepdims=True)
    a_embedding = a_embedding / \
        tf.norm(a_embedding, axis=-1, keepdims=True)
    similarity_matrix = tf.matmul(
        q_embedding, a_embedding, transpose_b=True)
    return tf.reduce_mean(tf.norm(y_true - similarity_matrix, axis=-1))


def qa_pair_cross_entropy_loss(y_true, y_pred):
    y_true = tf.eye(tf.shape(y_pred)[0])
    q_embedding, a_embedding = tf.unstack(y_pred, axis=1)
    similarity_matrix = tf.matmul(
        a=q_embedding, b=a_embedding, transpose_b=True)
    similarity_matrix_softmaxed = tf.nn.softmax(similarity_matrix)
    return tf.keras.losses.categorical_crossentropy(y_true, similarity_matrix_softmaxed, from_logits=False)
