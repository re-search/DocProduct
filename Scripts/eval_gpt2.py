import tensorflow.compat.v1 as tf
from gpt_2 import gpt2

tf.disable_eager_execution()

csv_path = 'data/GPT2_data_FFNN.csv'

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.generate(sess, prefix="`QUESTION: What is the best treatment for the flu? `ANSWER:")
