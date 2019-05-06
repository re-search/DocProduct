import os.path
import tensorflow.compat.v1 as tf
from gpt_2 import gpt2

tf.disable_eager_execution()

train = True

# make sure healthtapQAs.txt is in /data/
txt_path = 'data/healthtapQAs.txt'
csv_path = 'gpt2_train_data/bertffn_crossentropy.csv'

if not os.path.exists('models/117M'):
    gpt2.download_gpt2()

sess = gpt2.start_tf_sess()

if train:
    gpt2.finetune(sess, csv_path, steps=30000, batch_size=1)
else:
    gpt2.load_gpt2(sess)

gpt2.generate(sess, prefix="`QUESTION: What is the best treatment for the flu? `ANSWER:")
