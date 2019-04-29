import os.path
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gpt2

# make sure healthtapQAs.txt is in /data/
txt_path = 'gpt2/data/healthtapQAs.txt'

if not os.path.exists('models/117M'):
    gpt2.download_gpt2()

sess = gpt2.start_tf_sess()
gpt2.finetune(sess, txt_path, steps=1000)

gpt2.load_gpt2(sess)
gpt2.generate(sess, prefix="`QUESTION: What is the best treatment for the flu? `ANSWER:")
