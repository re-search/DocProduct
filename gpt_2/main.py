import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gpt2

sess = gpt2.start_tf_sess()
gpt2.finetune(sess, 'gpt2/data/healthtapQAs.txt', steps=1000) # have to put healthtapQAs.txt in gpt2/data/ for this to work

gpt2.load_gpt2(sess)
gpt2.generate(sess, prefix="`QUESTION: What is the best treatment for the flu? `ANSWER:")
