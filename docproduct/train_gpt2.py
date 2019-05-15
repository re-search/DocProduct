from gpt_2 import gpt2
import os
import json

import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow_estimator as tf_estimator

from docproduct import gpt2_estimator, ckpt_restore_hook

DEVICE = ["/gpu:0", "/gpu:1"]

tf.compat.v1.disable_eager_execution()


def train_gpt2(
        model_dir='models/gpt2',
        pretrained_path='models/117M',
        steps=100000,
        batch_size=2,
        num_gpu=1,
        learning_rate=0.0001):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    # tf.compat.v1.disable_v2_behavior()
    # if not os.path.exists('models/117M'):
    #     gpt2.download_gpt2()

    # sess = gpt2.start_tf_sess()
    # gpt2.finetune(sess, csv_path, steps=steps, batch_size=batch_size)
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICE[:num_gpu])
    learning_rate = learning_rate*1.5**num_gpu
    session_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True)
    session_config.gpu_options.allow_growth = False
    config = tf_estimator.estimator.RunConfig(
        session_config=session_config,
        train_distribute=mirrored_strategy,
        eval_distribute=mirrored_strategy,
        log_step_count_steps=5)

    gpt2_model_fn = gpt2_estimator.get_gpt2_model_fn(
        accumulate_gradients=5,
        learning_rate=learning_rate,
        length=512,
        batch_size=batch_size,
        temperature=0.7,
        top_k=0
    )
    hparams = gpt2.model.default_hparams()
    with open(os.path.join(pretrained_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    estimator = tf_estimator.estimator.Estimator(
        gpt2_model_fn,
        model_dir=model_dir,
        params=hparams,
        config=config)

    restore_hook = ckpt_restore_hook.RestoreCheckpointHook(pretrained_path)
    estimator.train(
        lambda: gpt2_estimator.train_input_fn(batch_size=batch_size), max_steps=steps, hooks=[restore_hook])
    pred = estimator.predict(
        lambda: gpt2_estimator.predict_input_fn('i am sick', batch_size=2)
    )
    # pred = estimator.predict(
    #     lambda: gpt2_estimator.train_input_fn(batch_size=batch_size)
    # )

    for p in pred:
        print(p)


if __name__ == "__main__":
    train_gpt2(steps=100)
