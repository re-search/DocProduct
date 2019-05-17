import os
import json
from shutil import copyfile

import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow_estimator as tf_estimator

import gpt2_estimator

DEVICE = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]


def train_gpt2(
        model_dir='models/gpt2',
        pretrained_path='models/117M',
        steps=100000,
        batch_size=4,
        num_gpu=4,
        learning_rate=0.0001):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=DEVICE[:num_gpu])
    learning_rate = learning_rate*1.5**num_gpu
    session_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    config = tf_estimator.estimator.RunConfig(
        session_config=session_config,
        train_distribute=mirrored_strategy,
        eval_distribute=mirrored_strategy,
        log_step_count_steps=500)

    gpt2_model_fn = gpt2_estimator.get_gpt2_model_fn(
        accumulate_gradients=5,
        learning_rate=learning_rate,
        length=512,
        batch_size=batch_size,
        temperature=0.7,
        top_k=0
    )
    copyfile(os.path.join(pretrained_path, 'hparams.json'),
             model_dir)
    copyfile(os.path.join(pretrained_path, 'vocab.bpe'),
             model_dir)
    copyfile(os.path.join(pretrained_path, 'encoder.json'),
             model_dir)
    hparams = gpt2_estimator.default_hparams()
    with open(os.path.join(pretrained_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    estimator = tf_estimator.estimator.Estimator(
        gpt2_model_fn,
        model_dir=model_dir,
        params=hparams,
        config=config)

    restore_hook = gpt2_estimator.RestoreCheckpointHook(pretrained_path)
    estimator.train(
        lambda: gpt2_estimator.train_input_fn(batch_size=batch_size), max_steps=steps, hooks=[restore_hook])

    # keep as an example
    # pred = estimator.predict(
    #     lambda: gpt2_estimator.predict_input_fn(
    #         'i am sick', batch_size=batch_size)
    # )


if __name__ == "__main__":
    train_gpt2(steps=5000000)
