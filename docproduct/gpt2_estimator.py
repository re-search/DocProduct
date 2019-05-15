# import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow_estimator as tf_estimator
import os
import numpy as np

from gpt_2.gpt2.src import model, sample, encoder
from gpt_2.gpt2.src.mqa_load_dataset import load_dataset, Sampler
from gpt_2.gpt2.src.accumulate import AccumulatingOptimizer


@tf.function
def accumulate_gradient(accumulated_steps, accumulate_gradients, opt_compute, opt_apply):
    if accumulated_steps >= accumulate_gradients:
        accumulated_steps.assign(0)
        return opt_apply
    else:
        accumulated_steps += 1
        return opt_compute


@tf.function
def reset_gradient(global_step, accumulate_gradients, opt_reset):
    accu = tf.floormod(global_step, accumulate_gradients)
    if accu == 0:
        return opt_reset
    else:
        return tf.no_op()


def get_gpt2_model_fn(
        accumulate_gradients,
        learning_rate,
        length,
        batch_size,
        temperature,
        top_k):

    def model_fn(features, labels, mode, params):
        context = features['context']

        if mode == tf_estimator.estimator.ModeKeys.TRAIN or mode == tf_estimator.estimator.ModeKeys.EVAL:
            loss_mask = features['loss_mask']
            output = model.model(hparams=params, X=context)
            loss_mask_float = tf.cast(loss_mask, tf.float32)
            # with loss mask -- reduce mean
            raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1])
            loss = tf.reduce_mean(
                loss_mask_float[:, :-1] * raw_loss)
            if mode == tf_estimator.estimator.ModeKeys.EVAL:
                return tf_estimator.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss)
            train_vars = [v for v in tf.compat.v1.trainable_variables()
                          if 'model' in v.name]
            if accumulate_gradients > 1:
                opt = AccumulatingOptimizer(
                    opt=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=learning_rate),
                    var_list=train_vars)

                global_step = tf.compat.v1.train.get_or_create_global_step()

                opt_compute = opt.compute_gradients(loss)
                opt_apply = opt.apply_gradients()
                opt_reset = opt.reset()

                # if apply gradient, clear gradient
                # accu = tf.cast(tf.mod(
                #     global_step, accumulate_gradients), tf.bool)
                accu = tf.equal(
                    tf.mod(global_step, accumulate_gradients), 0)

                opt_apply = tf.cond(
                    accu, true_fn=lambda: opt_apply, false_fn=lambda: opt_compute)
                opt_reset = tf.cond(
                    accu, true_fn=lambda: opt_reset, false_fn=lambda: 0.0
                )

                # opt_apply = accumulate_gradient(
                #     accumulated_steps, accumulate_gradients, opt_compute, opt_apply)
                # opt_reset = reset_gradient(
                #     accumulated_steps, accumulate_gradients, opt_reset)
                update_global_step = tf.compat.v1.assign(
                    global_step, global_step + 1, name='update_global_step')
                apply_and_reset = tf.group(
                    [opt_apply, opt_reset, update_global_step])
                tf.compat.v1.summary.scalar('loss', loss)
            else:
                apply_and_reset = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate).minimize(
                        loss, var_list=train_vars, global_step=tf.compat.v1.train.get_global_step())
                # tf.compat.v1.summary.scalar('loss', loss)
            return tf_estimator.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=apply_and_reset
            )
        elif mode == tf_estimator.estimator.ModeKeys.PREDICT:

            output = sample.sample_sequence(
                hparams=params, length=length,
                start_token=None,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k
            )[:, 1:]
            return tf_estimator.estimator.EstimatorSpec(
                mode=mode,
                predictions=output
            )

    return model_fn


def train_input_fn(checkpoint_path='models/117M', data_path='gpt2_train_data/bertffn_crossentropy_gpt2_train_data.zip', combine=50000, batch_size=8, max_seq_len=512):
    enc = encoder.get_encoder(checkpoint_path)
    chunks = load_dataset(enc, data_path, combine)
    data_sampler = Sampler(chunks)

    def sample_batch():
        while True:
            sampled_batch = [data_sampler.sample(
                max_seq_len) for _ in range(batch_size)]
            batch_len = min(max_seq_len, max([len(v) for v in sampled_batch]))
            batch_masks = np.zeros([batch_size, batch_len])
            for i, v in enumerate(sampled_batch):
                if len(v) > batch_len:
                    sampled_batch[i] = v[-batch_len:]
                mask_start = len(v) - list(v[::-1]).index(63) + 1
                # batch_masks[i,mask_start:len(v)] += 1 # without padding after endoftext
                # with padding after endoftext
                batch_masks[i, mask_start:] += 1
            if batch_size > 1:
                sampled_batch = np.asarray([
                    np.pad(v, [0, batch_len-len(v)],
                           'constant', constant_values=63)
                    for v in sampled_batch
                ], dtype=np.int32)

            yield {'context': sampled_batch, 'loss_mask': batch_masks}

    output_type = {'context': tf.int32, 'loss_mask': tf.int32}
    output_shape = {'context': (batch_size, None),
                    'loss_mask': (batch_size, None)}

    dataset = tf.data.Dataset.from_generator(
        sample_batch, output_types=output_type, output_shapes=output_shape)
    return dataset


def predict_input_fn(inputs, batch_size=8, checkpoint_path='models/117M'):
    enc = encoder.get_encoder(checkpoint_path)
    context_token = [enc.encode(inputs)]*batch_size
    output_shapes = {'context': (batch_size, None)}
    output_types = {'context': tf.int32}

    def g():
        yield {'context': context_token}
    return tf.data.Dataset.from_generator(g, output_shapes=output_shapes, output_types=output_types)


def serving_input_fn():
    features = {
        'context': tf.compat.v1.placeholder(tf.int32, [None, None]),
    }
    return tf_estimator.estimator.export.ServingInputReceiver(features, features)


def predictions_parsing(pred, enc):
    gen_texts = []
    for _, single_pred in enumerate(pred):
        gen_text = enc.decode(single_pred)
        gen_texts.append(gen_text)
    return gen_texts
