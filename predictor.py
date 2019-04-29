import os
from collections import defaultdict

import tensorflow as tf
import numpy as np
from time import time

from dataset import convert_text_to_feature
from models import MedicalQAModelwithBert
from tokenization import FullTokenizer


class QAEmbed(object):
    def __init__(
            self,
            hidden_size=768,
            dropout=0.2,
            residual=True,
            pretrained_path=None,
            batch_size=128,
            max_seq_length=256):
        super(QAEmbed, self).__init__()
        config_file = os.path.join(pretrained_path, 'bert_config.json')
        checkpoint_file = os.path.join(pretrained_path, 'biobert_model.ckpt')
        self.model = MedicalQAModelwithBert(
            hidden_size=768,
            dropout=0.2,
            residual=True,
            config_file=config_file,
            checkpoint_file=checkpoint_file)
        self.batch_size = batch_size
        self.tokenizer = FullTokenizer(
            os.path.join(pretrained_path, 'vocab.txt'))
        self.max_seq_length = max_seq_length

    def _type_check(self, inputs):
        if inputs is not None:
            if isinstance(inputs, str):
                inputs = [inputs]
            elif isinstance(inputs, list):
                pass
            else:
                raise TypeError(
                    'inputs are supposed to be str of list of str, got {0} instead.'.format(type(inputs)))
            return inputs

    def _make_inputs(self, questions=None, answers=None):

        if questions:
            data_size = len(questions)
            q_feature_dict = defaultdict(list)
            for q in questions:
                q_feature = convert_text_to_feature(
                    q, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
                q_feature_dict['q_input_ids'].append(q_feature[0])
                q_feature_dict['q_input_masks'].append(q_feature[1])
                q_feature_dict['q_segment_ids'].append(q_feature[2])

        if answers:
            data_size = len(answers)
            a_feature_dict = defaultdict(list)
            for a in answers:
                a_feature = convert_text_to_feature(
                    a, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
                a_feature_dict['a_input_ids'].append(a_feature[0])
                a_feature_dict['a_input_masks'].append(a_feature[1])
                a_feature_dict['a_segment_ids'].append(a_feature[2])

        if questions and answers:
            q_feature_dict.update(a_feature_dict)
            model_inputs = q_feature_dict
        elif questions:
            model_inputs = q_feature_dict
        elif answers:
            model_inputs = a_feature_dict

        model_inputs = {k: tf.convert_to_tensor(
            np.stack(v, axis=0)) for k, v in model_inputs.items()}

        model_inputs = tf.data.Dataset.from_tensor_slices(model_inputs)
        model_inputs = model_inputs.batch(self.batch_size)

        return model_inputs

    def predict(self, questions=None, answers=None):

        # type check
        questions = self._type_check(questions)
        answers = self._type_check(answers)

        if questions is not None and answers is not None:
            assert len(questions) == len(answers)

        model_inputs = self._make_inputs(questions, answers)
        model_outputs = []
        for batch in iter(model_inputs):
            model_outputs.append(self.model(batch))
        model_outputs = np.concatenate(model_outputs, axis=0)
        return model_outputs
