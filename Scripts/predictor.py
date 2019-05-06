import os
from collections import defaultdict

import tensorflow as tf
import numpy as np
from time import time
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import faiss

from Scripts.dataset import convert_text_to_feature
from Scripts.models import MedicalQAModelwithBert
from Scripts.tokenization import FullTokenizer
from keras_bert.loader import checkpoint_loader
from gpt_2 import gpt2


def load_weight(model, bert_ffn_weight_file=None, ffn_weight_file=None):
    if bert_ffn_weight_file:
        model.load_weights(bert_ffn_weight_file)
    elif ffn_weight_file:
        loader = checkpoint_loader(ffn_weight_file)
        model.get_layer('q_ffn').set_weights(
            [loader('q_ffn/ffn_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE'),
             loader('q_ffn/ffn_layer/bias/.ATTRIBUTES/VARIABLE_VALUE')])
        model.get_layer('a_ffn').set_weights(
            [loader('a_ffn/ffn_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE'),
             loader('a_ffn/ffn_layer/bias/.ATTRIBUTES/VARIABLE_VALUE')]
        )


class QAEmbed(object):
    def __init__(
            self,
            hidden_size=768,
            dropout=0.2,
            residual=True,
            pretrained_path=None,
            batch_size=128,
            max_seq_length=256,
            ffn_weight_file=None,
            bert_ffn_weight_file=None,
            load_pretrain=True,
            with_question=True,
            with_answer=True):
        super(QAEmbed, self).__init__()

        config_file = os.path.join(pretrained_path, 'bert_config.json')
        if load_pretrain:
            checkpoint_file = os.path.join(
                pretrained_path, 'biobert_model.ckpt')
        else:
            checkpoint_file = None

        # the ffn model takes 2nd to last layer
        if bert_ffn_weight_file is None:
            layer_ind = -2
        else:
            layer_ind = -1

        self.model = MedicalQAModelwithBert(
            hidden_size=768,
            dropout=0.2,
            residual=True,
            config_file=config_file,
            checkpoint_file=checkpoint_file,
            layer_ind=layer_ind)
        self.batch_size = batch_size
        self.tokenizer = FullTokenizer(
            os.path.join(pretrained_path, 'vocab.txt'))
        self.max_seq_length = max_seq_length

        # build mode in order to load
        question = 'fake' if with_question else None
        answer = 'fake' if with_answer else None
        self.predict(questions=question, answers=answer)
        load_weight(self.model, bert_ffn_weight_file, ffn_weight_file)

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

    def _make_inputs(self, questions=None, answers=None, dataset=True):

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
        if dataset:
            model_inputs = tf.data.Dataset.from_tensor_slices(model_inputs)
            model_inputs = model_inputs.batch(self.batch_size)

        return model_inputs

    def predict(self, questions=None, answers=None, dataset=True):

        # type check
        questions = self._type_check(questions)
        answers = self._type_check(answers)

        if questions is not None and answers is not None:
            assert len(questions) == len(answers)

        model_inputs = self._make_inputs(questions, answers, dataset)
        model_outputs = []

        if dataset:
            for batch in tqdm(iter(model_inputs), total=int(len(questions) / self.batch_size)):
                model_outputs.append(self.model(batch))
            model_outputs = np.concatenate(model_outputs, axis=0)
        else:
            model_outputs = self.model(model_inputs)
        return model_outputs


class FaissTopK(object):
    def __init__(self, embedding_file):
        super(FaissTopK, self).__init__()
        self.embedding_file = embedding_file
        self.df = pd.read_csv(self.embedding_file)
        self._get_faiss_index()
        self.df.drop(columns=["Q_FFNN_embeds", "A_FFNN_embeds"], inplace=True)

    def _get_faiss_index(self):
        with Pool(cpu_count()) as p:
            question_bert = p.map(eval, self.df["Q_FFNN_embeds"].tolist())
            answer_bert = p.map(eval, self.df["A_FFNN_embeds"].tolist())
        #question_bert = self.df["Q_FFNN_embeds"].tolist()
        #answer_bert = self.df["A_FFNN_embeds"].tolist()
        question_bert = np.array(question_bert)
        answer_bert = np.array(answer_bert)

        question_bert = question_bert.astype('float32')
        answer_bert = answer_bert.astype('float32')

        self.answer_index = faiss.IndexFlatIP(answer_bert.shape[-1])

        self.question_index = faiss.IndexFlatIP(question_bert.shape[-1])

        self.answer_index.add(answer_bert)
        self.question_index.add(question_bert)

    def predict(self, q_embedding, search_by='answer', topk=5, answer_only=True):
        if search_by == 'answer':
            _, index = self.answer_index.search(
                q_embedding.astype('float32'), topk)
        else:
            _, index = self.question_index.search(
                q_embedding.astype('float32'), topk)

        output_df = self.df.iloc[index[0], :]
        if answer_only:
            return output_df.answer.tolist()
        else:
            return (output_df.question.tolist(), output_df.answer.tolist())


class RetreiveQADoc(object):
    def __init__(self,
                 pretrained_path=None,
                 ffn_weight_file=None,
                 bert_ffn_weight_file='models/bertffn_crossentropy/bertffn',
                 embedding_file='qa_embeddings/bertffn_crossentropy.pkl'
                 ):
        super(RetreiveQADoc, self).__init__()
        self.qa_embed = QAEmbed(
            pretrained_path=pretrained_path,
            ffn_weight_file=ffn_weight_file,
            bert_ffn_weight_file=bert_ffn_weight_file
        )
        self.faiss_topk = FaissTopK(embedding_file)

    def predict(self, questions, search_by='answer', topk=5, answer_only=True):
        embedding = self.qa_embed.predict(questions=questions)
        return self.faiss_topk.predict(embedding, search_by, topk, answer_only)

    def getEmbedding(self, questions, search_by='answer', topk=5, answer_only=True):
        embedding = self.qa_embed.predict(questions=questions)
        return embedding


class GenerateQADoc(object):
    def __init__(self,
                 pretrained_path='pubmed_pmc_470k/',
                 ffn_weight_file=None,
                 bert_ffn_weight_file='models/bertffn_crossentropy/bertffn',
                 embedding_file='qa_embeddings/bertffn_crossentropy.pkl'
                 ):
        super(GenerateQADoc, self).__init__()
        tf.compat.v1.disable_eager_execution()
        self.sess = gpt2.start_tf_sess()
        with self.sess.as_default():
            self.qa_embed = QAEmbed(
                pretrained_path=pretrained_path,
                ffn_weight_file=ffn_weight_file,
                bert_ffn_weight_file=bert_ffn_weight_file,
                with_answer=False,
                load_pretrain=False
            )
        self.faiss_topk = FaissTopK(embedding_file)

        self.gpt2 = gpt2.load_gpt2(self.sess)

    def _get_gpt2_inputs(self, question, answer, questions, answers):
        assert len(questions) == len(answers)
        line = '`QUESTION: %s `ANSWER: %s' % (question, answer)
        for q, a in zip(questions, answers):
            line = '`QUESTION: %s `ANSWER: %s ' % (q, a) + line
        return line

    def predict(self, questions, search_by='answer', topk=5, answer_only=True):
        embedding = self.qa_embed.predict(
            questions=questions, dataset=False).eval(session=self.sess)
        if answer_only:
            topk_answer = self.faiss_topk.predict(
                embedding, search_by, topk, answer_only)
        else:
            topk_question, topk_answer = self.faiss_topk.predict(
                embedding, search_by, topk, answer_only)

        gpt2_input = self._get_gpt2_inputs(questions, topk_answer)
        raw_output = self.gpt2.generate(self.sess, prefix=gpt2_input)
        clipped_output = raw_output.split('`QUESTION')[1].split('`ANSWER:')[1]
        return clipped_output
