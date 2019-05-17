import argparse
import os
from glob import glob

import pandas as pd
import numpy as np

from docproduct.predictor import QAEmbed


def read_all(data_path):
    glob_pattern = os.path.join(data_path, '*.csv')
    df_list = []
    for f in glob(glob_pattern):
        print('Reading {0}'.format(f))
        if os.path.basename(f) == 'healthtap_data_cleaned.csv':
            df = pd.read_csv(f, lineterminator='\n')
            df.columns = ['index', 'question', 'answer']
            df.drop(columns=['index'], inplace=True)
        else:
            df = pd.read_csv(f, encoding='utf8', lineterminator='\n')
        try:
            df.drop(columns=['question_bert', 'answer_bert'], inplace=True)
        except:
            pass
        df_list.append(df)
    return pd.concat(df_list, axis=0)


def train_data_to_embedding(model_path='models/bertffn_crossentropy/bertffn',
                            data_path='data/mqa_csv',
                            output_path='qa_embeddings/bertffn_crossentropy.zip',
                            pretrained_path='models/pubmed_pmc_470k/'):
    """Function to generate similarity embeddings for QA pairs.

    Input file format:
        question,answer
        my eyes hurts, go see a doctor

    Keyword Arguments:
        model_path {str} -- Similarity embedding model path (default: {'models/bertffn_crossentropy/bertffn'})
        data_path {str} -- CSV data path (default: {'data/mqa_csv'})
        output_path {str} -- Embedding output path (default: {'qa_embeddings/bertffn_crossentropy.zip'})
        pretrained_path {str} -- Pretrained BioBert model path (default: {'models/pubmed_pmc_470k/'})
    """
    if os.path.basename(model_path) == 'ffn':
        ffn_weight_file = model_path
    else:
        ffn_weight_file = None

    if os.path.basename(model_path) == 'bertffn':
        bert_ffn_weight_file = model_path
    else:
        bert_ffn_weight_file = None
    embeder = QAEmbed(
        pretrained_path=pretrained_path,
        ffn_weight_file=ffn_weight_file,
        bert_ffn_weight_file=bert_ffn_weight_file
    )
    qa_df = read_all(data_path)
    qa_df.dropna(inplace=True)
    qa_vectors = embeder.predict(
        questions=qa_df.question.tolist(),
        answers=qa_df.answer.tolist())

    q_embedding, a_embedding = np.split(qa_vectors, 2, axis=1)
    qa_df['Q_FFNN_embeds'] = np.squeeze(q_embedding).tolist()
    qa_df['A_FFNN_embeds'] = np.squeeze(a_embedding).tolist()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    qa_df.to_parquet(output_path, index=False)


if __name__ == "__main__":

    train_data_to_embedding()
