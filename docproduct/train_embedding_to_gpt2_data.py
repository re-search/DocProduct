import pandas as pd
import numpy as np
import os
import csv
from tqdm import tqdm
import argparse
from glob import glob
import faiss
from multiprocessing import Pool, cpu_count
from math import ceil


def train_embedding_to_gpt2_data(
    data_path='qa_embeddings/bertffn_crossentropy.zip',
    output_path='gpt2_train_data/bertffn_crossentropy_gpt2_train_data.zip',
    number_samples=10,
    batch_size=512
):
    qa = pd.read_parquet(data_path)
    question_bert = qa["Q_FFNN_embeds"].tolist()
    answer_bert = qa["A_FFNN_embeds"].tolist()
    question_bert = np.array(question_bert)
    answer_bert = np.array(answer_bert)

    question_bert = question_bert.astype('float32')
    answer_bert = answer_bert.astype('float32')

    answer_index = faiss.IndexFlatIP(answer_bert.shape[-1])

    question_index = faiss.IndexFlatIP(question_bert.shape[-1])

    faiss.normalize_L2(question_bert)
    faiss.normalize_L2(answer_bert)

    answer_index.add(answer_bert)
    question_index.add(question_bert)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_path = output_path + '.csv'
    output = open(csv_path, "w")
    writer = csv.writer(output)

    firstrow = ['question', 'answer']
    for ii in range(0, number_samples):
        firstrow.append('question'+str(ii))
        firstrow.append('answer'+str(ii))

    writer.writerow(firstrow)

    def topKforGPT2(start_ind, end_ind, topk):
        D1, I1 = answer_index.search(
            question_bert[start_ind:end_ind].astype('float32'), topk)
        D2, I2 = question_index.search(
            question_bert[start_ind:end_ind].astype('float32'), topk)
        return I1, I2

    steps = ceil(qa.shape[0] / batch_size)

    # for k in tqdm(range(1000), mininterval=30, maxinterval=60):
    for k in tqdm(range(0, qa.shape[0], batch_size), total=steps):
        start_ind = k
        end_ind = k+batch_size

        a_batch_index, q_batch_index = topKforGPT2(
            start_ind, end_ind, int(number_samples/2))
        for a_index, q_index in zip(a_batch_index, q_batch_index):
            rowfill = []
            rowfill.append(qa["question"].iloc[k])
            rowfill.append(qa["answer"].iloc[k])
            aaa = qa.iloc[list(a_index), :]
            qqq = qa.iloc[list(q_index), :]
            aaaa = [*sum(zip(list(aaa['question']), list(aaa['answer'])), ())]
            qqqq = [*sum(zip(list(qqq['question']), list(qqq['answer'])), ())]
            finalfill = aaaa+qqqq
            rowfill = rowfill + finalfill
            writer.writerow(rowfill)
    output.close()

    # ugly fix
    pd.read_csv(csv_path, lineterminator='\n').to_parquet(
        output_path, index=False)


if __name__ == "__main__":

    train_embedding_to_gpt2_data()
