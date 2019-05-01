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


def main(args):
    qa = pd.read_hdf(args.data_path, key='qa_embedding')

    with Pool(cpu_count()) as p:
        question_bert = p.map(eval, qa["Q_FFNN_embeds"].tolist())
        answer_bert = p.map(eval, qa["A_FFNN_embeds"].tolist())

    question_bert = np.array(question_bert)
    answer_bert = np.array(answer_bert)

    question_bert = question_bert.astype('float32')
    answer_bert = answer_bert.astype('float32')

    answer_index = faiss.IndexFlatIP(answer_bert.shape[-1])

    question_index = faiss.IndexFlatIP(question_bert.shape[-1])

    answer_index.add(answer_bert)
    question_index.add(question_bert)

    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(
        args.output_path, os.path.basename(args.data_path))

    output = open(output_path, "w")
    writer = csv.writer(output)

    firstrow = ['question', 'answer']
    for ii in range(0, args.number_samples):
        firstrow.append('question'+str(ii))
        firstrow.append('answer'+str(ii))

    writer.writerow(firstrow)

    def topKforGPT2(start_ind, end_ind, topk):
        D1, I1 = answer_index.search(
            question_bert[start_ind:end_ind].astype('float32'), topk)
        D2, I2 = question_index.search(
            question_bert[start_ind:end_ind].astype('float32'), topk)
        return I1, I2

    steps = ceil(qa.shape[0] / args.batch_size)

    # for k in tqdm(range(1000), mininterval=30, maxinterval=60):
    for k in tqdm(range(0, qa.shape[0], args.batch_size), total=steps):
        start_ind = k
        end_ind = k+args.batch_size

        a_batch_index, q_batch_index = topKforGPT2(
            start_ind, end_ind, int(args.number_samples/2))
        for a_index, q_index in zip(a_batch_index, q_batch_index):
            rowfill = []
            rowfill.append(qa["question"][k])
            rowfill.append(qa["answer"][k])
            aaa = qa.loc[list(a_index), :]
            qqq = qa.loc[list(q_index), :]
            aaaa = [*sum(zip(list(aaa['question']), list(aaa['answer'])), ())]
            qqqq = [*sum(zip(list(qqq['question']), list(qqq['answer'])), ())]
            finalfill = aaaa+qqqq
            rowfill = rowfill + finalfill
            writer.writerow(rowfill)
    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='qa_embeddings/ffn_crossentropy.h5', help='path of input csv files')
    parser.add_argument('--output_path', type=str,
                        default='gpt2_train_data/')
    parser.add_argument('--number_samples', type=int,
                        default=10)
    parser.add_argument('--batch_size', type=int,
                        default=512)

    args = parser.parse_args()
    main(args)
