import pandas as pd
import numpy as np
import os
import csv
from tqdm import tqdm
import argparse
from glob import glob
import faiss


def main(args):
    qa = pd.read_csv(args.data_path)

    def fix_array(x):
        x = np.fromstring(
            x.replace('\n', '')
            .replace('[', '')
            .replace(']', '')
            .replace('  ', ' '), sep=' ')
        return x.reshape((1, 768))

    qa["Q_FFNN_embeds"] = qa["Q_FFNN_embeds"].apply(fix_array)
    qa["A_FFNN_embeds"] = qa["A_FFNN_embeds"].apply(fix_array)
    qa = qa.reset_index(drop=True)

    question_bert = np.concatenate(qa["Q_FFNN_embeds"].values, axis=0)
    answer_bert = np.concatenate(qa["A_FFNN_embeds"].values, axis=0)

    question_bert = question_bert.astype('float32')
    answer_bert = answer_bert.astype('float32')

    answer_index = faiss.IndexFlatIP(answer_bert.shape[-1])
    answer_index.add(answer_bert)

    question_index = faiss.IndexFlatIP(question_bert.shape[-1])
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

    def topKforGPT2(xf, kf):
        D1, I1 = answer_index.search(
            question_bert[xf:xf+1].astype('float32'), kf)
        D2, I2 = question_index.search(
            question_bert[xf:xf+1].astype('float32'), kf)
        return I1, I2

    # for k in tqdm(range(1000), mininterval=30, maxinterval=60):
    for k in tqdm(range(qa.shape[0]), mininterval=30, maxinterval=60):
        rowfill = []
        rowfill.append(qa["question"][k])
        rowfill.append(qa["answer"][k])
        aa, qq = topKforGPT2(k, int(args.number_samples/2))
        aaa = qa.loc[list(aa[0]), :]
        qqq = qa.loc[list(qq[0]), :]
        aaaa = [*sum(zip(list(aaa['question']), list(aaa['answer'])), ())]
        qqqq = [*sum(zip(list(qqq['question']), list(qqq['answer'])), ())]
        finalfill = aaaa+qqqq
        rowfill = rowfill + finalfill
        writer.writerow(rowfill)
    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='models/ffn_mse/ffn', help='path for trained models')
    parser.add_argument('--data_path', type=str,
                        default='qa_embeddings/ffn_crossentropy.csv', help='path of input csv files')
    parser.add_argument('--output_path', type=str,
                        default='gpt2_train_data/')
    parser.add_argument('--number_samples', type=int,
                        default=10)

    args = parser.parse_args()
    main(args)
