import glob
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
import tqdm
import csv
import pandas as pd


def load_dataset(enc, path, combine, pretokenize=True, topk=10):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)
    if paths == []:
        raise Exception("No data found")

    token_chunks = []

    if pretokenize:

        pt_path = path.split('.')[0] + '_pretokenized.' + 'npy'

        if not os.path.exists(pt_path):

            print('Pretokenizing data..')

            token_list = []

            for path in paths:

                df = pd.read_parquet(path)

                for sample_ind, sample in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc='Pretokenization'):
                    line = '`QUESTION: %s `ANSWER: %s' % (
                        sample[0], sample[1])
                    for i in range(2, len(sample), 2):
                        if i <= topk*2:
                            line = '`QUESTION: %s `ANSWER: %s ' % (
                                sample[i], sample[i+1]) + line
                    line = line.replace('\n', '')
                    if sample_ind <= 10:
                        print(line)
                    token_list.append(np.stack(enc.encode(line)))

                print('Pretokenization successful!')
            np.save(pt_path, np.array(token_list))

        print('Loading pretokenized data..')
        token_chunks = np.load(pt_path, allow_pickle=True)

        # with open(pt_path, 'r', encoding='utf8') as pt:
        #     pt_reader = csv.reader(pt)
        #     pt_iter = list(pt_reader)

        #     for j, sample in enumerate(tqdm.tqdm(pt_iter[1:])):
        #         tokens = np.asarray(
        #             sample[-1].strip('[]').replace(',', '').split(), dtype=np.int32)
        #         token_chunks.append(tokens)

    else:
        raise NotImplementedError

        for path in paths:
            '''
            if path.endswith('.npz'):
                # Pre-encoded
                with np.load(path) as npz:
                    for item in npz.files:
                        token_chunks.append(npz[item])
            else:
                # Plain text
                with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                    raw_text += fp.read()
                if len(raw_text) >= combine:
                    tokens = np.stack(enc.encode(raw_text))
                    token_chunks.append(tokens)
                    raw_text = ''
                else:
                    raw_text += '<|endoftext|>'
            '''
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                csv_reader = csv.reader(fp)

                for j, sample in enumerate(tqdm.tqdm(csv_reader)):
                    line = '`QUESTION: %s `ANSWER: %s' % (
                        sample[0], sample[1])
                    for i in range(len(sample), 2, -2):
                        line = '`QUESTION: %s `ANSWER: %s ' % (
                            sample[i-2], sample[i-1]) + line
                    tokens = np.stack(enc.encode(line))
                    token_chunks.append(tokens)
        '''
        if raw_text:
            tokens = np.stack(enc.encode(raw_text))
            token_chunks.append(tokens)
        '''

    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        '''
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]
        '''
        return random.choice(self.chunks)
