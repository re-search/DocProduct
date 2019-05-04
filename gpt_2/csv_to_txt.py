import glob
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
import tqdm
import csv

path = 'data/GPT2_data_FFNN.csv'
out_path = 'data/GPT2_data_FFNN.txt'

op = open(out_path, 'w', encoding='utf8')

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
raw_text = ''
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
        for j, sample in enumerate(tqdm.tqdm(list(csv_reader)[1:])):
            line = '`QUESTION: %s `ANSWER: %s' % (sample[0], sample[1])
            for i in range(len(sample), 2, -2):
                line = '`QUESTION: %s `ANSWER: %s ' % (sample[i-2], sample[i-1]) + line
            line = line.replace('\n', '')
            #if j == 1: print(line)
            op.write(line + '\n')
    '''
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks
    '''

op.close()