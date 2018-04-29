# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os

import numpy as np
from torch.utils.data import Dataset
import torch

import gensim

from konlpy.tag import Twitter
pos_tagger = Twitter()

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

class SentenceReader:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in codecs.open(self.filepath, encoding='utf-8'):
            yield line.split(' ')

class MovieReviewDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int, lstm=False, corpus=None, test=False):
        if test:
            data_review = os.path.join(dataset_path, 'train', 'train_data')
            data_label = os.path.join(dataset_path, 'train', 'train_label')

            # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
            with open(data_review, 'rt', encoding='utf-8') as f:
                self.reviews = preprocess2(f.readlines()[corpus.total_train:], max_length, corpus.dictionary)
            # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
            with open(data_label) as f:
                self.labels = [np.float32(x) for x in f.readlines()[corpus.total_train:]]
        else:
            self.reviews = [corpus.dictionary.to_vec(d[0], max_length) for d in corpus.train_docs]
            self.labels = [np.float32(d[1]) for d in corpus.train_docs]

        assert len(self.reviews) == len(self.labels)

    def __len__(self):
        """
        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """
        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.reviews[idx], self.labels[idx]

def preprocess2(data: list, max_length: int, dictionary):
    vectorized_data = []
    for datum in data:
        tokens = tokenize(datum)
        txt = dictionary.to_vec(tokens, max_length=max_length)
        vectorized_data.append(txt)
    return vectorized_data
