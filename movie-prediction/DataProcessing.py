import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle

from konlpy.tag import Twitter
pos_tagger = Twitter()

def tokenize(doc):
    skip_pos = ['Josa', 'Punctuation']
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # pretrained vocabulary
        filename = "idx2word_final.pkl"

        with open(filename, 'rb') as f:
            self.idx2word = pickle.load(f)

        for i, word in enumerate(self.idx2word):
            self.word2idx[word] = i

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def to_vec(self, tokens, max_length):
        count = 0
        clip = False
        txt = [0]*max_length

        tokens = tokens + ['<eos>']
        for word in tokens:
            if word.strip() in self.word2idx:
                if count > max_length - 1:
                    clip = True
                    break
                txt[count] = self.word2idx[word.strip()]
                count += 1
            if clip: break
        return txt

class Corpus(object):
    def __init__(self, DATA_DIR, total_train):
        self.dictionary = Dictionary()
        self.total_train = total_train
        filename = "train_docs_final.pkl"
        self.make_pkl(DATA_DIR, filename)
        print('[*]', 'Complete loading pkl from', filename)

    def make_pkl(self, dataset_path, pickle_name):
        print('[*]', 'Make pkl...')
        train_data = os.path.join(dataset_path, 'train', 'train_data')
        train_label = os.path.join(dataset_path, 'train', 'train_label')

        reviews = []
        labels = []

        print('[*]', 'Read train_data', train_data)
        print('[*]', 'Read train_label', train_label)
        with open(train_data, 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        with open(train_label, 'rt', encoding='utf-8') as f:
            labels = f.readlines()

        assert len(reviews) == len(labels)

        import pickle
        print('[*]', 'Tokenize...')
        if not os.path.exists(pickle_name):
            with open(pickle_name, 'wb') as f:
                zipped = zip(reviews[:self.total_train], labels[:self.total_train])
                self.train_docs = [(tokenize(review), float(label)) for review, label in zipped]
                pickle.dump(self.train_docs, f)
        else:
            with open(pickle_name, 'rb') as f:
                self.train_docs = pickle.load(f)

class TxtDatasetProcessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):
        self.txt_path = os.path.join(data_path, txt_path)
        txt_filepath = os.path.join(data_path, txt_filename)
        fp = open(txt_filepath, 'r')
        self.txt_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels
        self.corpus = corpus
        self.sen_len = sen_len

    def __getitem__(self, index):
        filename = os.path.join(self.txt_path, self.txt_filename[index])
        fp = open(filename, 'r')
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))
        count = 0
        clip = False
        for words in fp:
            for word in words.split():
                if word.strip() in self.corpus.dictionary.word2idx:
                    if count > self.sen_len - 1:
                        clip = True
                        break
                    txt[count] = self.corpus.dictionary.word2idx[word.strip()]
                    count += 1
            if clip: break
        label = torch.LongTensor([self.label[index]])
        return txt, label

    def __len__(self):
        return len(self.txt_filename)
