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

import argparse
import os

import numpy as np
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

import nsml
from dataset import MovieReviewDataset, preprocess2
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
import DataProcessing as DP
import LSTMClassifier as LSTMC

np.random.seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)

def RMSE(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2)))

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('[*] Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        dictionary = DP.Dictionary()
        data = preprocess2(raw_data, config.strmaxlen, dictionary)
        model.eval()
        model.batch_size = len(data)
        model.hidden = model.init_hidden()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        data = torch.LongTensor(np.array(data))
        if GPU_NUM:
            data = data.cuda()
        output_prediction = model(Variable(data).t())
        point = output_prediction.data.squeeze(dim=1).tolist()

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)

def adjust_lr(optimizer, epoch, lr):
    lr = lr * (0.7 ** (epoch // 9))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=25)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=20)
    args.add_argument('--embedding', type=int, default=128)
    args.add_argument('--hidden_dim', type=int, default=50)
    args.add_argument('--lr', type=float, default=0.01)
    args.add_argument('--num_layers', type=int, default=1)
    args.add_argument('--cnn', type=bool, default=False)
    args.add_argument('--dropout_rate', type=float, default=0.1)
    args.add_argument('--dr_embed', type=float, default=0.35)
    args.add_argument('--bi', type=bool, default=False)
    args.add_argument('--total_train', type=int, default=1810000)

    config = args.parse_args()

    use_gpu = GPU_NUM or torch.cuda.is_available()
    lr = float(config.lr)

    dictionary = DP.Dictionary()
    nlabel = 10
    total_train = config.total_train

    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=config.embedding, hidden_dim=config.hidden_dim,
                           vocab_size=len(dictionary),label_size=nlabel,
                           batch_size=config.batch, use_gpu=use_gpu, num_layers=config.num_layers,
                           cnn=config.cnn, dropout_rate=config.dropout_rate, bi=config.bi, dr_embed=config.dr_embed)
    if use_gpu:
        model = model.cuda()

    print('[!] use_gpu:', use_gpu)
    print('[!] available gpus:', torch.cuda.device_count())

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    ### Training mode
    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
            DATASET_PATH = '../sample_data/movie_review/'
        corpus = DP.Corpus(DATASET_PATH, total_train)
        print('[*]', 'Load corpus')

        # Load training data
        train_dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen, True, corpus)
        print('[*]', 'Load train dataset')
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=1)
        total_train = len(train_loader)

        # Load validation data
        test_dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen, True, corpus, test=True)
        print('[*]', 'Load test dataset')
        test_loader = DataLoader(dataset=test_dataset,
                                  batch_size=config.batch,
                                  collate_fn=collate_fn,
                                  num_workers=1)
        total_test = len(test_loader)

        print('[!]', '# of train:', len(train_dataset), '# of test:', len(test_dataset))

        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()
        #loss_function = RMSE
        #loss_function = nn.CrossEntropyLoss()

        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            print('='*20)
            print('[*] Epoch:', epoch)

            #optimizer = adjust_lr(optimizer, epoch, lr)
            train_avg_lost = 0.0

            for i, (data, labels) in enumerate(train_loader):
                data = torch.LongTensor(np.array(data))
                labels = torch.FloatTensor(np.array(labels))
                labels = torch.squeeze(labels)

                if use_gpu:
                    data = Variable(data.cuda())
                    labels = labels.cuda()

                model.batch_size = labels.shape[0]
                model.hidden = model.init_hidden()
                output = model(data.t())

                loss = loss_function(output, Variable(labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_avg_lost += loss.data[0]

            print('[*] Train Loss:', float(train_avg_lost/total_train))
            #print('[*] PR:', ["%0.1f" % i for i in output.data.squeeze(dim=1).tolist()[:5]])
            #print('[*] GT:', ["%0.1f" % i for i in labels.tolist()[:5]])

            ### Validation Step
            test_avg_lost = 0.0
            test_score_gt = np.array([])
            test_score_pr = np.array([])
            for i, (data, labels) in enumerate(test_loader):
                data = torch.LongTensor(np.array(data))
                labels = torch.FloatTensor(np.array(labels))
                labels = torch.squeeze(labels)

                if use_gpu:
                    data = Variable(data.cuda())
                    labels = labels.cuda()

                model.batch_size = labels.shape[0]
                model.hidden = model.init_hidden()
                output = model(data.t())

                loss = loss_function(output, Variable(labels))
                test_avg_lost += loss.data[0]

                test_score_gt = np.concatenate((test_score_gt, (Variable(labels).data).cpu().numpy()), axis=0)
                test_score_pr = np.concatenate((test_score_pr, (torch.squeeze(output.data)).cpu().numpy()), axis=0)

            #print('-'*20)
            print('[*] Test Loss:', float(test_avg_lost/total_test))
            print('VAL:GT', '0~5', '5~10', np.mean(test_score_gt[test_score_gt < 5]), np.mean(test_score_gt[test_score_gt > 5]))
            print('VAL:PR', '0~5', '5~10', np.mean(test_score_pr[test_score_pr < 5]), np.mean(test_score_pr[test_score_pr > 5]))
            print('VAL:GT', '0~3', '7~10', np.mean(test_score_gt[test_score_gt < 3]), np.mean(test_score_gt[test_score_gt > 7]))
            print('VAL:PR', '0~3', '7~10', np.mean(test_score_pr[test_score_pr < 3]), np.mean(test_score_pr[test_score_pr > 7]))
            #print('[*] GT:', float(test_score_gt/total_test), 'PR:', float(test_score_pr/total_test))
            #print('[*] PR:', ["%0.1f" % i for i in output.data.squeeze(dim=1).tolist()[:5]])
            #print('[*] GT:', ["%0.1f" % i for i in labels.tolist()[:5]])

            ### Log intermediate results(loss, accuracy)
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        loss__train=float(train_avg_lost/total_train),
                        loss__val=float(test_avg_lost/total_test),
                        step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    ### Test Mode
    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)

        print('='*20)
        print('[*] Test Results:')
        print('[*]', res)
