import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu, num_layers, cnn, dropout_rate, bi, dr_embed):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.num_layers = num_layers
        self.cnn = cnn
        self.dropout_rate = dropout_rate
        self.dr_embed = dr_embed
        self.bi = bi

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if self.cnn:
            self.kernel_sizes = [
                [5, 300],
                [5, 256],
            ]
            self.nkernels = [256, 128]
            self.convs = nn.ModuleList([
                nn.Conv2d(1, Nk, Ks) for Ks, Nk in zip(self.kernel_sizes, self.nkernels)
            ])
            self.lstm = nn.LSTM(self.nkernels[-1], hidden_dim, dropout=dropout_rate,
             num_layers=num_layers, bidirectional=self.bi)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout_rate,
            num_layers=num_layers, bidirectional=self.bi)

        self.dropout = nn.Dropout()
        self.dropout_embed = nn.Dropout(self.dr_embed)
        if self.bi:
            self.fc1 = nn.Linear(hidden_dim*2, 10*label_size)
        else:
            self.fc1 = nn.Linear(hidden_dim, 10*label_size)
        self.fc2 = nn.Linear(10*label_size, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())
            if self.bi:
            	h0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).cuda())
            	c0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
            if self.bi:
            	h0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).cuda())
            	c0 = Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).cuda())
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.dropout_embed(embeds)

        if self.cnn:
            x = embeds.transpose(0, 1)
            for i, conv in enumerate(self.convs):
                x = x.unsqueeze(1)
                x = F.relu(conv(x)).squeeze(3)
                x = x.transpose(1, 2)
            x = x.transpose(0, 1)

            lstm_out, self.hidden = self.lstm(x, self.hidden)
        else:
            x = embeds.view(len(sentence), self.batch_size, -1)
            lstm_out, self.hidden = self.lstm(x, self.hidden)

        output  = self.fc1(lstm_out[-1])
        output = self.fc2(output)
        output = torch.sigmoid(output) * 9 + 1

        return output
