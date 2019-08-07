import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence

from info_clusters.encoders.model_utils import *

UNK = 'UNK'

class AUTO(nn.Module):


    def __init__(self, labelset_size, embedding_dim, num_feature_maps, kernel_size, vocab_size, dropout, encoder_dim, decoder_hidden_dim, stride=1, padding=1, dilation=1):
        super(AUTO, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_feature_maps = num_feature_maps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dropout = dropout

        self.encoder_dim = encoder_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        # define the encoder layers
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_feature_maps, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(in_features=num_feature_maps, out_features=encoder_dim)

        self.dropout_layer = nn.Dropout(p=dropout)

        # define the decoder layers
        self.lstm = nn.LSTM(encoder_dim, decoder_hidden_dim)

    def encode(self, x):

        #embedding
        embedded_seqs = self.embedding(x)

        # transpose the embedded sequences such that it fits [bs, in_channels, seq_len]
        embedded_seqs_transposed = embedded_seqs.transpose(dim0=1, dim1=2)

        # convolution
        out = self.conv(embedded_seqs_transposed)

        # max pooling
        pooled_output = nn.MaxPool1d(kernel_size=x.data.shape[-1])(out)

        # transpose the pooled output such that it fits [bs, _, num_feature_maps]
        # transposed_output = pooled_out.reshape(pooled_out.size(0), -1)
        transposed_output = pooled_output.transpose(dim0=1, dim1=2)
        output = self.fc1(transposed_output)
        return output

    def decode(self, x_encoded):

        def init_hidden(self, batch_size):
            return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                    autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))



def evaluate_validation_set(model, seqs, golds, lengths, sentences ,criterion):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in seqs2minibatches(seqs, golds, lengths, sentences, batch_size=1):
        #for r,t in zip(raw_data,targets):
        #    print(r, t)
        batch, targets, lengths = sort_batch(batch, targets, lengths)
        pred = model(batch)

        loss = criterion(pred, targets)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    per_class_p_r(cm)

    return total_loss.data.float()/len(seqs), acc


if __name__=="__main__":

    seed = 42
    num_epochs = 50
    batch_size = 256
    embedding_dim = 10
    num_feature_maps = 10
    kernel_size = 3
    lr = 0.1
    p = 0.2

    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logging()

    feature_extractor = sents2seqs

    d = load_json('/home/mareike/PycharmProjects/sheffield/data/test_data/duch_vernite.json')
    #d = load_json('/home/mareike/PycharmProjects/frames/code/data/mh17/mh17_80_20_20.json')
    sentences =d['train']['seq']
    #labels = ['0' if l[0] == '3' else '1' for l in d['train']['label']]

    labels = d['train']['label']

    #sentences, labels = upsample(sentences, labels)
    print(print_class_distributions(labels))

    dev_sentences = d['dev']['seq']
    dev_labels = d['dev']['label']
    #dev_labels =['0' if l[0] == '3' else '1' for l in  d['dev']['label']]

    print(print_class_distributions(dev_labels))



    # prepare train set
    seqs, lengths, word2idx = feature_extractor(sentences, None)
    logging.info('Vocabulary has {} entries'.format(len(word2idx)))
    logging.info(word2idx)
    golds, labelset = prepare_labels(labels, None)

    # prepare dev set
    dev_seqs, dev_lengths, _ = feature_extractor(dev_sentences, word2idx)
    dev_golds, _ = prepare_labels(dev_labels, labelset)

    # upsample the dev data
    dev_sentences_upsampled, dev_labels_upsampled = upsample(dev_sentences, dev_labels)
    print(print_class_distributions(dev_labels_upsampled))
    dev_seqs_upsampled, dev_lengths_upsampled, _ = sents2seqs(dev_sentences, word2idx)
    dev_golds_upsampled, _ = prepare_labels(dev_labels_upsampled, labelset)


    model = CNN(embedding_dim=embedding_dim, num_feature_maps=num_feature_maps, kernel_size=kernel_size,
                vocab_size=len(word2idx), labelset_size=len(labelset), dropout=p)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dev_loss, dev_acc = evaluate_validation_set(model, dev_seqs, dev_golds, dev_lengths, dev_sentences, loss_function)
    logging.info('Epoch {}: val acc {:.4f}'.format(-1, dev_acc))


    for epoch in range(num_epochs):
        preds = []
        gold_labels = []
        total_loss = 0
        for seqs_batch, gold_batch, lengths_batch, raw_batch in seqs2minibatches(seqs, golds, lengths, sentences, batch_size=batch_size):
            seqs_batch, gold_batch, lengths_batch = sort_batch(seqs_batch, gold_batch, lengths_batch)
            model.zero_grad()
            out = model(seqs_batch)
            loss = loss_function(out, gold_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss

            pred_idx = torch.max(out, 1)[1]
            gold_labels += list(gold_batch.int())
            preds += list(pred_idx.data.int())

        # predict the train data
        train_acc = accuracy_score(gold_labels, preds)
        # predict the val data
        dev_loss, dev_acc = evaluate_validation_set(model, dev_seqs, dev_golds, dev_lengths, dev_sentences, loss_function)
        dev_loss_up, dev_acc_up = evaluate_validation_set(model, dev_seqs_upsampled, dev_golds_upsampled, dev_lengths_upsampled, dev_sentences_upsampled, loss_function)
        logging.info('Epoch {}: Train loss {:.4f}, train acc {:.4f}, val_up loss {:.4f},  val_up acc {:.4f}, val loss {:.4f},  val acc {:.4f}'.format(epoch, total_loss.data.float()/len(seqs), train_acc, dev_loss_up, dev_acc_up, dev_loss, dev_acc))

    # Prepare test data
    test_sentences = d['test']['seq']
    test_labels = d['test']['label']

    test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
    test_golds, _ = prepare_labels(test_labels, labelset)
    test_loss, test_acc = evaluate_validation_set(model, test_seqs, test_golds, test_lengths, test_sentences, loss_function)
    logging.info('Epoch {}: Test loss {:.4f}, test acc {:.4f}'.format(epoch, test_loss, test_acc))