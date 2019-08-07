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

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, pretrained_embeddings, labelset_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2out = nn.Linear(hidden_dim, labelset_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.init_emb(pretrained_embeddings)

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(-1))

        embeds = self.word_embeddings(batch)
        packed_input = pack_padded_sequence(embeds,lengths)

        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])

        output = self.hidden2out(output)
        output = self.softmax(output)
        return output

    def init_emb(self, pretrained_embeddings):
        if pretrained_embeddings is not None:
            self.word_embeddings.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
        else:
            self.word_embeddings.weight.data.normal_(0, 0.1)

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
        pred = model(batch.transpose(0,1), lengths)
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
    embedding_dim = 100
    hidden_dim = 100
    lr = 0.1
    p = 0.2
    embeddings_file = ''
    max_vocab = -1

    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logging()

    if embeddings_file != '':
        pretrained_embeddings, word2idx, idx2word = load_embeddings_from_file(embeddings_file, max_vocab=max_vocab)
    else:
        pretrained_embeddings, word2idx, idx2word = None, None, None

    feature_extractor = sents2charseqs

    d = load_json('/home/mareike/PycharmProjects/sheffield/data/test_data/duch_vernite.json')
    #d = load_json('/home/mareike/PycharmProjects/frames/code/data/mh17/mh17_80_20_20.json')
    sentences =d['train']['seq']
    #labels = ['0' if l[0] == '3' else '1' for l in d['train']['label']]

    labels = d['train']['label']

    sentences, labels = upsample(sentences, labels)
    print(print_class_distributions(labels))

    dev_sentences = d['dev']['seq']
    dev_labels = d['dev']['label']
    #dev_labels =['0' if l[0] == '3' else '1' for l in  d['dev']['label']]

    print(print_class_distributions(dev_labels))



    # prepare train set
    seqs, lengths, word2idx = feature_extractor(sentences, word2idx)
    logging.info('VoCabulary has {} entries'.format(len(word2idx)))
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

    model = LSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(word2idx), pretrained_embeddings=pretrained_embeddings, labelset_size=len(labelset), dropout=p)
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
            out = model(seqs_batch.transpose(0,1), lengths_batch)
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
        dev_loss_up, dev_acc_up = evaluate_validation_set(model, dev_seqs_upsampled, dev_golds_upsampled, dev_lengths_upsampled, dev_sentences_upsampled,
                                                    loss_function)
        logging.info('Epoch {}: Train loss {:.4f}, train acc {:.4f}, val_up loss {:.4f},  val_up acc {:.4f}, val loss {:.4f},  val acc {:.4f}'.format(epoch, total_loss.data.float()/len(seqs), train_acc, dev_loss_up,
                                                                                                              dev_acc_up, dev_loss, dev_acc))

    # Prepare test data
    test_sentences = d['test']['seq']
    test_labels = d['test']['label']

    test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
    test_golds, _ = prepare_labels(test_labels, labelset)
    test_loss, test_acc = evaluate_validation_set(model, test_seqs, test_golds, test_lengths, test_sentences, loss_function)
    logging.info('Epoch {}: Test loss {:.4f}, test acc {:.4f}'.format(epoch, test_loss, test_acc))