import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence

from info_clusters.encoders.model_utils import *



UNK = 'UNK'

class CNN(nn.Module):


    def __init__(self, labelset_size, embedding_dim, num_feature_maps, kernel_size, pretrained_embeddings, vocab_size, dropout, stride=1, padding=1, dilation=1):
        super(CNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_feature_maps = num_feature_maps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dropout = dropout

        # define the layers
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_feature_maps, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(in_features=num_feature_maps, out_features=labelset_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.init_emb(pretrained_embeddings)


    def forward(self, x):

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
        output= output.squeeze(1)
        output = self.softmax(output)

        return output

    def init_emb(self, pretrained_embeddings):
        if pretrained_embeddings is not None:
            self.embedding.weight = torch.nn.Parameter(torch.Tensor(pretrained_embeddings))
        else:
            self.embedding.weight.data.normal_(0, 0.1)



def evaluate_validation_set(model, seqs, golds, lengths, sentences ,criterion, labelset):
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

    results = p_r_f(y_true, y_pred, labelset)

    return total_loss.data.float()/len(seqs), results

def p_r_f(gold, preds, labelset):
    results = {}
    results['macro'] = precision_recall_fscore_support(gold, preds, average='macro')
    results['micro'] = precision_recall_fscore_support(gold, preds, average='micro')
    results['per_class'] = {}
    labels = list(set([int(pred) for pred in preds]))

    # it's a tuple with precision, recall, f-score elements
    per_class_results = precision_recall_fscore_support(gold, preds, average=None, labels=labels)

    for i, label in enumerate(labels):
        results['per_class'][labelset[label]] = [elm[i] for elm in per_class_results]
    for label in labelset:
        if label not in results['per_class']:
            results['per_class'][label] = [0 for elm in results['macro']]
    return results

def print_result_summary(results):
    s =  '\nLabel\tP\tR\tF\t\nMacro\t{:.4f}\t{:.4f}\t{:.4f}\nMicro\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['macro'][0],results['macro'][1],results['macro'][2],
                                                                        results['micro'][0], results['micro'][1], results['micro'][2])
    labels = sorted(results['per_class'].keys())
    for label in labels:
        s += '{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(label, results['per_class'][label][0],results['per_class'][label][1],results['per_class'][label][2])
    return s




if __name__=="__main__":

    seed = 42
    num_epochs = 500
    batch_size = 256
    embedding_dim = 300
    num_feature_maps = 10
    kernel_size = 3
    lr = 0.1
    p = 0.2
    embeddings_file = '/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/resources/mh17_60_20_20_additional_embs.txt.prefixed'
    max_vocab = -1

    additional_data_file = '/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/additional_traindata_1.json'

    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logging()

    feature_extractor = sents2seqs

    if embeddings_file != '':
        pretrained_embeddings, word2idx, idx2word = load_embeddings_from_file(embeddings_file, max_vocab=max_vocab)
        print(word2idx)
    else:
        pretrained_embeddings, word2idx, idx2word = None, None, None

    #d = load_json('/home/mareike/PycharmProjects/sheffield/data/test_data/duch_vernite.json')
    #d = load_json('/home/mareike/PycharmProjects/frames/code/data/mh17/mh17_80_20_20.json')
    d = load_json('/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/mh17_60_20_20.json')
    additional_data = load_json(additional_data_file)
    sentences = [prefix_sequence(sent, 'en') for sent in d['train']['seq']] + [prefix_sequence(sent, 'ru') for sent in additional_data['seq']]
    labels = d['train']['label'] + additional_data['label']
    #labels = ['0' if l[0] == '3' else '1' for l in d['train']['label']]
    #labels = [ l[0] for  l in d['train']['label']]

    sentences, labels = upsample(sentences, labels)
    print(print_class_distributions(labels))

    dev_sentences = [prefix_sequence(sent, 'en') for sent in d['dev']['seq']]
    dev_labels = d['dev']['label']
    #dev_labels =['0' if l[0] == '3' else '1' for l in  d['dev']['label']]
    #dev_labels = [ l[0] for  l in d['dev']['label']]
    print(print_class_distributions(dev_labels))



    # prepare train set
    seqs, lengths, word2idx = feature_extractor(sentences, word2idx)
    print(seqs)
    logging.info('Vocabulary has {} entries'.format(len(word2idx)))
    logging.info(word2idx)
    golds, labelset = prepare_labels(labels, None)

    # prepare dev set
    dev_seqs, dev_lengths, _ = feature_extractor(dev_sentences, word2idx)
    dev_golds, _ = prepare_labels(dev_labels, labelset)
    print(dev_seqs)

    # upsample the dev data
    dev_sentences_upsampled, dev_labels_upsampled = upsample(dev_sentences, dev_labels)
    print(print_class_distributions(dev_labels_upsampled))
    dev_seqs_upsampled, dev_lengths_upsampled, _ = sents2seqs(dev_sentences, word2idx)
    dev_golds_upsampled, _ = prepare_labels(dev_labels_upsampled, labelset)


    model = CNN(embedding_dim=embedding_dim, num_feature_maps=num_feature_maps, pretrained_embeddings=pretrained_embeddings, kernel_size=kernel_size,
                vocab_size=len(word2idx), labelset_size=len(labelset), dropout=p)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dev_loss, results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds, lengths=dev_lengths, sentences=dev_sentences, criterion=loss_function, labelset=labelset)

    logging.info('Epoch {}: val f_macro {:.4f}'.format(-1, results['macro'][2]))
    logging.info('Summary val')
    logging.info(print_result_summary(results))


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
        train_results = p_r_f(gold_labels, preds, labelset)
        # predict the val data


        dev_loss_up, dev_results_up = evaluate_validation_set(model=model, seqs=dev_seqs_upsampled, golds=dev_golds_upsampled, lengths=dev_lengths_upsampled,
                                                    sentences=dev_sentences_upsampled, criterion=loss_function, labelset=labelset)
        dev_loss, dev_results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds, lengths=dev_lengths,
                                                    sentences=dev_sentences, criterion=loss_function, labelset=labelset)
        logging.info('Epoch {}: Train loss {:.4f}, train f_macro {:.4f}, val_up loss {:.4f},  val_up f_macro {:.4f}, val loss {:.4f},  val f_macro {:.4f}'.format(epoch, total_loss.data.float()/len(seqs), train_results['macro'][2], dev_loss_up, dev_results_up['macro'][2], dev_loss, dev_results['macro'][2]))

        logging.info('Summary train')
        logging.info(print_result_summary(train_results))
        logging.info('Summary dev')
        logging.info(print_result_summary(dev_results))
        logging.info('Summary dev_up')
        logging.info(print_result_summary(dev_results_up))

    # Prepare test data
    test_sentences = [prefix_sequence(sent, 'en') for sent in d['test']['seq']]
    test_labels = d['test']['label']

    test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
    test_golds, _ = prepare_labels(test_labels, labelset)
    test_loss, test_acc = evaluate_validation_set(model, test_seqs, test_golds, test_lengths, test_sentences, loss_function)
    logging.info('Epoch {}: Test loss {:.4f}, test acc {:.4f}'.format(epoch, test_loss, test_acc))