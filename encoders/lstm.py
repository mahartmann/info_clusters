import logging

import numpy as np
import torch
import configparser
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence

from info_clusters.encoders.model_utils import *
from info_clusters.encoders import param_reader

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





def evaluate_validation_set(model, seqs, golds, lengths, sentences ,criterion, labelset):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in seqs2minibatches(seqs, golds, lengths, sentences, batch_size=1):

        batch, targets, lengths = sort_batch(batch, targets, lengths)
        pred = model(batch.transpose(0,1), lengths)

        loss = criterion(pred, targets)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss

    results = p_r_f(y_true, y_pred, labelset)

    return total_loss.data.float()/len(seqs), results


def main(args):

    # read params from csv and update the arguments
    if args.hyperparam_csv != '':
        csv_params = param_reader.read_hyperparams_from_csv(args.hyperparam_csv, args.rowid)
        vars(args).update(csv_params)

    seed = args.seed
    num_epochs = args.epochs
    batch_size = args.bs
    embedding_dim = args.emb_dim

    lr = args.lr
    hidden_dim = args.hid_dim
    p = args.dropout
    embeddings_file = args.emb_file
    datafile = args.data
    max_vocab = args.max_vocab
    use_additional_data = args.additional_data

    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logging()

    log_params(vars(args))

    feature_extractor = sents2seqs

    if embeddings_file != '':
        pretrained_embeddings, word2idx, idx2word = load_embeddings_from_file(embeddings_file, max_vocab=max_vocab)
    else:
        pretrained_embeddings, word2idx, idx2word = None, None, None

    d = load_json(datafile)

    if use_additional_data is True:
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(args.config)
        additional_data_file = config.get('Files', 'additional_data')
        logging.info('Loading additional data from {}'.format(additional_data_file))
        additional_data = load_json(additional_data_file)
    else:
        additional_data = {'seq': [], 'label': []}
    sentences = [prefix_sequence(sent, 'en') for sent in d['train']['seq']] + [prefix_sequence(sent, 'ru') for sent in
                                                                               additional_data['seq']]
    labels = d['train']['label'] + additional_data['label']

    if args.upsample is True:
        logging.info('Upsampling the train data')
        sentences, labels = upsample(sentences, labels)

    dev_sentences = [prefix_sequence(sent, 'en') for sent in d['dev']['seq']]
    dev_labels = d['dev']['label']

    # prepare train set
    seqs, lengths, word2idx = feature_extractor(sentences, word2idx)
    logging.info('Vocabulary has {} entries'.format(len(word2idx)))
    logging.info(word2idx)
    golds, labelset = prepare_labels(labels, None)

    # prepare dev set
    dev_seqs, dev_lengths, _ = feature_extractor(dev_sentences, word2idx)
    dev_golds, _ = prepare_labels(dev_labels, labelset)

    # upsample the dev data
    dev_sentences_upsampled, dev_labels_upsampled = upsample(dev_sentences, dev_labels)
    dev_seqs_upsampled, dev_lengths_upsampled, _ = sents2seqs(dev_sentences, word2idx)
    dev_golds_upsampled, _ = prepare_labels(dev_labels_upsampled, labelset)

    model = LSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(word2idx), pretrained_embeddings=pretrained_embeddings, labelset_size=len(labelset), dropout=p)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dev_loss, results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds, lengths=dev_lengths,
                                                sentences=dev_sentences, criterion=loss_function, labelset=labelset)

    logging.info('Epoch {}: val f_macro {:.4f}'.format(-1, results['macro'][2]))
    logging.info('Summary val')
    logging.info(print_result_summary(results))

    best_epoch = 0
    best_macro_f = 0


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
        train_results = p_r_f(gold_labels, preds, labelset)
        # predict the val data
        dev_loss_up, dev_results_up = evaluate_validation_set(model=model, seqs=dev_seqs_upsampled,
                                                                  golds=dev_golds_upsampled,
                                                                  lengths=dev_lengths_upsampled,
                                                                  sentences=dev_sentences_upsampled,
                                                                  criterion=loss_function, labelset=labelset)
        dev_loss, dev_results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds,
                                                            lengths=dev_lengths,
                                                            sentences=dev_sentences, criterion=loss_function,
                                                            labelset=labelset)
        if dev_results['macro'][2] > best_macro_f:
            best_macro_f = dev_results['macro'][2]
            best_epoch = epoch

        logging.info(
                'Epoch {}: Train loss {:.4f}, train f_macro {:.4f}, val_up loss {:.4f},  val_up f_macro {:.4f}, val loss {:.4f},  val f_macro {:.4f}, best_epoch {}, best val_f_macro {:.4f}'.format(
                    epoch, total_loss.data.float() / len(seqs), train_results['macro'][2], dev_loss_up,
                    dev_results_up['macro'][2], dev_loss, dev_results['macro'][2],
                    best_epoch, best_macro_f))

        logging.info('Summary train')
        logging.info(print_result_summary(train_results))
        logging.info('Summary dev')
        logging.info(print_result_summary(dev_results))
        # logging.info('Summary dev_up')
        # logging.info(print_result_summary(dev_results_up))

    dev_results['best_epoch'] = best_epoch
    dev_results['best_macro_f'] = best_macro_f
    param_reader.write_results_and_hyperparams(args.result_csv, dev_results, vars(args))

    """                                                                                                          dev_acc_up, dev_loss, dev_acc))

    # Prepare test data
    test_sentences = d['test']['seq']
    test_labels = d['test']['label']

    test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
    test_golds, _ = prepare_labels(test_labels, labelset)
    test_loss, test_acc = evaluate_validation_set(model, test_seqs, test_golds, test_lengths, test_sentences, loss_function)
    logging.info('Epoch {}: Test loss {:.4f}, test acc {:.4f}'.format(epoch, test_loss, test_acc))
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tweet classification using CNN')
    parser.add_argument('--data', type=str,
                            default='/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/mh17_60_20_20.json',
                            help="File with main data")
    parser.add_argument('--additional_data', type=bool,
                        default=True,
                        help="Use additional train data. Location of the data is specified in the config")
    parser.add_argument('--config', type=str, default='config.cfg',
                        help="Config file")
    parser.add_argument('--exp_dir', type=str, default='out',
                            help="Path to experiment folder")
    parser.add_argument('--seed', type=int, default=42,
                            help="Random seed")
    parser.add_argument('--bs', type=int, default=256,
                            help="Batch size")
    parser.add_argument('--epochs', type=int, default=500,
                            help="Number of epochs")
    parser.add_argument('--emb_dim', type=int, default=300,
                            help="Embedding dimension")
    parser.add_argument('--hid_dim', type=int, default=300,
                        help="Dimension of LSMT hidden state")
    parser.add_argument('--upsample', type=bool_flag, default=True,
                            help="if enabled upsample the train data according to size of largest class")
    parser.add_argument('--lr', type=float, default=0.01,
                            help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.2,
                            help="Keep probability for dropout")
    parser.add_argument('--emb_file', type=str,
                            default='/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/resources/mh17_60_20_20_additional_embs.txt.prefixed',
                            help="File with pre-trained embeddings")
    parser.add_argument('--max_vocab', type=int, default=-1,
                            help="Maximum number of words read in from the pretrained embeddings. -1 to disable")
    parser.add_argument('--hyperparam_csv', type=str,
                            default='../hyperparams_lstm.csv',
                            help="File with hyperparams. If set, values for specified hyperparams are read from the csv")
    parser.add_argument('--result_csv', type=str,
                            default='../results_lstm.csv',
                            help="File the results and hyperparams are written to")
    parser.add_argument('--rowid', type=int,
                            default=2,
                            help="Row from which hyperparams are read")
    args = parser.parse_args()
    main(args)