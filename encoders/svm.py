import torch.nn as nn
import torch.optim as optim

import configparser

from info_clusters.encoders import param_reader
from info_clusters.encoders.model_utils import *
from info_clusters.encoders.model_utils import p_r_f, print_result_summary, log_params

from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

UNK = 'UNK'

# klearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
# class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

def evaluate_validation_set(model, seqs, golds, lengths, sentences ,criterion, labelset):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in seqs2minibatches(seqs, golds, lengths, sentences, batch_size=1):

        batch, targets, lengths = sort_batch(batch, targets, lengths)
        pred = model(batch)

        loss = criterion(pred, targets)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss

    results = p_r_f(y_true, y_pred, labelset)

    return total_loss.data.float()/len(seqs), results


def embed(seq, embs):
    """
    embed a sentence as an average over word embeddings
    :param word2idx:
    :return:
    """
    sent_embedding = np.array([embs[i] for i in seq])
    return np.mean(sent_embedding, 0)


def main(args):

    # read params from csv and update the arguments
    if args.hyperparam_csv != '':
        csv_params = param_reader.read_hyperparams_from_csv(args.hyperparam_csv, args.rowid)
        vars(args).update(csv_params)


    seed = args.seed
    num_epochs = args.epochs
    batch_size = args.bs
    embedding_dim = args.emb_dim
    num_feature_maps = args.num_feature_maps
    kernel_size = args.ks
    lr = args.lr
    p = args.dropout
    use_pretrained_embeddings = args.embs
    datafile = args.data
    max_vocab = args.max_vocab
    additional_data_file = args.additional_data

    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logging(logfile='log_cnn_{}.log'.format(args.rowid))

    log_params(vars(args))

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    feature_extractor = sents2seqs

    if use_pretrained_embeddings is True:
        embeddings_file = config.get('Files', 'emb_file')
        logging.info('Loading pretrained embedding from {}'.format(embeddings_file))
        pretrained_embeddings, word2idx, idx2word = load_embeddings_from_file(embeddings_file, max_vocab=max_vocab)
    else:
        pretrained_embeddings, word2idx, idx2word = None, None, None

    d = load_json(datafile)

    if additional_data_file != '':
        additional_data_file = config.get('Files', 'additional_data_{}'.format(args.additional_data))
        logging.info('Loading additional data from {}'.format(additional_data_file))
        additional_data = load_json(additional_data_file)
    else:
        additional_data = {'seq':[], 'label':[]}
    sentences = [prefix_sequence(sent, 'en') for sent in d['train']['seq']] + [prefix_sequence(sent, 'ru') for sent in additional_data['seq']]
    labels = d['train']['label'] + additional_data['label']

    if args.upsample is True:
        logging.info('Upsampling the train data')
        sentences, labels = upsample(sentences, labels)


    dev_sentences = [prefix_sequence(sent, 'en') for sent in d['dev']['seq']]
    dev_labels = d['dev']['label']


    # prepare train set
    seqs, lengths, word2idx = feature_extractor(sentences, word2idx)
    embeded_seqs = np.vstack([embed(seq, pretrained_embeddings) for seq in seqs])
    logging.info('Vocabulary has {} entries'.format(len(word2idx)))
    logging.info(word2idx)
    golds, labelset = prepare_labels(labels, None)

    # prepare dev set
    dev_seqs, dev_lengths, _ = feature_extractor(dev_sentences, word2idx)
    embeded_dev_seqs = np.vstack([embed(seq, pretrained_embeddings) for seq in dev_seqs])
    dev_golds, _ = prepare_labels(dev_labels, labelset)


    model = SGDClassifier()
    logging.info('Fitting the data')
    model.fit(embeded_seqs, golds.numpy())
    logging.info('Predicting')
    train_preds = model.predict(embeded_seqs)

    dev_preds = model.predict(embeded_dev_seqs)

    # predict the train data

    train_results = p_r_f(golds, train_preds, labelset)
    # predict the val data
    dev_results = p_r_f(dev_golds, dev_preds, labelset)
    logging.info('Summary train')
    logging.info(print_result_summary(train_results))
    logging.info('Summary dev')
    logging.info(print_result_summary(dev_results))


    dev_results['best_epoch'] = 1

    dev_results['best_macro_f'] = dev_results['macro'][2]
    param_reader.write_results_and_hyperparams(args.result_csv, dev_results, vars(args))

    test_sentences = [prefix_sequence(sent, 'en') for sent in d['test']['seq']]
    test_labels = d['test']['label']
    # prepare test set
    test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
    embeded_test_seqs = np.vstack([embed(seq, pretrained_embeddings) for seq in test_seqs])
    test_golds, _ = prepare_labels(test_labels, labelset)
    test_preds = model.predict(embeded_test_seqs)
    # predict the train data
    test_results = p_r_f(test_golds, test_preds, labelset)
    logging.info('Summary test')
    logging.info(print_result_summary(test_results))

    """
    # Prepare test data
    test_sentences = [prefix_sequence(sent, 'en') for sent in d['test']['seq']]
    test_labels = d['test']['label']

    test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
    test_golds, _ = prepare_labels(test_labels, labelset)
    test_loss, test_acc = evaluate_validation_set(model, test_seqs, test_golds, test_lengths, test_sentences, loss_function)
    logging.info('Epoch {}: Test loss {:.4f}, test acc {:.4f}'.format(epoch, test_loss, test_acc))

    """
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Tweet classification using SVM')
    parser.add_argument('--data', type=str, default='/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/mh17_60_20_20.json',
                            help="File with main data")
    parser.add_argument('--additional_data', type=str,
                        default='combi',
                        choices=['', 'combi', 'sbil', 'skaschi'],
                        help="Additional train data. if empty string, no additional data is used")
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
    parser.add_argument('--num_feature_maps', type=int, default=10,
                        help="Number of CNN feature maps")
    parser.add_argument('--ks', type=int, default=3,
                        help="kernel size")
    parser.add_argument('--upsample', type=bool_flag, default=True,
                        help="if enabled upsample the train data according to size of largest class")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Keep probability for dropout")
    parser.add_argument('--embs', type=bool, default=True,
                        help="Use pre-trained embeddings")
    parser.add_argument('--max_vocab', type=int, default=-1,
                        help="Maximum number of words read in from the pretrained embeddings. -1 to disable")
    parser.add_argument('--hyperparam_csv', type=str,
                        default='../hyperparams_svm.csv',
                        help="File with hyperparams. If set, values for specified hyperparams are read from the csv")
    parser.add_argument('--result_csv', type=str,
                        default='../results_svm_pc.csv',
                        help="File the results and hyperparams are written to")
    parser.add_argument('--rowid', type=int,
                        default=2,
                        help="Row from which hyperparams are read")
    args = parser.parse_args()
    main(args)