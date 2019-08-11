import json

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
import os
import logging
from collections import Counter
import numpy as np
import subprocess
import random
import argparse

#from info_clusters.encoders.lstm import UNK
from info_clusters.myutils import read_file, write_file

UNK='UNK'

class PaddedTensorDataset(Dataset):
    """Dataset wrapping data, target and length tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        length (Tensor): contains sample lengths.
        raw_data (Any): The data that has been transformed into tensor, useful for debugging
    """

    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data, tids):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data
        self.tids = tids

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.length_tensor[index], self.raw_data[index], self.tids[index]

    def __len__(self):
        return self.data_tensor.size(0)

def setup_logging(exp_path='.', logfile='log.txt'):
    # create a logger and set parameters
    logfile = os.path.join(exp_path, logfile)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

def print_class_distributions(labels):
    c = Counter(labels)
    s = ''
    for key, val in c.most_common():
        s+= '#seqs with label {}: {} ({:.2f}%)\n'.format(key, val, float(val)/len(labels)*100)
    return s

def upsample(seqs, labels):

    # upsample the data, such that the number of instances for each label is appr. the same
    c = Counter(labels)

    target_num = c.most_common(1)[0][1]
    labelset = c.keys()
    upsampled_idxs = []
    for target_label in labelset:
        print(target_num)
        print(c[target_label])
        idxs = [i for i in range(len(seqs)) if labels[i] == target_label]
        upsampled_idxs += list(np.random.choice(idxs, target_num-c[target_label]))

    seqs += [seqs[i] for i in upsampled_idxs]
    labels += [labels[i] for i in upsampled_idxs]
    return seqs, labels


def save_json(fname, data):
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_json(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        j = json.load(f)
    return j




def prepare_labels(labels, labelset):
    if labelset is None:
        labelset = list(set(labels))
    labelset = sorted(labelset)
    prepared_labels = []
    for label in labels:
        prepared_labels.append(labelset.index(label))
    return torch.LongTensor(prepared_labels), labelset


def sort_batch(batch, targets, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]
    return seq_tensor, target_tensor, seq_lengths


def get_tokens(line, lower):
    if lower:
        return [elm.strip().lower() for elm in line.strip().split()]
    else:
        return [elm.strip() for elm in line.strip().split()]

def get_chars(line, lower):
    if lower:
        line = line.lower()
    return [c for c in line]

def compute_dim_feature_map(len_in, kernel_size, stride, padding, dilation):
    """
    computes the kernel size for the pytorch pooling layer, such that the output has dimensionality out_dim
    :return:
    """
    out = ((len_in + 2*padding - dilation*(kernel_size - 1) -1)/float(stride)) + 1
    return np.ceil(out)


def compute_kernel_size(len_in, len_out, stride, padding, dilation):
    out = (len_in + 2*padding - 1 -(len_out-1)*stride + dilation)/dilation
    return np.ceil(out)



def pad_sequences(vectorized_seqs, seq_lengths):
    # create a zero matrix
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


def seqs2minibatches(seqs, golds, lengths, raw_sents, batch_size, tids=[]):
    padded_seqs = pad_sequences(seqs, lengths)
    if len(tids) == 0:
        tids = [0 for elm in raw_sents]
    return DataLoader(PaddedTensorDataset(padded_seqs, golds, lengths, raw_sents, tids), batch_size=batch_size)


def sents2seqs(sents, feat2idx, lower=True):
    """
        map sequences of words to sequences of indexes, pad the sequences and sort the minibatches according to lentgh
        :param sents:
        :param feat2idx:
        :param lower:
        :return:
        """
    if feat2idx is None:
        vocab = Counter()
        logging.info('Building vocabulary')
        for sent in sents:
            vocab.update(Counter(get_tokens(sent, lower=lower)))
        if '' in vocab.keys():
            del vocab['']
        feat2idx = dict()
        words = sorted(list(vocab.keys()))
        for word in words:
            feat2idx[word] = len(feat2idx)
    # add the unk token
    if UNK not in feat2idx:
        feat2idx[UNK] = len(feat2idx)
    logging.info('Mapping sentences to sequences')
    seqs = []
    lengths = []
    for sent in sents:
        if len(get_tokens(sent, lower=lower)) == 0:
            seq = [feat2idx[UNK]]
        else:
            seq = [feat2idx[tok] if tok in feat2idx else feat2idx[UNK] for tok in get_tokens(sent, lower=lower)]
        seqs.append(seq)
        lengths.append(len(seq))
    lengths = torch.LongTensor(lengths)
    return seqs, lengths, feat2idx

def sents2charseqs(sents, feat2idx, lower=True):
    """
    map sequences of chracters to sequences of character indexes, pad the sequences and sort the minibatches according to lentgh
    :param sents:
    :param word2idx:
    :param lower:
    :return:
    """
    if feat2idx is None:
        vocab = Counter()
        logging.info('Building char vocabulary')
        for sent in sents:
            vocab.update(Counter(get_chars(sent, lower=lower)))
        if '' in vocab.keys():
            del vocab['']
        feat2idx = dict()
        chars = sorted(list(vocab.keys()))
        for char in chars:
            feat2idx[char] = len(feat2idx)
    if not UNK in feat2idx:
        feat2idx[UNK] = len(feat2idx)
    logging.info('Mapping sentences to char sequences')
    seqs = []
    lengths = []
    for sent in sents:
        if len(get_chars(sent, lower=lower)) == 0:
            seq = [feat2idx[UNK]]
        else:
            seq = [feat2idx[tok] if tok in feat2idx else feat2idx[UNK] for tok in get_chars(sent, lower=lower)]
        seqs.append(seq)
        lengths.append(len(seq))
    lengths = torch.LongTensor(lengths)
    return seqs, lengths, feat2idx


def per_class_p_r(cm):
    for i in range(cm.shape[0]):
        p = float(cm[i,i])/(sum(cm[:,i]))
        r = float(cm[i, i]) / (sum(cm[i, :]))
        print('%d Prec: %.2f Rec:%.2f'%(i+1, p*100, r*100))


def load_embeddings_from_file(fname, max_vocab=-1):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    with open(fname, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) == 2:
                continue
            else:
                word, vect = line.rstrip().split(' ', 1)

                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                assert word not in word2id
                word2id[word] = len(word2id)
                vectors.append(vect[None])
            if max_vocab > 0 and i >= max_vocab:
                break
    # add a zero vector for the UNK token
    dim = vectors[-1].shape[1]
    if not UNK in word2id:
        word2id[UNK] = len(word2id)
        vectors.append(np.array([0 for i in range(dim)])[None])
    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    return embeddings, word2id, id2word

def prefix_sequence(seq, prefix):
    return ' '.join(['{}:{}'.format(prefix, elm) for elm in seq.split()])

def deprefix_sequence(seq):
    return ' '.join([elm.split(':')[-1] for elm in seq.split()])


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")

def get_exp_path(exp_path, exp_name, exp_id):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = exp_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir {}".format(exp_folder), shell=True).wait()
    assert exp_name != ''
    exp_folder = os.path.join(exp_folder, exp_name)
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir {}".format(exp_folder), shell=True).wait()
    if exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, exp_id)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, exp_id)
        assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir {}".format(exp_path), shell=True).wait()
    return exp_path

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


def log_params(args):
    for key in sorted(args.keys()):
        logging.info('{}: {}'.format(key, args[key]))


if __name__=="__main__":
    fname = '/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/resources/mh17_60_20_20_vocab_extended_embs.txt'
    prefix = 'en'

    lines = read_file(fname)
    outlines = ['{}:{}'.format(prefix, line) for line in lines]
    write_file(fname + 'prefixed', outlines)


