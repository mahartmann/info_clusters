import itertools
from collections import Counter, defaultdict
import numpy as np
from info_clusters.encoders.model_utils import load_json, save_json, p_r_f, print_result_summary
"""
this baseline computes a hashtag2class dictionary using pmi scores of the training data
"""

def compute_pmis(seqs, labels, labelset):
    token_counts = Counter(itertools.chain.from_iterable([seq.split() for seq in seqs]))
    label_counts = Counter(labels)
    token_sum = np.sum(list(token_counts.values()))
    print(token_sum)
    label_sum = np.sum(list(label_counts.values()))
    coocs = {}
    for seq, label in zip(seqs, labels):
        for tok in seq.split():
            coocs.setdefault(tok, defaultdict(int))[label] += 1
    pmis = {}
    for tok, tok_count in token_counts.items():
        for l, label in enumerate(labelset):
            p_x_y = coocs[tok][label]/float(label_counts[label])
            if p_x_y == 0:
                pmi = 0
            else:
                p_x = tok_count/float(token_sum)
                pmi = np.log(p_x_y/p_x)
            pmis.setdefault(tok, []).append(pmi)
    return pmis, coocs


def sents2seqs(sents):
    train_seqs = []
    for sent in sents:
        train_seqs.append(' '.join([elm for elm in sent.split() if elm.startswith('#') and elm != '#mh17']))
    return train_seqs


def majority(labels):
    c = Counter(labels)
    if len(set(c.values())) == 1:
        return np.random.choice(labels)
    else:
        return c.most_common(1)[0][0]

def predict(seqs, pmis, coocs, thr=3):
    oovs = []
    multis = []
    predictions = []
    for seq in seqs:
        seq_predictions = []
        for elm in seq.split():
            # check that the hashtag occurs more than threshold times

            if elm in pmis and sum(list(coocs[elm].values())) > thr:
                print(sum(list(coocs[elm].values())))
                pred = labelset[np.argmax(pmis[elm])]
                seq_predictions.append(pred)
        if len(seq_predictions) == 0:
            oovs.extend(seq.split())
            predictions.append(np.random.choice(labelset))
        elif len(seq_predictions) > 1:
            # check if we can make a majority decision
            majority_label = majority(seq_predictions)
            predictions.append(majority_label)
            if len(set(seq_predictions)) > 1:
                multis.append(len(set(seq_predictions)))
        else:
            predictions.append(seq_predictions[0])

    return predictions, oovs, multis

if __name__=="__main__":

    np.random.seed(42)

    d = load_json('/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/mh17_60_20_20.json')
    train_seqs = sents2seqs(d['train']['seq'])
    train_labels = d['train']['label']


    labelset = list(set(train_labels))

    pmis, coocs = compute_pmis(train_seqs, train_labels, labelset)

    sorted_keys = sorted(list(pmis.keys()), key=lambda x: np.max(pmis[x]), reverse=True)

    for tok in sorted_keys:
        vals = pmis[tok]
        print(tok, vals, [coocs[tok][label] for label in labelset])

    dev_seqs = sents2seqs(d['dev']['seq'])
    dev_labels = d['dev']['label']

    # predict

    dev_preds, oovs, multis = predict(dev_seqs, pmis, coocs, thr=10)
    print(dev_labels)
    print(dev_preds)
    label2idx = {label: labelset.index(label) for label in labelset}
    results = p_r_f([label2idx[label] for label in dev_labels], [label2idx[label] for label in dev_preds], labelset)
    print(print_result_summary(results))

    print('\nOOVs" {} {}'.format(len(oovs), 100*len(oovs)/float(len(dev_seqs))))
    for key, val in Counter(oovs).most_common():
        print(key, val)
    print('\nMultis: {} {}'.format(len(multis),  100*len(multis)/float(len(dev_seqs))))
    for key, val in Counter(multis).most_common():
        print(key, val)