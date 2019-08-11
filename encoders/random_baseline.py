import itertools
from collections import Counter, defaultdict
import numpy as np
from info_clusters.encoders.model_utils import load_json, save_json, p_r_f, print_result_summary


if __name__=="__main__":

    np.random.seed(42)

    d = load_json('/home/mareike/PycharmProjects/catPics/data/twitter/mh17/experiments/mh17_60_20_20.json')

    train_labels = d['train']['label']
    labelset = list(set(train_labels))

    dev_labels = d['dev']['label']

    preds = []
    for elm in dev_labels:
        preds.append(np.random.choice(labelset))

    label2idx = {label: labelset.index(label) for label in labelset}
    results = p_r_f([label2idx[label] for label in dev_labels], [label2idx[label] for label in preds], labelset)
    print(print_result_summary(results))

