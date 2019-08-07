import json
import codecs
import os
import logging


def load_json(fname):
    with codecs.open(fname, 'r', 'utf-8') as f:
        j = json.load(f)
    return j

def save_json(fname, data):
    with codecs.open(fname, 'w', 'utf-8') as f:
        json.dump(data, f)

def read_file(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip('\n') for line in lines]

def write_file(fname, data):
    with open(fname, 'w', encoding='utf-8') as f:
        for line in data:
            f.write('{}\n'.format(line))
    f.close()


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

def per_class_p_r(cm):
    for i in range(cm.shape[0]):
        p = float(cm[i,i])/(sum(cm[:,i]))
        r = float(cm[i, i]) / (sum(cm[i, :]))
        print('%d Prec: %.2f Rec:%.2f'%(i+1, p*100, r*100))