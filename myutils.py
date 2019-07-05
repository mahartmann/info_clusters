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


def setup_logging(exp_path, logfile='log.txt'):
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