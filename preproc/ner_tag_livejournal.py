import csv
import codecs
import stanfordnlp
import re
import sys
import itertools
from deeppavlov import configs, build_model
import json
import os
import logging
from info_clusters.myutils import load_json, save_json
import os

# filter out urls
# filter out @mentions
# filter out RT syntax

urlPattern = re.compile('(http.*?//[^ ]+) ?')
retweetCommandPattern = re.compile('RT @([^ ]+?): ')
atMentionPattern = re.compile('@([^ ]+?) ')

def clean_text(replacements, s):
    """
    replacements is a mapping from regex pattern to replacement
    """
    for pattern in replacements.keys():
        m = re.search(pattern, s)
        if m:
            for elm in re.finditer(pattern, s):
                s = s.replace(elm.group(), replacements[pattern])
    return s

def isalpha_or_hash(s):
    """
    returns True the word is alphanumeric in the sense of pythons isalpha
    or contains a hashtag
    """
    s = s.replace('-', '')
    if s.isalpha():
        return True
    if s.startswith('#'):
        return True
    else:
        return False


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


if __name__=="__main__":

    indir = sys.argv[1]

    minibatch_size = 500

    setup_logging(logfile='log-livejournal_ner.txt')

    repls = {urlPattern: 'URL ', retweetCommandPattern: 'RETW ', atMentionPattern: 'ATMENTION '}

    logging.info('Loading model...')
    ner_model = build_model(configs.ner.ner_rus_bert, download=False)


    c = 0

    #iterate through the livejournal texts

    fnames = sorted([f for f in os.listdir(indir) if f.endswith('.text.json')])
    for fname in fnames:
        logging.info('Processing {}'.format(fname))
        d = load_json(os.path.join(indir, fname))
        ner = {'text':[], 'tags': []}

        tweet_batch = []
        batch_ids = []
        for elm in d['text']:
            t = clean_text(repls, elm)
            if len(t) > 0:
                tweet_batch.append(t)

            if len(tweet_batch) >= minibatch_size:
                # run ner
                logging.info('Tagging...')
                output = ner_model(tweet_batch)
                # dump to file
                ner.setdefault('text', []).extend(output[0])
                ner.setdefault('tags', []).extend(output[1])
                # reset the list
                tweet_batch = []

        if len(tweet_batch) > 0:
            output = ner_model(tweet_batch)
            ner.setdefault('text', []).extend(output[0])
            ner.setdefault('tags', []).extend(output[1])


        d['ner'] = ner
        outname = os.path.join(indir, '{}.ner.json'.format(fname.strip('.text.json')))
        logging.info('Dumping output')
        save_json(outname, d)