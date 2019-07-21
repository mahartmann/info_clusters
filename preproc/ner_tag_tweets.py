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


def dump_ner_output_to_file(fname, output, batch_ids):
    with open(fname, 'a', encoding='utf-8') as fout:
        # generate a json dict for each
        for out, tid in zip(output, batch_ids):
            orig = out[0]
            tags = out[1]
            d = {'tid': tid, 'text': orig, 'tags': tags}
            json.dump(d, fout)
    fout.close()

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

    infile = sys.argv[1]
    lang=sys.argv[2]
    batch = int(sys.argv[3])

    bs = 1000000
    minibatch_size = 100

    setup_logging(logfile='log{}.txt'.format(batch))
    outfile = '{}.{}.{}_{}'.format(infile , lang, 'NER', batch)
    start = bs*batch
    end = bs*(batch+1)

    repls = {urlPattern: 'URL ', retweetCommandPattern: 'RETW ', atMentionPattern: 'ATMENTION '}


    ner_model = build_model(configs.ner.ner_rus_bert, download=True)


    c = 0
    with codecs.open(infile, 'r', 'utf-8') as csvfile:
        header_reader = csv.reader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"')
        for elm in header_reader:
            header = elm
        reader = csv.DictReader(itertools.islice(csvfile, start, end), delimiter=',', quotechar='"', fieldnames=header)

        tweet_batch = []
        batch_ids = []
        for row in reader:
            c+= 1
            if c%10000==0:
                logging.info('Processed {} lines'.format(c))
            if row['tweet_language'] == lang:
                t = clean_text(repls, row['tweet_text'])

                batch_ids.append(row['tweetid'])
                tweet_batch.append(t)

                assert len(batch_ids) == len(tweet_batch)

                if len(tweet_batch) >= minibatch_size:
                    # run ner
                    logging.info('Tagging...')
                    output = ner_model(tweet_batch)
                    # dump to file
                    logging.info('Dumping to file...')
                    dump_ner_output_to_file(outfile, output, batch_ids)
                    # reset the list
                    tweet_batch = []
                    batch_ids = []
        output = ner_model(tweet_batch)
        # dump to file
        dump_ner_output_to_file(outfile, output, batch_ids)
        # reset the list
        tweet_batch = []
        batch_ids = []

    csvfile.close()