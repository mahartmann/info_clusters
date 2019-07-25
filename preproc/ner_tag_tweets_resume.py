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
from info_clusters.preproc import pointers
from info_clusters.preproc.pointers import load_pointers

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
        for orig, tags, tid in zip(output[0], output[1], batch_ids):
            d = {'tid': tid, 'text': orig, 'tags': tags}
            fout.write(json.dumps(d))
            fout.write('\n')
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
    logging.info('Start logging...')

    repls = {urlPattern: 'URL ', retweetCommandPattern: 'RETW ', atMentionPattern: 'ATMENTION '}

    outfile = '{}.{}.{}_{}.add'.format(infile , lang, 'NER', batch)
    tfile = '{}.{}.{}_{}'.format(infile, lang, 'NER', batch)
    tfile_next = '{}.{}.{}_{}'.format(infile, lang, 'NER', batch+1)

    # get the last processed id from the .tokenized documents
    with codecs.open(tfile, 'r', 'utf-8') as f:
        lines = f.readlines()
    firstId = json.loads(lines[-1].strip('\n'))['tid']

    # get the first processed id from the next batch
    with codecs.open(tfile_next, 'r', 'utf-8') as f:
        lines = f.readlines()
    firstId_next_batch = json.loads(lines[0].strip('\n'))['tid']

    #start = bs*batch
    end = bs*(batch+1)



    logging.info('Loading model...')
    ner_model = build_model(configs.ner.ner_rus_bert, download=True)

    dl = pointers.DataLoader()
    pointers = load_pointers(dl.config.get('Files', 'tweets_pointers'))


    c = 0
    with codecs.open(infile, 'r', 'utf-8') as csvfile:
        header_reader = csv.reader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"')
        for elm in header_reader:
            header = elm

        # jump to position of last processed line
        csvfile.seek(pointers[firstId])
        start = 1
        end = bs


        reader = csv.DictReader(itertools.islice(csvfile, start, end), delimiter=',', quotechar='"', fieldnames=header)

        tweet_batch = []
        batch_ids = []
        for row in reader:
            c+= 1
            if c%1000==0:
                logging.info('Processed {} lines'.format(c))
            if row['tweet_language'] == lang:
                t = clean_text(repls, row['tweet_text'])

                #break if it's the first id of the next batch
                assert type(row['tweetid']) == type(firstId_next_batch)
                if row['tweetid'] == firstId_next_batch:
                    break
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