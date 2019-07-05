import json
import codecs
import csv
import itertools
import logging
import sys
from myutils import load_json, save_json, setup_logging

def timestamp2tweet_mapping(fname, outname):
    """
    generate a mapping from a timestamp (str) to a list of tids
    :param fname:
    :return:
    """
    ti2tw = dict()
    c = 0

    with codecs.open(fname, 'r', 'utf-8') as csvfile:
        header_reader = csv.reader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"')
        for elm in header_reader:
            header = elm
        reader = csv.DictReader(itertools.islice(csvfile, 1, None), delimiter=',', quotechar='"', fieldnames=header)
        for row in reader:

            ts = row['tweet_time']
            tid = row['tweetid']
            ti2tw.setdefault(ts, []).append(tid)
            c += 1
            if c %1000 == 0:
                logging.info('Processed {}'.format())
    csvfile.close()
    save_json(outname, ti2tw)


if __name__=="__main__":

    fname = sys.argv[1]
    outname = sys.argv[2]
    setup_logging()
    #fname = '/home/mareike/PycharmProjects/sheffield/data/ira_tweets_csv_hashed.csv.1000'
    #outname = '/home/mareike/PycharmProjects/sheffield/data/time2tweet.json'

    timestamp2tweet_mapping(fname, outname)

    d = load_json(outname)
    print(len(d.keys()))
