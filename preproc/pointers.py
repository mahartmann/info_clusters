import json
from time import time
import sys
import configparser
import csv
import json

"""
Rahul's code for generating pointers and using them to access data
"""


def make_pointers(path):
    with open(path, 'r', encoding='utf-8') as f:
        tokens = {}
        row = f.readline()  # ignore the first line
        counter = 0
        while True:
            pos = f.tell()
            row = f.readline()
            if not row:
                break
            counter += 1
            row = row.strip()
            token = row.split(',')[0]
            token = token.strip('"')
            key = token
            tokens[key] = pos
            if counter % 10 ** 3 == 0:
                print('Processed {} vectors...'.format(counter))
    return tokens

def make_pointers_mh17(path):
    with open(path, 'r', encoding='utf-8') as f:
        tokens = {}
        counter = 0
        while True:
            pos = f.tell()
            row = f.readline()
            if not row:
                break
            counter += 1
            row = json.loads(row).strip()
            token = row['tweetid']
            token = token.strip()
            key = token
            tokens[key] = pos
            if counter % 10 ** 3 == 0:
                print('Processed {} vectors...'.format(counter))
    return tokens


def save_pointers(tokens, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tokens, f)


def load_pointers(path):
    print('Loading TID to line mapping...')
    start_load = time()
    with open(path, 'r', encoding='utf-8') as f:
        pointers = json.load(f)
    print('Done. Took {0:.2f} seconds.'.format(time() - start_load))
    return pointers


def file_open(path):
    return open(path, 'r', encoding='utf-8')


def file_close(fp):
    fp.close()


def get_data(fp, n):
    start_load = time()
    fp.seek(n)
    emb = fp.readline()
    tid = emb.split(',')[0].strip('"')
    #print('Accessed {} at byte {} in {:.2f} seconds.'.format(tid, n, time() - start_load))
    return emb


class DataLoader(object):

    """
    loads data via pointers from file specified in the config
    """

    def __init__(self):
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read('../config.cfg')

        self.data_file, self.pointers = self._load_tweets()
        self.header = ["tweetid", "userid", "user_display_name", "user_screen_name", "user_reported_location",
                       "user_profile_description", "user_profile_url", "follower_count", "following_count",
                       "account_creation_date", "account_language", "tweet_language", "tweet_text", "tweet_time",
                       "tweet_client_name", "in_reply_to_tweetid", "in_reply_to_userid", "quoted_tweet_tweetid",
                       "is_retweet", "retweet_userid", "retweet_tweetid", "latitude", "longitude", "quote_count",
                       "reply_count", "like_count", "retweet_count", "hashtags", "urls", "user_mentions",
                       "poll_choices"]

    def load_tweets(self, tids):
        data = []
        for tid in tids:
            if tid not in self.pointers:
                print('{} is not in the pointers dict. Skipping'.format(tid))
            emb = get_data(self.data_file, self.pointers[tid])
            data.append(emb)
        reader = csv.DictReader(data, delimiter=',', quotechar='"', fieldnames=self.header)
        return [row for row in reader]

    def _load_tweets(self):
        data_file = file_open(self.config.get('Files', 'tweets'))
        pointers = load_pointers(self.config.get('Files', 'tweets_pointers'))
        return data_file, pointers




    def get_row(self, tif):
        # return the row number that the tweet is located in
        pass


if __name__ == '__main__':
    embeddings_path = sys.argv[1]
    pointers_path = sys.argv[2]
    #embeddings_path = '/home/mareike/PycharmProjects/sheffield/data/ira_tweets_csv_hashed.csv'
    #pointers_path = '/home/mareike/PycharmProjects/sheffield/data/ira_tweets_csv_hashed.pointers.json'
    tokens = make_pointers_mh17(embeddings_path)
    save_pointers(tokens, pointers_path)
    """
    tids = ['492388766930444288',
            '710845825622999040',
            '498874533361627136'
            ]

    dl = DataLoader()
    ts = dl.load_tweets(tids)
    for t in ts:
        print(t)


    emb_file = file_open(embeddings_path)
    lines = load_pointers(pointers_path)
    tids = ['492388766930444288',
            '710845825622999040',
            '498874533361627136'
            ]

    for tid in tids:
        emb = get_embedding(emb_file, lines[tid])
        print(emb)
    """

