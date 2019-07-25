import sys
import os
from info_clusters.myutils import setup_logging, save_json
from info_clusters.preproc.pointers import DataLoader
import logging


if __name__=="__main__":
    #tokenized_file = '/home/mareike/PycharmProjects/sheffield/data/ira_tweets_csv_hashed.csv.ru.tokenized.100000'
    tokenized_file = sys.argv[1]
    outfile = sys.argv[2]

    setup_logging(exp_path='.', logfile='log.txt')
    dl = DataLoader()

    with open(tokenized_file, 'r', encoding='utf-8') as f:
        data = dict()
        for line in f:
            line = line.strip()
            splt = line.split('\t')
            if len(splt) > 1:
                tid = splt[0]
                text = splt[1]
                seen_text = set()
                if dl.load_tweets([tid])[0]['is_retweet'] != 'True':
                    hashtags = set()
                    cleaned_text = []
                    for tok in text.split(' '):
                        if tok.startswith('#') and tok != '#####':
                            hashtags.add(tok)
                        else:
                            cleaned_text.append(tok)
                    if len(cleaned_text) > 0 and len(hashtags) > 0:
                        cleaned_text = ' '.join(cleaned_text)

                        if cleaned_text not in seen_text:
                            seen_text.add(cleaned_text)
                            data.setdefault('text', []).append(cleaned_text)
                            data.setdefault('label', []).append(list(hashtags))
                            data.setdefault('tids', []).append(tid)
                        else:
                            logging.info('### {} {} already in data'.format(cleaned_text, hashtags))

    save_json(outfile, data)


