"""
tokenize and ne tag the mh17 tweets
"""
import csv
import codecs
import stanfordnlp
import re
import sys
import itertools
from deeppavlov import configs, build_model
import logging
from info_clusters.myutils import setup_logging, save_json
import json


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

def lowercase_hashtags(s):
    """
    lowercase all hashtags to prevent them from being tokenized by stanfordnlp
    """
    return ' '.join([elm.lower() if elm.startswith('#') else elm for elm in s.split(' ')])

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


if __name__=="__main__":

    infile = sys.argv[1]
    outfile = sys.argv[2]
    lang=sys.argv[3]

    outfile_tmp = outfile + '.tmp'

    setup_logging('.', 'log.txt')

    # setup the models
    nlp = stanfordnlp.Pipeline(lang=lang)
    if lang == 'en':
        ner_model =  build_model(configs.ner.ner_ontonotes_bert, download=False)

    elif lang == 'ru':
        ner_model = build_model(configs.ner.ner_rus_bert, download=False)



    repls = {urlPattern: 'URL ', retweetCommandPattern: 'RETW ', atMentionPattern: 'ATMENTION '}

    data = {}

    c = 0
    with codecs.open(infile, 'r', 'utf-8') as f:
        with codecs.open(outfile_tmp, 'a', 'utf-8') as fout:
            for row in f:
                tweet_lang = row.split('\t')[-2]
                if tweet_lang == lang:
                        tweet_data = {}
                        tweet_id = row.split('\t')[0]
                        tweet_text = row.split('\t')[9]
                        # replace twitter lingo
                        t = clean_text(repls, tweet_text)
                        #lower case hashtags
                        t = lowercase_hashtags(t)
                        #tokenize
                        doc = nlp(t)
                        tweet = []
                        for sentence in doc.sentences:
                            s = ' '.join([word.text for word in sentence.words])
                            tweet.append(s)
                        tokenized_tweet = ' ##### '.join(tweet)

                        # ner
                        output = ner_model([tokenized_tweet])

                        # clean non alphanumerics
                        cleaned_tokenized_tweet = ' '.join([tok.lower() for tok in tokenized_tweet.split(' ') if isalpha_or_hash(tok)])

                        tweet_data['tid'] = tweet_id
                        tweet_data['text'] = tweet_text
                        tweet_data['cleaned_text'] = cleaned_tokenized_tweet
                        tweet_data['ner'] = {'text': output[0][0], 'tags': output[1][0]}

                        data[tweet_id] = tweet_data
                        c += 1
                        if c % 100 == 0:
                            logging.info('---> Processed {}'.format(c))
                        if c > 1000:break

                        fout.write(json.dumps(tweet_data))
                        fout.write('\n')

    f.close()
    save_json(outfile, data)