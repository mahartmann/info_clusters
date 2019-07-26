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

def dump_ner_output_to_file(fname, output, batch):
    # info in batch is: (tokenized_tweet, cleaned_tokenized_tweet, tweet_text, tweet_id)
    with open(fname, 'a', encoding='utf-8') as fout:
        # generate a json dict for each tweet
        for ner_text, ner_tags, cleaned_tokenized, orig_tweet, tid  in zip(output[0], output[1], [elm[1] for elm in batch],  [elm[2] for elm in batch],  [elm[3] for elm in batch]):
            d = {'tid': tid, 'ner:': {'text': ner_text, 'tags': ner_tags}, 'text': orig_tweet, 'cleaned_text': cleaned_tokenized}
            fout.write(json.dumps(d))
            fout.write('\n')
    fout.close()



if __name__=="__main__":

    infile = sys.argv[1]
    outfile = sys.argv[2]
    lang=sys.argv[3]

    outfile_tmp = outfile + '.tmp'

    setup_logging('.', 'log.txt')
    minibatch_size = 100

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

        batch = []
        for row in f:
            if len(row.split('\t')) < 2: continue
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

                    tokenized_tweet = ' '.join(tweet)
                    tokenized_tweet_sent_sep = '#####'.join(tweet)

                    #strip non alphanumerics, lowercase
                    cleaned_tokenized_tweet = ' '.join(
                        [tok.lower() for tok in tokenized_tweet_sent_sep.split(' ') if isalpha_or_hash(tok)])

                    if len(cleaned_tokenized_tweet) > 0:
                        batch.append((tokenized_tweet_sent_sep, cleaned_tokenized_tweet, tweet_text, tweet_id))
                    # ner
                    if len(batch) >= minibatch_size:


                        output = ner_model([elm[0] for elm in batch])
                        dump_ner_output_to_file(outfile, output, batch)
                        batch = []


                    c += 1
                    if c % 1000 == 0:
                        logging.info('---> Processed {}'.format(c))



        dump_ner_output_to_file(outfile, output, batch)
    f.close()
