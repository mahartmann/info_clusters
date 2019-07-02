import csv
import codecs
import stanfordnlp
import re
import sys
import logging


# filter out urls
# filter out @mentions
# filter out RT syntax

urlPattern = re.compile('(http.*?//[^ ]+) ?')
retweetCommandPattern = re.compile('RT @([^ ]+?): ')
shortenedTextString = '\xe2\x80\xa6'

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
    if s.isalpha():
        return True
    if s.startswith('#'):
        return True
    else:
        return False


if __name__=="__main__":

    infile = sys.argv[0]
    outfile=sys.argv[1]
    lang=sys.argv[2]

    nlp = stanfordnlp.Pipeline(lang=lang)
    repls = {urlPattern: 'URL ', retweetCommandPattern: 'RETW '}


    c = 0
    with codecs.open(infile, 'r', 'utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        with codecs.open(outfile, 'w', 'utf-8') as fout:
            for row in reader:
                if row['tweet_language'] == lang:
                    t = clean_text(repls, row['tweet_text'])
                    doc = nlp(t.lower())
                    tweet = []
                    for sentence in doc.sentences:
                        s = ' '.join([word.text for word in sentence.words if isalpha_or_hash(word.text)])
                        tweet.append(s)
                    fout.write('{}\t{}\n'.format(row['tweetid'], ' ##### '.join(tweet)))
                    c += 1
                    if c % 100 == 0:
                        logging.info('---> Processed {}'.format(c))
        fout.close()
    csvfile.close()