import csv
import codecs
import stanfordnlp
import re
import sys
import itertools


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


if __name__=="__main__":

    infile = sys.argv[1]
    lang=sys.argv[2]
    batch = int(sys.argv[3])

    bs = 1000000

    outfile = '{}.{}.{}_{}'.format(infile , lang, 'tokenized', batch)
    start = bs*batch
    end = bs*(batch+1)

    nlp = stanfordnlp.Pipeline(lang=lang)
    repls = {urlPattern: 'URL ', retweetCommandPattern: 'RETW ', atMentionPattern: 'ATMENTION '}



    c = 0
    with codecs.open(infile, 'r', 'utf-8') as csvfile:
        header_reader = csv.reader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"')
        for elm in header_reader:
            header = elm
        reader = csv.DictReader(itertools.islice(csvfile, start, end), delimiter=',', quotechar='"', fieldnames=header)
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
                        print('---> Processed {}'.format(c))
        fout.close()
    csvfile.close()