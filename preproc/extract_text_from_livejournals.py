"""
extract russian text from livejournal html
"""
import codecs
import os
import sys
from info_clusters.myutils import setup_logging, save_json
import logging
import re

title_pattern = re.compile('<title>(.*)</title>')
text_pattern = re.compile('[\u0400-\u04FF]*')


def read_file(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip(' \n') for line in lines]


def contains_kyrillic(s):
    return bool(re.search('[\u0400-\u04FF]', s))

def is_valid(line):
    sep = '…'
    if not line.startswith('<'):
        return False
    elif sep in line:
        return False
    return True

def extract_kyrillic(s):
    # match every span of text that contains only kyrillic letters or special signs
    s = s.replace('  ', ' ')
    m = re.findall('([\u0400-\u04FF]+([\u0400-\u04FF]|[- !,\.\?\'":]|[0-9])+)', s)
    matches = []
    for m1 in m:
        matches.append(m1[0])
    return matches

def get_title(s):
    m = re.search(title_pattern, s)
    if m:
        return m.group(1)


if __name__=="__main__":
    user2elms = dict()
    entry2elms = dict()
    path = sys.argv[1]
    setup_logging(logfile='text_from_livejournal.log')

    fnames = sorted([f for f in os.listdir(path) if not f.endswith('.zip')])
    for c, fname in enumerate(fnames):
        logging.info('\n\n############ {} FNAME {}'.format(c, fname))
        lines = read_file(os.path.join(path, fname))
        texts = dict()
        d = {}
        d = {'filename': fname, 'domain': fname.split('_')[0]}
        text = []
        for line in lines:
            if contains_kyrillic(line):
                # extract the title
                if line.startswith('<title>'):
                    title = get_title(line)
                    d['title'] = title
                elif is_valid(line):
                    # print(line)
                    for elm in extract_kyrillic(line):
                        if len(elm.split()) > 1:
                            # if elm not in texts:
                            text.append(elm)
                            user2elms.setdefault(fname.split('_')[0], []).append(elm)
                            entry2elms.setdefault(fname, set()).add(elm)
        d['text'] = text
        save_json(os.path.join(path, '{}.text.json'.format(fname)), d)


