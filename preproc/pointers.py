import json
from time import time
import sys

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
            print(row)
            if not row:
                break
            counter += 1
            row = row.strip()
            token = row.split(',')[0]
            token = token.strip('"')
            key = token
            tokens[key] = pos
            if counter % 10 ** 6 == 0:
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


def get_embedding(fp, n):
    start_load = time()
    fp.seek(n)
    emb = fp.readline()
    tid = emb.split(',')[0].strip('"')
    print('Accessed {} at byte {} in {:.2f} seconds.'.format(tid, n, time() - start_load))
    print(emb)
    return emb


if __name__ == '__main__':
    embeddings_path = sys.argv[1]
    pointers_path = sys.argv[2]

    tokens = make_pointers(embeddings_path)
    save_pointers(tokens, pointers_path)

    """
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

