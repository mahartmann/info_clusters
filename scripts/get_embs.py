import sys


def read_tids(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip('\n"') for line in lines]

def write_lines(fname, lines):
    with open(fname,'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)
    f.close()

if __name__=="__main__":
    emb_file = sys.argv[1]
    id_file = sys.argv[2]
    out_file = sys.argv[3]

    ids = set(read_tids(id_file))
    outlines = []
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            splt = line.split('\t')[0]
            if splt in ids:
                outlines.append(line)
    f.close()
    write_lines(out_file, outlines)