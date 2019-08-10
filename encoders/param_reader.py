import csv
import itertools
import argparse


def read_hyperparams_from_csv(fname, rowid):
    type_conversion = {'int': int, 'float': float, 'str': str, 'bool': bool_flag}
    with open(fname, 'r', encoding='utf-8') as csvfile:
        header_reader = csv.reader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"')
        header = [elm for elm in header_reader][0]
        type_reader = csv.DictReader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"', fieldnames=header)
        types = [elm for elm in type_reader][0]
    csvfile.close()
    with open(fname, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(itertools.islice(csvfile, rowid, rowid+1), delimiter=',', quotechar='"', fieldnames=header)
        params = [elm for elm in reader][0]
    # type the arguments
    for key, val in params.items():
        params[key] = type_conversion[types[key]](val)
    return params


def write_results_and_hyperparams(fname, results, params):

    metrics = ['p', 'r', 'f']
    results_prefixed = {}
    for key in ['macro', 'micro']:
        for i, m in enumerate(metrics):
            results_prefixed['{}_{}'.format(key, metrics[i])] = results[key][i]

    for key in results['per_class'].keys():
        for i, m in enumerate(metrics):
            results_prefixed['{}_{}'.format(key, metrics[i])] = results['per_class'][key][i]
    results_prefixed['best_epoch'] = results['best_epoch']
    results_prefixed['best_macro_f'] = results['best_macro_f']

    with open(fname, 'r', encoding='utf-8') as csvfile:
        header_reader = csv.reader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"')
        header = [elm for elm in header_reader][0]
    csvfile.close()

    results_prefixed.update({key: val for key, val in params.items() if key in header})
    values = results_prefixed

    for elm in header:
        if elm not in values:
            values[elm] = 'Missing'

    with open(fname, 'a', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', quotechar='"', fieldnames=header)
        writer.writerow(values)
    csvfile.close()

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


if __name__=="__main__":
    params = read_hyperparams_from_csv('../hyperparams.csv', 2)

    parser = argparse.ArgumentParser(
        description='Tweet classification using CNN')
    parser.add_argument('--seed', type=int,
                        default=42,
                        help="File with main data")
    args = parser.parse_args()
    print(vars(args))

    vars(args).update(params)
    print(vars(args))
    print(args)