import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import random
import math
import numpy as np

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def is_valid_description(d):
    if type(d) != list:
        return False
    elif (type(d[0]) != str) or (pd.isna(d[0])) or (d[0] == 'No abstract available'):
        return False
    else:
        return True


def is_valid_term_set(t):
    if type(t) != list:
        return False
    elif len(t) == 0:
        return False
    else:
        return True


def clean_description(d):
    d = d[0]
    d = d.replace('\t', ' ').replace('\n', ' ').strip()
    return d


def clean_term_sets(ts):
    clean_ts = [t.strip().lower() for t in ts]
    return clean_ts


def spread_rand_select(n, l):
    interval = math.floor(len(l) / n)
    selected_terms = []
    for i in range(n):
        random_offset = np.random.randint(0, int(interval))
        index = i * interval + random_offset
        term = l[index]
        selected_terms.append(term)
    return selected_terms


def has_key_term(l, kwd0, kwd1):
    terms = [0]
    if kwd0 in l:
        terms.append(1)
    if kwd1 in l:
        terms.append(2)
    return terms


def get_train_test_dev(df, descriptions, kwd0, field='clean.NASATerms', shrink_factor=None, balance=False):
    has_term = df.loc[:, field].apply(lambda l: 1 if kwd0 in l else 0)
    full_df = pd.DataFrame({'label': has_term, 'abstract': descriptions})

    yes_df = full_df[full_df.label == 1]
    no_df = full_df[full_df.label == 0]
    if shrink_factor is not None:
        shrink_nu = math.ceil(len(yes_df) * shrink_factor)
        yes_inds = yes_df.sample(shrink_nu).index.tolist()

        shrink_no_nu = math.ceil(len(no_df) * shrink_factor)
        no_inds = no_df.sample(shrink_no_nu).index.tolist()
    else:
        yes_inds = yes_df.index.tolist()
        no_inds = no_df.index.tolist()

    # assumption: more "no"s than "yes"s
    if balance is True:
        no_inds = no_df.sample(len(yes_inds)).index.tolist()

    sample_inds = yes_inds + no_inds
    random.shuffle(sample_inds)
    shuffled_df = full_df.loc[sample_inds]

    s = 0.333333333
    train_test_delim = math.ceil(len(shuffled_df) * s)

    train_set = shuffled_df.iloc[0:train_test_delim]
    test_set = shuffled_df.iloc[train_test_delim: 2 * train_test_delim]
    dev_set = shuffled_df.iloc[2 * train_test_delim:]

    return train_set, test_set, dev_set, full_df


def main(infile, outdir, n_terms=5, shrink_factor=0.1, balance=False):

    LOG.info('Reading {}'.format(infile))
    df = pd.read_json(infile)

    LOG.info('Cleaning dataframe')
    valid_descriptions_index = df.loc[:, 'description'].apply(is_valid_description)
    valid_terms_index = df.loc[:, 'subject.NASATerms'].apply(is_valid_term_set)
    valid_terms_and_descriptions = np.vectorize(lambda x, y: x and y)(valid_descriptions_index, valid_terms_index)

    term_sets = df.loc[valid_terms_and_descriptions, 'subject.NASATerms'].apply(clean_term_sets).tolist()
    terms = Counter([term for term_set in term_sets for term in term_set if term is not ''])

    df.loc[valid_terms_and_descriptions, 'clean.NASATerms'] = term_sets

    terms_over_t = [(term, count) for term, count in terms.most_common() if count > 500]

    selected_terms = spread_rand_select(n_terms, terms_over_t)
    descriptions = df.loc[valid_terms_and_descriptions, 'description'].apply(clean_description)

    LOG.info('Making ml sets with shrink factor {} and balance set to {}'.format(shrink_factor, balance))
    ml_sets = {}
    for t in selected_terms:
        tmp_df = df.loc[:valid_terms_and_descriptions, :]
        train_set, test_set, dev_set, full_df = get_train_test_dev(tmp_df, descriptions, t[0], field='clean.NASATerms',
                                                                   shrink_factor=shrink_factor, balance=balance)
        ml_sets[t[0]] = [train_set, test_set, dev_set]

    for kwd, (train_set, test_set, dev_set) in ml_sets.items():
        kwd = kwd.replace(' ', '_')
        d = outdir / Path(kwd)
        d.mkdir(exist_ok=True)
        LOG.info('Writing sets to {}'.format(d))
        train_set[train_set.label.notna()].to_csv(d / Path('train.tsv'), sep='\t', index=False)
        test_set[test_set.label.notna()].to_csv(d / Path('test.tsv'), sep='\t', index=False)
        dev_set[dev_set.label.notna()].to_csv(d / Path('dev.tsv'), sep='\t', index=False)

    outkwd = outdir / Path('kwds.txt')
    LOG.info('Writing list of kwds to {}'.format(outkwd))
    with open(str(outkwd), 'w') as f0:
        for kwd in ml_sets.keys():
            kwd = kwd.replace(' ', '_')
            f0.write(kwd)
            f0.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create sample train, test, dev sets for several keywords for BERT learning.")
    parser.add_argument('infile', help='input file with abstracts and keywords', type=Path)
    parser.add_argument('outdir', help='output directory in which to place directories of ml sets', type=Path)
    parser.add_argument('n_terms', help='number of keywords for which to create ml sets', type=int)
    parser.add_argument('shrink_factor', help='by what factor should we shrink our ml sets', type=float)
    parser.add_argument('--balance', dest='balance', action='store_true')
    parser.add_argument('--no-balance', dest='balance', action='store_false')
    parser.set_defaults(balance=True)
    args = parser.parse_args()

    main(args.infile, args.outdir, args.n_terms, args.shrink_factor, args.balance)
