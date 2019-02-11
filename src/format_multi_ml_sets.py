import argparse
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import json

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def is_valid_description(d):
    if type(d) != list:
        return False
    elif (type(d[0]) != str) or (pd.isna(d[0])) \
            or (d[0] == 'No abstract available') \
            or (d[0].strip() == ''):
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


def which_terms(l, kwds):
    kwd_inds = []
    for t in l:
        try:
            i = kwds.index(t)
            kwd_inds.append(i)
        except ValueError:
            continue
    return kwd_inds


def get_repeats(record):
    repeat_records = []
    for kwd in record['label']:
        d = {'label': kwd, 'abstract': record['abstract']}
        repeat_records.append(d)
    return repeat_records


def main(infile, outdir, shrink_factor=1, min_term_threshold=500, term_field='subject.NASATerms'):

    LOG.info('Reading {}'.format(infile))
    df = pd.read_json(infile)

    LOG.info('Cleaning dataframe')
    valid_descriptions_index = df.loc[:, 'description'].apply(is_valid_description)
    LOG.info('Using keyword field: {}'.format(term_field))
    valid_terms_index = df.loc[:, term_field].apply(is_valid_term_set)
    valid_terms_and_descriptions = np.vectorize(lambda x, y: x and y)(valid_descriptions_index, valid_terms_index)

    term_sets = df.loc[valid_terms_and_descriptions, term_field].apply(clean_term_sets).tolist()
    terms = Counter([term for term_set in term_sets for term in term_set if term is not ''])
    df.loc[valid_terms_and_descriptions, 'clean.NASATerms'] = term_sets

    LOG.info('Using min_term_threshold of {}.'.format(min_term_threshold))
    terms_over_t = [(term, count) for term, count in terms.most_common() if count >= min_term_threshold]
    # selected_terms = spread_rand_select(n_terms, terms_over_t)

    descriptions = df.loc[valid_terms_and_descriptions, 'description'].apply(clean_description)
    valid_df = df.loc[valid_terms_and_descriptions, :]

    id_to_label = {i: l[0] for i, l in enumerate(terms_over_t)}
    label_to_id = {l[0]: i for i, l in enumerate(terms_over_t)}
    LOG.info('Total of {} ids.'.format(len(id_to_label)))

    LOG.info('Transforming terms to ids.')
    term_ids = np.vectorize(lambda ts: [label_to_id[t] for t in ts if t in label_to_id], otypes=[list])(
        valid_df['clean.NASATerms'])

    full_ml_set = pd.DataFrame()
    full_ml_set['label'] = term_ids
    full_ml_set['abstract'] = valid_df.loc[:, 'description'].apply(lambda x: x[0])
    full_ml_set = full_ml_set[~full_ml_set.abstract.isna()]
    full_ml_set = full_ml_set[~full_ml_set.label.isna()]

    nu_kwds = full_ml_set['label'].apply(lambda x: len(x))
    LOG.info('Average number of targeted keywords per abstract: {}'.format(nu_kwds.mean()))

    LOG.info('Taking {0:.0%} of dataset'.format(shrink_factor))
    shrink_nu = math.ceil(len(full_ml_set) * shrink_factor)
    shuffled_df = full_ml_set.sample(shrink_nu)

    s = 0.333333333
    train_test_delim = math.ceil(len(shuffled_df) * s)

    ml_sets = {'train': shuffled_df.iloc[0:train_test_delim],
               'test': shuffled_df.iloc[train_test_delim: 2 * train_test_delim],
               'dev': shuffled_df.iloc[2 * train_test_delim:]}

    all_kwds = pd.Series()
    expanded_sets = {}
    col_order = ['label', 'abstract']
    for set_type, ml_set in ml_sets.items():
        repeats = ml_set.apply(get_repeats, axis=1)
        all_records = [repeat for repeat_set in repeats for repeat in repeat_set]
        repeated_df = pd.DataFrame(all_records)
        all_kwds = all_kwds.append(repeated_df.label)
        expanded_sets[set_type] = repeated_df.sample(len(repeated_df))[col_order]

    for set_type, ml_set in expanded_sets.items():
        outfile = outdir / Path('{}.tsv'.format(set_type))
        LOG.info('Writing to {}'.format(outfile))
        ml_set[ml_set.label.notna()].to_csv(outfile, sep='\t', index=False)

    out_id_to_label = str(outdir / Path('id_to_label.json'))
    out_label_to_id = str(outdir / Path('label_to_id.json'))

    LOG.info('Writing to {}'.format(out_id_to_label))
    with open(out_id_to_label, 'w') as f0:
        json.dump(id_to_label, f0)

    LOG.info('Writing to {}'.format(out_label_to_id))
    with open(out_label_to_id, 'w') as f0:
        json.dump(label_to_id, f0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create sample train, test, dev sets for several keywords for BERT learning.")
    parser.add_argument('infile', help='input file with abstracts and keywords', type=Path)
    parser.add_argument('outdir', help='output directory in which to place directories of ml sets', type=Path)
    parser.add_argument('shrink_factor', help='by what factor should we shrink our ml sets', type=float)
    parser.add_argument('min_term_threshold', help='under what threshold should we throw out keywords', type=float)
    parser.add_argument('term_field', help='what field has kwds to tag', type=str)
    args = parser.parse_args()

    main(args.infile, args.outdir, args.shrink_factor, args.min_term_threshold, args.term_field)
    LOG.info('Complete.')
