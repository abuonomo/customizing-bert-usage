"""
@author: Anthony Buonomo
@email: anthony.r.buonomo@nasa.gov

"""

import argparse
import logging

import pandas as pd


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(infile):
    LOG.info(f'Reading {infile}')
    df = pd.read_json(infile)
    print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create sample tsv files for BERT training.")
    parser.add_argument('infile', help='input json file')
    args = parser.parse_args()

    main(args.infile)
