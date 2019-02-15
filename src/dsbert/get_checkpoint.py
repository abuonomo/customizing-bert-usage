import argparse
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def main(indir, outfile):
    files = os.listdir(indir)
    r = re.compile('model.ckpt-[0-9]+.index')
    checkpoints = list(filter(r.match, files))
    values = [int(re.search(r'\d+', checkpoint).group()) for checkpoint in checkpoints]
    i = values.index(max(values))
    LOG.info('Outputting {} to {}'.format(checkpoints[i], outfile))
    with open(str(outfile), 'w') as f0:
        f0.write(checkpoints[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Finds latest model checkpoint file name.")
    parser.add_argument('indir', help='directory in which to find checkpoint', type=Path)
    parser.add_argument('outfile', help='file in which to store the maximum checkpoint', type=Path)
    args = parser.parse_args()

    main(args.indir, args.outfile)