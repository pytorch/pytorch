from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import torch


def dump(filename):
    schemas = torch._C._jit_get_all_schemas()
    with open(filename, 'w') as f:
        for s in schemas:
            f.write(str(s))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '-f',
        '--filename',
        help='filename to dump the schemas',
        type=str,
        default='schemas.txt')
    args = parser.parse_args()
    dump(args.filename)
