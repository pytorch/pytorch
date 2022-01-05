# Owner(s): ["module: unknown"]

import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required args. Raises error if they aren't passed.
    parser.add_argument('--foo', help='foo', required=True)
    parser.add_argument('--bar', help='bar', required=True)
    _ = parser.parse_args()

    x = torch.ones((3, 3), requires_grad=True)
    (3 * x).sum().backward()
