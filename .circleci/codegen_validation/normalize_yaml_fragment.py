#!/usr/bin/env python3

import os
import sys
import yaml

# Need to import modules that lie on an upward-relative path
sys.path.append(os.path.join(sys.path[0], '..'))

import cimodel.lib.miniyaml as miniyaml


def regurgitate(depth, use_pyyaml_formatter=False):
    data = yaml.safe_load(sys.stdin)

    if use_pyyaml_formatter:
        output = yaml.dump(data, sort_keys=True)
        sys.stdout.write(output)
    else:
        miniyaml.render(sys.stdout, data, depth)


if __name__ == "__main__":
    regurgitate(3)
