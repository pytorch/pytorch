#!/usr/bin/env python3

import sys
import yaml
import cimodel.lib.miniyaml as miniyaml


USE_PYYAML_FORMATTER = False


def regurgitate(depth):
    data = yaml.safe_load(sys.stdin)

    if USE_PYYAML_FORMATTER:
        output = yaml.dump(data, sort_keys=True)
        sys.stdout.write(output)
    else:

        miniyaml.render(sys.stdout, data, depth)


if __name__ == "__main__":
    regurgitate(3)

