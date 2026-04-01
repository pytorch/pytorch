#!/usr/bin/env python3

import subprocess
import sys

for args in [['slangc',
              '-target', 'wgsl',
              '-stage', '{}'.format(stage),
              '-entry', '{}Main'.format(stage),
              '-o', 'shader.{}.wgsl'.format(stage),
              'shader.slang']
             for stage in ['vertex', 'fragment']]:
    print("Running '{}'...".format(' '.join(args)))
    result = subprocess.run(args)
    if result.returncode != 0:
        print('Failed!')
        sys.exit(1)
    else:
        print('Succeeded!')
