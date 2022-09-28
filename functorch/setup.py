# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# This is a dummy setup.py that does not do anything

import os
import subprocess
from setuptools import setup
import warnings
import torch

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    sha = 'Unknown'
package_name = 'functorch'

requirements = [
    # This represents a nightly version of PyTorch.
    # It can be installed as a binary or from source.
    "torch>=1.13.0.dev",
]

extras = {}
extras["aot"] = ["networkx", ]


if __name__ == '__main__':
    try:
        setup(
            # Metadata
            name=package_name,
            version=torch.__version__,
            author='PyTorch Core Team',
            url="https://github.com/pytorch/functorch",
            description='JAX-like composable function transforms for PyTorch',
            license='BSD',

            # Package info
            packages=[],
            install_requires=requirements,
            extras_require=extras,
        )
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    warnings.warn(
        'Installing PyTorch from source or from a nightly binary already '
        'installs functorch (as of 9/14/2022), so there is no need to cd '
        'into functorch and run `python setup.py {install, develop}` anymore. '
        'We will soon remove this method of installing functorch.',
        DeprecationWarning)
