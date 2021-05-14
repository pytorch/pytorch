#!/usr/bin/env python3

import sys

from flake8.main import git

if __name__ == '__main__':
    sys.exit(
        git.hook(
            strict=True,
            lazy=git.config_for('lazy'),
        )
    )
