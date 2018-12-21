import sys

from flake8.main import git

if __name__ == '__main__':
    sys.exit(
        git.hook(
            strict=git.config_for('strict'),
            lazy=git.config_for('lazy'),
        )
    )
