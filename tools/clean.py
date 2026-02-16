import glob
import os
import shutil

from pathlib import Path

def clean():
    """Clean, that is remove all files in .gitignore except in the NOT-CLEAN-FILES section."""
    ignores = Path(".gitignore").read_text(encoding="utf-8")
    for wildcard in filter(None, ignores.splitlines()):
        if wildcard.strip().startswith("#"):
            if "BEGIN NOT-CLEAN-FILES" in wildcard:
                # Marker is found and stop reading .gitignore.
                break
            # Ignore lines which begin with '#'.
        else:
            # Don't remove absolute paths from the system
            wildcard = wildcard.lstrip("./")
            for filename in glob.iglob(wildcard):
                try:
                    os.remove(filename)
                except OSError:
                    shutil.rmtree(filename, ignore_errors=True)
