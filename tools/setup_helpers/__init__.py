import os


def escape_path(path):
    "Can be replaced by pathlib.as_posix when Python 3 is required."

    if os.path.sep != '/' and path is not None:
        return path.replace(os.path.sep, '/')
    return path
