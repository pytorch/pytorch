import os
import sys


def escape_path(path):
    "Can be replaced by pathlib.as_posix when Python 3 is required."

    if os.path.sep != '/' and path is not None:
        return path.replace(os.path.sep, '/')
    return path


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None
