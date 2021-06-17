import sys
import pathlib
import collections
import os
import trace
import contextlib
import io
import json

# content = open(sys.argv[1], "r").read()
# 

# print(t.results())
INSTALL_DIR = str(pathlib.Path(__file__).resolve().parent)

# print(INSTALL_DIR)
# sys.exit(0)

t = trace.Trace()

progname = sys.argv[1]
arguments = sys.argv[2:]


buf = io.StringIO()
try:
    sys.argv = [progname, *arguments]
    sys.path[0] = os.path.dirname(progname)

    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')
    # try to emulate __main__ namespace as much as possible
    globs = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }
    with contextlib.redirect_stdout(buf):
        t.runctx(code, globs, globs)
except OSError as err:
    sys.exit("Cannot run file %r because: %s" % (sys.argv[0], err))
except SystemExit:
    pass

results = t.results()
r = results
c = results.counts

files = collections.defaultdict(list)

def keep_filename(name):
    if name.startswith("<"):
        return None

    if name.startswith("/"):
        if name.startswith(INSTALL_DIR):
            return name[len(INSTALL_DIR) + 1:]
        return None

    return name


for key_line_tuple, count in results.counts.items():
    filename, lineno = key_line_tuple
    filename = keep_filename(filename)
    if filename:
        files[filename].append(lineno)

# print(buf.getvalue())
# print(json.dumps(files))
# print(json.dumps(files, indent=2))
