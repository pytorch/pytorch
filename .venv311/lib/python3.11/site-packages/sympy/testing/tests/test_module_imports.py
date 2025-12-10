"""
Checks that SymPy does not contain indirect imports.

An indirect import is importing a symbol from a module that itself imported the
symbol from elsewhere. Such a constellation makes it harder to diagnose
inter-module dependencies and import order problems, and is therefore strongly
discouraged.

(Indirect imports from end-user code is fine and in fact a best practice.)

Implementation note: Forcing Python into actually unloading already-imported
submodules is a tricky and partly undocumented process. To avoid these issues,
the actual diagnostic code is in bin/diagnose_imports, which is run as a
separate, pristine Python process.
"""

import subprocess
import sys
from os.path import abspath, dirname, join, normpath
import inspect

from sympy.testing.pytest import XFAIL

@XFAIL
def test_module_imports_are_direct():
    my_filename = abspath(inspect.getfile(inspect.currentframe()))
    my_dirname = dirname(my_filename)
    diagnose_imports_filename = join(my_dirname, 'diagnose_imports.py')
    diagnose_imports_filename = normpath(diagnose_imports_filename)

    process = subprocess.Popen(
        [
            sys.executable,
            normpath(diagnose_imports_filename),
            '--problems',
            '--by-importer'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=-1)
    output, _ = process.communicate()
    assert output == '', "There are import problems:\n" + output.decode()
