#!/usr/bin/env python3
"""Fortran to Python Interface Generator.

"""
__all__ = ['run_main', 'compile', 'f2py_testing']

import sys
import subprocess
import os

from . import f2py2e
from . import f2py_testing
from . import diagnose

run_main = f2py2e.run_main
main = f2py2e.main


def compile(source,
            modulename='untitled',
            extra_args='',
            verbose=True,
            source_fn=None,
            extension='.f'
           ):
    """
    Build extension module from a Fortran 77 source string with f2py.

    Parameters
    ----------
    source : str or bytes
        Fortran source of module / subroutine to compile

        .. versionchanged:: 1.16.0
           Accept str as well as bytes

    modulename : str, optional
        The name of the compiled python module
    extra_args : str or list, optional
        Additional parameters passed to f2py

        .. versionchanged:: 1.16.0
            A list of args may also be provided.

    verbose : bool, optional
        Print f2py output to screen
    source_fn : str, optional
        Name of the file where the fortran source is written.
        The default is to use a temporary file with the extension
        provided by the `extension` parameter
    extension : {'.f', '.f90'}, optional
        Filename extension if `source_fn` is not provided.
        The extension tells which fortran standard is used.
        The default is `.f`, which implies F77 standard.

        .. versionadded:: 1.11.0

    Returns
    -------
    result : int
        0 on success

    Examples
    --------
    .. include:: compile_session.dat
        :literal:

    """
    import tempfile
    import shlex

    if source_fn is None:
        f, fname = tempfile.mkstemp(suffix=extension)
        # f is a file descriptor so need to close it
        # carefully -- not with .close() directly
        os.close(f)
    else:
        fname = source_fn

    if not isinstance(source, str):
        source = str(source, 'utf-8')
    try:
        with open(fname, 'w') as f:
            f.write(source)

        args = ['-c', '-m', modulename, f.name]

        if isinstance(extra_args, str):
            is_posix = (os.name == 'posix')
            extra_args = shlex.split(extra_args, posix=is_posix)

        args.extend(extra_args)

        c = [sys.executable,
             '-c',
             'import numpy.f2py as f2py2e;f2py2e.main()'] + args
        try:
            output = subprocess.check_output(c)
        except subprocess.CalledProcessError as exc:
            status = exc.returncode
            output = ''
        except OSError:
            # preserve historic status code used by exec_command()
            status = 127
            output = ''
        else:
            status = 0
            output = output.decode()
        if verbose:
            print(output)
    finally:
        if source_fn is None:
            os.remove(fname)
    return status

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
