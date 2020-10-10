"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://www.scipy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as `np`::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

To search for documents containing a keyword, do::

  >>> np.lookfor('keyword')
  ... # doctest: +SKIP

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::

  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP

Available subpackages
---------------------
doc
    Topical documentation on broadcasting, indexing, etc.
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
f2py
    Fortran to Python Interface Generator.
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more.

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
dual
    Overwrite certain functions with high-performance Scipy tools
matlib
    Make everything matrices.
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------
Start IPython with the NumPy profile (``ipython -p numpy``), which will
import `numpy` under the alias `np`.  Then, use the ``cpaste`` command to
paste examples into the shell.  To see which functions are available in
`numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""
import sys
import warnings

from ._globals import ModuleDeprecationWarning, VisibleDeprecationWarning
from ._globals import _NoValue

# We first need to detect if we're being called as part of the numpy setup
# procedure itself in a reliable manner.
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False

if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    try:
        from numpy.__config__ import show as show_config
    except ImportError:
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg)

    from .version import git_revision as __git_revision__
    from .version import version as __version__

    __all__ = ['ModuleDeprecationWarning',
               'VisibleDeprecationWarning']

    # Allow distributors to run custom init code
    from . import _distributor_init

    from . import core
    from .core import *
    from . import compat
    from . import lib
    # NOTE: to be revisited following future namespace cleanup.
    # See gh-14454 and gh-15672 for discussion.
    from .lib import *

    from . import linalg
    from . import fft
    from . import polynomial
    from . import random
    from . import ctypeslib
    from . import ma
    from . import matrixlib as _mat
    from .matrixlib import *

    # Make these accessible from numpy name-space
    # but not imported in from numpy import *
    # TODO[gh-6103]: Deprecate these
    from builtins import bool, int, float, complex, object, str
    from .compat import long, unicode

    from .core import round, abs, max, min
    # now that numpy modules are imported, can initialize limits
    core.getlimits._register_known_types()

    __all__.extend(['__version__', 'show_config'])
    __all__.extend(core.__all__)
    __all__.extend(_mat.__all__)
    __all__.extend(lib.__all__)
    __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma'])

    # These are added by `from .core import *` and `core.__all__`, but we
    # overwrite them above with builtins we do _not_ want to export.
    __all__.remove('long')
    __all__.remove('unicode')

    # Remove things that are in the numpy.lib but not in the numpy namespace
    # Note that there is a test (numpy/tests/test_public_api.py:test_numpy_namespace)
    # that prevents adding more things to the main namespace by accident.
    # The list below will grow until the `from .lib import *` fixme above is
    # taken care of
    __all__.remove('Arrayterator')
    del Arrayterator

    # Filter out Cython harmless warnings
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    # oldnumeric and numarray were removed in 1.9. In case some packages import
    # but do not use them, we define them here for backward compatibility.
    oldnumeric = 'removed'
    numarray = 'removed'

    if sys.version_info[:2] >= (3, 7):
        # Importing Tester requires importing all of UnitTest which is not a
        # cheap import Since it is mainly used in test suits, we lazy import it
        # here to save on the order of 10 ms of import time for most users
        #
        # The previous way Tester was imported also had a side effect of adding
        # the full `numpy.testing` namespace
        #
        # module level getattr is only supported in 3.7 onwards
        # https://www.python.org/dev/peps/pep-0562/
        def __getattr__(attr):
            if attr == 'testing':
                import numpy.testing as testing
                return testing
            elif attr == 'Tester':
                from .testing import Tester
                return Tester
            else:
                raise AttributeError("module {!r} has no attribute "
                                     "{!r}".format(__name__, attr))

        def __dir__():
            return list(globals().keys() | {'Tester', 'testing'})

    else:
        # We don't actually use this ourselves anymore, but I'm not 100% sure that
        # no-one else in the world is using it (though I hope not)
        from .testing import Tester

    # Pytest testing
    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester


    def _sanity_check():
        """
        Quick sanity checks for common bugs caused by environment.
        There are some cases e.g. with wrong BLAS ABI that cause wrong
        results under specific runtime conditions that are not necessarily
        achieved during test suite runs, and it is useful to catch those early.

        See https://github.com/numpy/numpy/issues/8577 and other
        similar bug reports.

        """
        try:
            x = ones(2, dtype=float32)
            if not abs(x.dot(x) - 2.0) < 1e-5:
                raise AssertionError()
        except AssertionError:
            msg = ("The current Numpy installation ({!r}) fails to "
                   "pass simple sanity checks. This can be caused for example "
                   "by incorrect BLAS library being linked in, or by mixing "
                   "package managers (pip, conda, apt, ...). Search closed "
                   "numpy issues for similar problems.")
            raise RuntimeError(msg.format(__file__))

    _sanity_check()
    del _sanity_check

    def _mac_os_check():
        """
        Quick Sanity check for Mac OS look for accelerate build bugs.
        Testing numpy polyfit calls init_dgelsd(LAPACK)
        """
        try:
            c = array([3., 2., 1.])
            x = linspace(0, 2, 5)
            y = polyval(c, x)
            _ = polyfit(x, y, 2, cov=True)
        except ValueError:
            pass

    import sys
    if sys.platform == "darwin":
        with warnings.catch_warnings(record=True) as w:
            _mac_os_check()
            # Throw runtime error, if the test failed Check for warning and error_message
            error_message = ""
            if len(w) > 0:
                error_message = "{}: {}".format(w[-1].category.__name__, str(w[-1].message))
                msg = (
                    "Polyfit sanity test emitted a warning, most likely due "
                    "to using a buggy Accelerate backend. "
                    "If you compiled yourself, "
                    "see site.cfg.example for information. "
                    "Otherwise report this to the vendor "
                    "that provided NumPy.\n{}\n".format(
                        error_message))
                raise RuntimeError(msg)
    del _mac_os_check

    # We usually use madvise hugepages support, but on some old kernels it
    # is slow and thus better avoided.
    # Specifically kernel version 4.6 had a bug fix which probably fixed this:
    # https://github.com/torvalds/linux/commit/7cf91a98e607c2f935dbcc177d70011e95b8faff
    import os
    use_hugepage = os.environ.get("NUMPY_MADVISE_HUGEPAGE", None)
    if sys.platform == "linux" and use_hugepage is None:
        # If there is an issue with parsing the kernel version,
        # set use_hugepages to 0. Usage of LooseVersion will handle
        # the kernel version parsing better, but avoided since it
        # will increase the import time. See: #16679 for related discussion.
        try:
            use_hugepage = 1
            kernel_version = os.uname().release.split(".")[:2]
            kernel_version = tuple(int(v) for v in kernel_version)
            if kernel_version < (4, 6):
                use_hugepage = 0
        except ValueError:
            use_hugepages = 0
    elif use_hugepage is None:
        # This is not Linux, so it should not matter, just enable anyway
        use_hugepage = 1
    else:
        use_hugepage = int(use_hugepage)

    # Note that this will currently only make a difference on Linux
    core.multiarray._set_madvise_hugepage(use_hugepage)
