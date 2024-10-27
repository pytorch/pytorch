# flake8: noqa

import contextlib
import builtins
import sys

from test.support import requires_zlib
import test.support


ModuleNotFoundError = getattr(builtins, 'ModuleNotFoundError', ImportError)

try:
    from test.support.warnings_helper import check_warnings
except (ModuleNotFoundError, ImportError):
    from test.support import check_warnings


try:
    from test.support.os_helper import (
        rmtree,
        EnvironmentVarGuard,
        unlink,
        skip_unless_symlink,
        temp_dir,
    )
except (ModuleNotFoundError, ImportError):
    from test.support import (
        rmtree,
        EnvironmentVarGuard,
        unlink,
        skip_unless_symlink,
        temp_dir,
    )


try:
    from test.support.import_helper import (
        DirsOnSysPath,
        CleanImport,
    )
except (ModuleNotFoundError, ImportError):
    from test.support import (
        DirsOnSysPath,
        CleanImport,
    )


if sys.version_info < (3, 9):
    requires_zlib = lambda: test.support.requires_zlib
