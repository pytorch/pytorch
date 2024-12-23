"""
Backward-compatability shims to support Python 3.9 and earlier.
"""

from jaraco.test.cpython import from_test_support, try_import

import_helper = try_import('import_helper') or from_test_support(
    'modules_setup', 'modules_cleanup', 'DirsOnSysPath'
)
os_helper = try_import('os_helper') or from_test_support('temp_dir')
