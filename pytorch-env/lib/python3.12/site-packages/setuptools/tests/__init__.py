import locale
import sys

import pytest


__all__ = ['fail_on_ascii']

locale_encoding = (
    locale.getencoding()
    if sys.version_info >= (3, 11)
    else locale.getpreferredencoding(False)
)
is_ascii = locale_encoding == 'ANSI_X3.4-1968'
fail_on_ascii = pytest.mark.xfail(is_ascii, reason="Test fails in this locale")
