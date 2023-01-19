# -*- coding: utf-8 -*-

"""
requests.compat
~~~~~~~~~~~~~~~

This module handles import compatibility issues between Python 2 and
Python 3.
"""

try:
    import chardet
except ImportError:
    import charset_normalizer as chardet

import sys

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = (_ver[0] == 2)

#: Python 3.x?
is_py3 = (_ver[0] == 3)

has_simplejson = False
try:
    import simplejson as json
    has_simplejson = True
except ImportError:
    import json

# ---------
# Specifics
# ---------

if is_py2:
    from urllib import (
        quote, unquote, quote_plus, unquote_plus, urlencode, getproxies,
        proxy_bypass, proxy_bypass_environment, getproxies_environment)
    from urlparse import urlparse, urlunparse, urljoin, urlsplit, urldefrag
    from urllib2 import parse_http_list
    import cookielib
    from Cookie import Morsel
    from StringIO import StringIO
    # Keep OrderedDict for backwards compatibility.
    from collections import Callable, Mapping, MutableMapping, OrderedDict

    builtin_str = str
    bytes = str
    str = unicode
    basestring = basestring
    numeric_types = (int, long, float)
    integer_types = (int, long)
    JSONDecodeError = ValueError

elif is_py3:
    from urllib.parse import urlparse, urlunparse, urljoin, urlsplit, urlencode, quote, unquote, quote_plus, unquote_plus, urldefrag
    from urllib.request import parse_http_list, getproxies, proxy_bypass, proxy_bypass_environment, getproxies_environment
    from http import cookiejar as cookielib
    from http.cookies import Morsel
    from io import StringIO
    # Keep OrderedDict for backwards compatibility.
    from collections import OrderedDict
    from collections.abc import Callable, Mapping, MutableMapping
    if has_simplejson:
        from simplejson import JSONDecodeError
    else:
        from json import JSONDecodeError

    builtin_str = str
    str = str
    bytes = bytes
    basestring = (str, bytes)
    numeric_types = (int, float)
    integer_types = (int,)
