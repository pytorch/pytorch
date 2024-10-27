"""ASCII-ART 2D pretty-printer"""

from .pretty import (pretty, pretty_print, pprint, pprint_use_unicode,
    pprint_try_use_unicode, pager_print)

# if unicode output is available -- let's use it
pprint_try_use_unicode()

__all__ = [
    'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode', 'pager_print',
]
