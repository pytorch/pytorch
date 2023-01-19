import os
import sys

from pip._vendor.six.moves.urllib import parse as urllib_parse
from pip._vendor.six.moves.urllib import request as urllib_request

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Optional, Text, Union


def get_url_scheme(url):
    # type: (Union[str, Text]) -> Optional[Text]
    if ':' not in url:
        return None
    return url.split(':', 1)[0].lower()


def path_to_url(path):
    # type: (Union[str, Text]) -> str
    """
    Convert a path to a file: URL.  The path will be made absolute and have
    quoted path parts.
    """
    path = os.path.normpath(os.path.abspath(path))
    url = urllib_parse.urljoin('file:', urllib_request.pathname2url(path))
    return url


def url_to_path(url):
    # type: (str) -> str
    """
    Convert a file: URL to a path.
    """
    assert url.startswith('file:'), (
        "You can only turn file: urls into filenames (not {url!r})"
        .format(**locals()))

    _, netloc, path, _, _ = urllib_parse.urlsplit(url)

    if not netloc or netloc == 'localhost':
        # According to RFC 8089, same as empty authority.
        netloc = ''
    elif sys.platform == 'win32':
        # If we have a UNC path, prepend UNC share notation.
        netloc = '\\\\' + netloc
    else:
        raise ValueError(
            'non-local file URIs are not supported on this platform: {url!r}'
            .format(**locals())
        )

    path = urllib_request.url2pathname(netloc + path)
    return path
