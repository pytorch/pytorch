"""xmlrpclib.Transport implementation
"""

import logging

# NOTE: XMLRPC Client is not annotated in typeshed as on 2017-07-17, which is
#       why we ignore the type on this import
from pip._vendor.six.moves import xmlrpc_client  # type: ignore
from pip._vendor.six.moves.urllib import parse as urllib_parse

from pip._internal.exceptions import NetworkConnectionError
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Dict
    from pip._internal.network.session import PipSession


logger = logging.getLogger(__name__)


class PipXmlrpcTransport(xmlrpc_client.Transport):
    """Provide a `xmlrpclib.Transport` implementation via a `PipSession`
    object.
    """

    def __init__(self, index_url, session, use_datetime=False):
        # type: (str, PipSession, bool) -> None
        xmlrpc_client.Transport.__init__(self, use_datetime)
        index_parts = urllib_parse.urlparse(index_url)
        self._scheme = index_parts.scheme
        self._session = session

    def request(self, host, handler, request_body, verbose=False):
        # type: (str, str, Dict[str, str], bool) -> None
        parts = (self._scheme, host, handler, None, None, None)
        url = urllib_parse.urlunparse(parts)
        try:
            headers = {'Content-Type': 'text/xml'}
            response = self._session.post(url, data=request_body,
                                          headers=headers, stream=True)
            raise_for_status(response)
            self.verbose = verbose
            return self.parse_response(response.raw)
        except NetworkConnectionError as exc:
            assert exc.response
            logger.critical(
                "HTTP error %s while getting %s",
                exc.response.status_code, url,
            )
            raise
