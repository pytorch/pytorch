from __future__ import absolute_import
import re
import datetime
import logging
import os
import socket
from socket import error as SocketError, timeout as SocketTimeout
import warnings
from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException  # noqa: F401

try:  # Compiled with SSL?
    import ssl

    BaseSSLError = ssl.SSLError
except (ImportError, AttributeError):  # Platform-specific: No SSL.
    ssl = None

    class BaseSSLError(BaseException):
        pass


try:
    # Python 3: not a no-op, we're adding this to the namespace so it can be imported.
    ConnectionError = ConnectionError
except NameError:
    # Python 2
    class ConnectionError(Exception):
        pass


from .exceptions import (
    NewConnectionError,
    ConnectTimeoutError,
    SubjectAltNameWarning,
    SystemTimeWarning,
)
from .packages.ssl_match_hostname import match_hostname, CertificateError

from .util.ssl_ import (
    resolve_cert_reqs,
    resolve_ssl_version,
    assert_fingerprint,
    create_urllib3_context,
    ssl_wrap_socket,
)


from .util import connection

from ._collections import HTTPHeaderDict

log = logging.getLogger(__name__)

port_by_scheme = {"http": 80, "https": 443}

# When it comes time to update this value as a part of regular maintenance
# (ie test_recent_date is failing) update it to ~6 months before the current date.
RECENT_DATE = datetime.date(2019, 1, 1)

_CONTAINS_CONTROL_CHAR_RE = re.compile(r"[^-!#$%&'*+.^_`|~0-9a-zA-Z]")


class DummyConnection(object):
    """Used to detect a failed ConnectionCls import."""

    pass


class HTTPConnection(_HTTPConnection, object):
    """
    Based on httplib.HTTPConnection but provides an extra constructor
    backwards-compatibility layer between older and newer Pythons.

    Additional keyword parameters are used to configure attributes of the connection.
    Accepted parameters include:

      - ``strict``: See the documentation on :class:`urllib3.connectionpool.HTTPConnectionPool`
      - ``source_address``: Set the source address for the current connection.
      - ``socket_options``: Set specific options on the underlying socket. If not specified, then
        defaults are loaded from ``HTTPConnection.default_socket_options`` which includes disabling
        Nagle's algorithm (sets TCP_NODELAY to 1) unless the connection is behind a proxy.

        For example, if you wish to enable TCP Keep Alive in addition to the defaults,
        you might pass::

            HTTPConnection.default_socket_options + [
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            ]

        Or you may want to disable the defaults by passing an empty list (e.g., ``[]``).
    """

    default_port = port_by_scheme["http"]

    #: Disable Nagle's algorithm by default.
    #: ``[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]``
    default_socket_options = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]

    #: Whether this connection verifies the host's certificate.
    is_verified = False

    def __init__(self, *args, **kw):
        if not six.PY2:
            kw.pop("strict", None)

        # Pre-set source_address.
        self.source_address = kw.get("source_address")

        #: The socket options provided by the user. If no options are
        #: provided, we use the default options.
        self.socket_options = kw.pop("socket_options", self.default_socket_options)

        _HTTPConnection.__init__(self, *args, **kw)

    @property
    def host(self):
        """
        Getter method to remove any trailing dots that indicate the hostname is an FQDN.

        In general, SSL certificates don't include the trailing dot indicating a
        fully-qualified domain name, and thus, they don't validate properly when
        checked against a domain name that includes the dot. In addition, some
        servers may not expect to receive the trailing dot when provided.

        However, the hostname with trailing dot is critical to DNS resolution; doing a
        lookup with the trailing dot will properly only resolve the appropriate FQDN,
        whereas a lookup without a trailing dot will search the system's search domain
        list. Thus, it's important to keep the original host around for use only in
        those cases where it's appropriate (i.e., when doing DNS lookup to establish the
        actual TCP connection across which we're going to send HTTP requests).
        """
        return self._dns_host.rstrip(".")

    @host.setter
    def host(self, value):
        """
        Setter for the `host` property.

        We assume that only urllib3 uses the _dns_host attribute; httplib itself
        only uses `host`, and it seems reasonable that other libraries follow suit.
        """
        self._dns_host = value

    def _new_conn(self):
        """ Establish a socket connection and set nodelay settings on it.

        :return: New socket connection.
        """
        extra_kw = {}
        if self.source_address:
            extra_kw["source_address"] = self.source_address

        if self.socket_options:
            extra_kw["socket_options"] = self.socket_options

        try:
            conn = connection.create_connection(
                (self._dns_host, self.port), self.timeout, **extra_kw
            )

        except SocketTimeout:
            raise ConnectTimeoutError(
                self,
                "Connection to %s timed out. (connect timeout=%s)"
                % (self.host, self.timeout),
            )

        except SocketError as e:
            raise NewConnectionError(
                self, "Failed to establish a new connection: %s" % e
            )

        return conn

    def _prepare_conn(self, conn):
        self.sock = conn
        # Google App Engine's httplib does not define _tunnel_host
        if getattr(self, "_tunnel_host", None):
            # TODO: Fix tunnel so it doesn't depend on self.sock state.
            self._tunnel()
            # Mark this connection as not reusable
            self.auto_open = 0

    def connect(self):
        conn = self._new_conn()
        self._prepare_conn(conn)

    def putrequest(self, method, url, *args, **kwargs):
        """Send a request to the server"""
        match = _CONTAINS_CONTROL_CHAR_RE.search(method)
        if match:
            raise ValueError(
                "Method cannot contain non-token characters %r (found at least %r)"
                % (method, match.group())
            )

        return _HTTPConnection.putrequest(self, method, url, *args, **kwargs)

    def request_chunked(self, method, url, body=None, headers=None):
        """
        Alternative to the common request method, which sends the
        body with chunked encoding and not as one block
        """
        headers = HTTPHeaderDict(headers if headers is not None else {})
        skip_accept_encoding = "accept-encoding" in headers
        skip_host = "host" in headers
        self.putrequest(
            method, url, skip_accept_encoding=skip_accept_encoding, skip_host=skip_host
        )
        for header, value in headers.items():
            self.putheader(header, value)
        if "transfer-encoding" not in headers:
            self.putheader("Transfer-Encoding", "chunked")
        self.endheaders()

        if body is not None:
            stringish_types = six.string_types + (bytes,)
            if isinstance(body, stringish_types):
                body = (body,)
            for chunk in body:
                if not chunk:
                    continue
                if not isinstance(chunk, bytes):
                    chunk = chunk.encode("utf8")
                len_str = hex(len(chunk))[2:]
                self.send(len_str.encode("utf-8"))
                self.send(b"\r\n")
                self.send(chunk)
                self.send(b"\r\n")

        # After the if clause, to always have a closed body
        self.send(b"0\r\n\r\n")


class HTTPSConnection(HTTPConnection):
    default_port = port_by_scheme["https"]

    cert_reqs = None
    ca_certs = None
    ca_cert_dir = None
    ca_cert_data = None
    ssl_version = None
    assert_fingerprint = None

    def __init__(
        self,
        host,
        port=None,
        key_file=None,
        cert_file=None,
        key_password=None,
        strict=None,
        timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
        ssl_context=None,
        server_hostname=None,
        **kw
    ):

        HTTPConnection.__init__(self, host, port, strict=strict, timeout=timeout, **kw)

        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname

        # Required property for Google AppEngine 1.9.0 which otherwise causes
        # HTTPS requests to go out as HTTP. (See Issue #356)
        self._protocol = "https"

    def set_cert(
        self,
        key_file=None,
        cert_file=None,
        cert_reqs=None,
        key_password=None,
        ca_certs=None,
        assert_hostname=None,
        assert_fingerprint=None,
        ca_cert_dir=None,
        ca_cert_data=None,
    ):
        """
        This method should only be called once, before the connection is used.
        """
        # If cert_reqs is not provided we'll assume CERT_REQUIRED unless we also
        # have an SSLContext object in which case we'll use its verify_mode.
        if cert_reqs is None:
            if self.ssl_context is not None:
                cert_reqs = self.ssl_context.verify_mode
            else:
                cert_reqs = resolve_cert_reqs(None)

        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.key_password = key_password
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

    def connect(self):
        # Add certificate verification
        conn = self._new_conn()
        hostname = self.host

        # Google App Engine's httplib does not define _tunnel_host
        if getattr(self, "_tunnel_host", None):
            self.sock = conn
            # Calls self._set_hostport(), so self.host is
            # self._tunnel_host below.
            self._tunnel()
            # Mark this connection as not reusable
            self.auto_open = 0

            # Override the host with the one we're requesting data from.
            hostname = self._tunnel_host

        server_hostname = hostname
        if self.server_hostname is not None:
            server_hostname = self.server_hostname

        is_time_off = datetime.date.today() < RECENT_DATE
        if is_time_off:
            warnings.warn(
                (
                    "System time is way off (before {0}). This will probably "
                    "lead to SSL verification errors"
                ).format(RECENT_DATE),
                SystemTimeWarning,
            )

        # Wrap socket using verification with the root certs in
        # trusted_root_certs
        default_ssl_context = False
        if self.ssl_context is None:
            default_ssl_context = True
            self.ssl_context = create_urllib3_context(
                ssl_version=resolve_ssl_version(self.ssl_version),
                cert_reqs=resolve_cert_reqs(self.cert_reqs),
            )

        context = self.ssl_context
        context.verify_mode = resolve_cert_reqs(self.cert_reqs)

        # Try to load OS default certs if none are given.
        # Works well on Windows (requires Python3.4+)
        if (
            not self.ca_certs
            and not self.ca_cert_dir
            and not self.ca_cert_data
            and default_ssl_context
            and hasattr(context, "load_default_certs")
        ):
            context.load_default_certs()

        self.sock = ssl_wrap_socket(
            sock=conn,
            keyfile=self.key_file,
            certfile=self.cert_file,
            key_password=self.key_password,
            ca_certs=self.ca_certs,
            ca_cert_dir=self.ca_cert_dir,
            ca_cert_data=self.ca_cert_data,
            server_hostname=server_hostname,
            ssl_context=context,
        )

        if self.assert_fingerprint:
            assert_fingerprint(
                self.sock.getpeercert(binary_form=True), self.assert_fingerprint
            )
        elif (
            context.verify_mode != ssl.CERT_NONE
            and not getattr(context, "check_hostname", False)
            and self.assert_hostname is not False
        ):
            # While urllib3 attempts to always turn off hostname matching from
            # the TLS library, this cannot always be done. So we check whether
            # the TLS Library still thinks it's matching hostnames.
            cert = self.sock.getpeercert()
            if not cert.get("subjectAltName", ()):
                warnings.warn(
                    (
                        "Certificate for {0} has no `subjectAltName`, falling back to check for a "
                        "`commonName` for now. This feature is being removed by major browsers and "
                        "deprecated by RFC 2818. (See https://github.com/urllib3/urllib3/issues/497 "
                        "for details.)".format(hostname)
                    ),
                    SubjectAltNameWarning,
                )
            _match_hostname(cert, self.assert_hostname or server_hostname)

        self.is_verified = (
            context.verify_mode == ssl.CERT_REQUIRED
            or self.assert_fingerprint is not None
        )


def _match_hostname(cert, asserted_hostname):
    try:
        match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        log.warning(
            "Certificate did not match expected hostname: %s. Certificate: %s",
            asserted_hostname,
            cert,
        )
        # Add cert to exception and reraise so client code can inspect
        # the cert when catching the exception, if they want to
        e._peer_cert = cert
        raise


if not ssl:
    HTTPSConnection = DummyConnection  # noqa: F811


VerifiedHTTPSConnection = HTTPSConnection
