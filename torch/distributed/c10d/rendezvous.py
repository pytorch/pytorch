try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

from . import FileStore, TCPStore


_rendezvous_handlers = {}


def register_rendezvous_handler(scheme, handler):
    """Registers a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Arguments:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    """
    global _rendezvous_handlers
    if scheme in _rendezvous_handlers:
        raise RuntimeError(
            "Rendezvous handler for {}:// already registered".format(scheme)
        )
    _rendezvous_handlers[scheme] = handler


def rendezvous(url, **kwargs):
    global _rendezvous_handlers
    result = urlparse(url)
    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError("No rendezvous handler for {}://".format(result.scheme))
    return _rendezvous_handlers[result.scheme](url, **kwargs)


def _file_rendezvous_handler(url):
    def _error(msg):
        return ValueError("file:// rendezvous: " + msg)

    result = urlparse(url)
    path = result.path
    if not path:
        raise _error("path missing")
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))
    if "rank" not in query:
        raise _error("rank parameter missing")
    if "size" not in query:
        raise _error("size parameter missing")

    rank = int(query["rank"])
    size = int(query["size"])
    store = FileStore(path)
    yield (store, rank, size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using file:// method")


def _tcp_rendezvous_handler(url):
    def _error(msg):
        return ValueError("tcp:// rendezvous: " + msg)

    result = urlparse(url)
    if not result.port:
        raise _error("port number missing")
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))
    if "rank" not in query:
        raise _error("rank parameter missing")
    if "size" not in query:
        raise _error("size parameter missing")

    rank = int(query["rank"])
    size = int(query["size"])
    start_daemon = rank == 0
    store = TCPStore(result.hostname, result.port, start_daemon)
    yield (store, rank, size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using tcp:// method")


register_rendezvous_handler("file", _file_rendezvous_handler)
register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
