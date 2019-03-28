try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

import os
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


def _rendezvous_error(msg):
    return ValueError("Error initializing torch.distributed using " + msg)


def _file_rendezvous_handler(url):
    def _error(msg):
        return _rendezvous_error("file:// rendezvous: " + msg)

    result = urlparse(url)
    path = result.path
    if not path:
        raise _error("path missing")
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))
    if "rank" not in query:
        raise _error("rank parameter missing")
    if "world_size" not in query:
        raise _error("world size parameter missing")

    rank = int(query["rank"])
    world_size = int(query["world_size"])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using file:// method")


def _tcp_rendezvous_handler(url):
    def _error(msg):
        return _rendezvous_error("tcp:// rendezvous: " + msg)

    result = urlparse(url)
    if not result.port:
        raise _error("port number missing")
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))
    if "rank" not in query:
        raise _error("rank parameter missing")
    if "world_size" not in query:
        raise _error("world size parameter missing")

    rank = int(query["rank"])
    world_size = int(query["world_size"])
    start_daemon = rank == 0
    store = TCPStore(result.hostname, result.port, world_size, start_daemon)
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using tcp:// method")


def _env_rendezvous_handler(url):
    def _error(msg):
        return _rendezvous_error("env:// rendezvous: " + msg)

    def _env_error(var):
        return _error("environment variable %s expected, but not set" % var)

    if not url.startswith("env://"):
        raise _error("url must be equal to `env://`")
    result = urlparse(url)
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))

    if "rank" in query:
        rank = int(query["rank"])
    else:
        rank = os.environ.get("RANK", None)
        if rank is None:
            raise _env_error("RANK")

    if "world_size" in query:
        world_size = int(query["world_size"])
    else:
        world_size = os.environ.get("WORLD_SIZE", None)
        if world_size is None:
            raise _env_error("WORLD_SIZE")

    master_addr = os.environ.get("MASTER_ADDR", None)
    if master_addr is None:
        raise _env_error("MASTER_ADDR")

    master_port = os.environ.get("MASTER_PORT", None)
    if master_port is None:
        raise _env_error("MASTER_PORT")

    # Converting before creating the store
    rank = int(rank)
    world_size = int(world_size)
    master_port = int(master_port)

    # Now start the TCP store daemon on the rank 0
    start_daemon = rank == 0
    store = TCPStore(master_addr, master_port, world_size, start_daemon)
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using env:// method")


register_rendezvous_handler("file", _file_rendezvous_handler)
register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
