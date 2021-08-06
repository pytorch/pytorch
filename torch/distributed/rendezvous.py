try:
    from urllib.parse import urlparse, urlunparse
except ImportError:
    raise ImportError("urllib cannot be found, urlparse from python2 is no longer supported.")

import torch._six as six
import numbers
import os
import sys
from datetime import timedelta
from typing import Optional, Dict, Union
from torch.distributed import FileStore, TCPStore, PrefixStore
from .constants import default_pg_timeout

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

    Args:
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


def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    if not isinstance(url, six.string_classes):
        raise RuntimeError("`url` must be a string. {}: {}".format(type(url), url))

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError("`rank` must be an integer. {}".format(rank))

    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError("`world_size` must be an integer. {}".format(world_size))

    # Append node-specific arguments.
    result = urlparse(url)
    if rank != -1 or world_size != -1:
        query_dict: Dict[str, Union[int, str]] = dict(
            pair.split("=") for pair in filter(None, result.query.split("&"))
        )
        assert (
            "rank" not in query_dict and "world_size" not in query_dict
        ), "The url: {url} has node-specific arguments(rank, world_size) already.".format(
            url=url
        )
        if rank != -1:
            query_dict["rank"] = rank
        if world_size != -1:
            query_dict["world_size"] = world_size

        result = result._replace(
            query="{}".format("&".join(["{}={}".format(k, v) for k, v in query_dict.items()]))
        )
        url = urlunparse(result)

    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError("No rendezvous handler for {}://".format(result.scheme))
    return _rendezvous_handlers[result.scheme](url, **kwargs)


def _rendezvous_error(msg):
    return ValueError("Error initializing torch.distributed using " + msg)


def _file_rendezvous_handler(url: str, **kwargs):
    def _error(msg):
        return _rendezvous_error("file:// rendezvous: " + msg)

    result = urlparse(url)
    path = result.path
    if sys.platform == 'win32':
        import urllib.request
        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            # Normalizing an empty string produces ".", which is not expected.
            path = os.path.normpath(path)

    if not path:
        raise _error("path missing")
    query: Dict[str, str]
    # mypy doesn't allow dict() to accept List of values (#257)
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))  # type: ignore[misc, arg-type]
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


def _tcp_rendezvous_handler(url: str, timeout: timedelta = default_pg_timeout, **kwargs):
    def _error(msg):
        return _rendezvous_error("tcp:// rendezvous: " + msg)

    result = urlparse(url)
    if not result.port:
        raise _error("port number missing")
    query: Dict[str, Union[int, str]]
    # mypy doesn't allow dict() to accept List of values (#257)
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))  # type: ignore[misc, arg-type]
    if "rank" not in query:
        raise _error("rank parameter missing")
    if "world_size" not in query:
        raise _error("world size parameter missing")

    rank = int(query["rank"])
    world_size = int(query["world_size"])
    start_daemon = rank == 0
    assert result.hostname is not None
    store = TCPStore(  # type: ignore[call-arg]
        result.hostname, result.port, world_size, start_daemon, timeout, multi_tenant=True
    )
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using tcp:// method")


def _env_rendezvous_handler(url: str, timeout: timedelta = default_pg_timeout, **kwargs):
    def _error(msg):
        return _rendezvous_error("env:// rendezvous: " + msg)

    def _env_error(var):
        return _error("environment variable %s expected, but not set" % var)

    def _get_env_or_raise(env_var: str) -> str:
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val

    result = urlparse(url)
    query: Dict[str, Union[int, str]]
    # mypy doesn't allow dict() to accept List of values (#257)
    query = dict(pair.split("=") for pair in filter(None, result.query.split("&")))  # type: ignore[misc, arg-type]

    rank: Optional[Union[str, int]]
    world_size: Optional[Union[str, int]]
    master_port: Optional[Union[str, int]]

    if "rank" in query:
        rank = int(query["rank"])
    else:
        rank = int(_get_env_or_raise("RANK"))

    if "world_size" in query:
        world_size = int(query["world_size"])
    else:
        world_size = int(_get_env_or_raise("WORLD_SIZE"))

    master_addr = _get_env_or_raise("MASTER_ADDR")
    master_port = int(_get_env_or_raise("MASTER_PORT"))


    use_torchelastic_store = os.environ.get("TORCHELASTIC_USE_AGENT_STORE", None)

    if use_torchelastic_store == str(True):
        attempt = os.environ["TORCHELASTIC_RESTART_COUNT"]
        worker_process_prefix = f"/worker/attempt_{attempt}"
        # When TORCHELASTIC_USE_AGENT_STORE is set up, the worker process is assumed
        # to be invoked by the torchelastic agent. Torchelastic agent creates a tcp daemon thread
        # on the GROUP_RANK=0, as a result all user worker processes should create store with: daemon=False
        tcp_store = TCPStore(master_addr, master_port, world_size, False, timeout)
        # Each if-else condition returns due to: https://github.com/python/mypy/issues/1191
        yield (PrefixStore(worker_process_prefix, tcp_store), rank, world_size)
    else:
        # Start the TCP store daemon on the rank 0
        start_daemon = rank == 0
        store = TCPStore(  # type: ignore[call-arg]
            master_addr, master_port, world_size, start_daemon, timeout, multi_tenant=True
        )
        # Each if-else condition returns due to: https://github.com/python/mypy/issues/1191
        yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using env:// method")

register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
register_rendezvous_handler("file", _file_rendezvous_handler)
