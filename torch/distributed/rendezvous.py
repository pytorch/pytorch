# mypy: allow-untyped-defs
try:
    from urllib.parse import urlparse, urlunparse
except ImportError as e:
    raise ImportError(
        "urllib cannot be found, urlparse from python2 is no longer supported."
    ) from e

import numbers
import os
import sys
from datetime import timedelta
from typing import Dict, Optional, Callable, Iterator, Tuple

from torch.distributed import FileStore, PrefixStore, Store, TCPStore

from .constants import default_pg_timeout


_rendezvous_handlers: Dict[str, Callable[..., Iterator[Tuple[Store, int, int]]]] = {}

__all__ = ["register_rendezvous_handler", "rendezvous"]

def register_rendezvous_handler(scheme, handler):
    """
    Register a new rendezvous handler.

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
            f"Rendezvous handler for {scheme}:// already registered"
        )
    _rendezvous_handlers[scheme] = handler


# Query will have format "rank=0&world_size=1" and is
# converted into {"rank": 0, "world_size": 1}
def _query_to_dict(query: str) -> Dict[str, str]:
    return {pair[0]: pair[1] for pair in (pair.split("=") for pair in filter(None, query.split("&")))}


def _get_use_libuv_from_query_dict(query_dict: Dict[str, str]) -> bool:
    # libuv is the default backend for TCPStore. To enable the non-libuv backend,
    # user can explicitly specify ``use_libuv=0`` in the URL parameter.
    return query_dict.get("use_libuv", os.environ.get("USE_LIBUV", "1")) == "1"


def _rendezvous_helper(url: str, rank: int, world_size_opt: Optional[int], **kwargs):
    result = urlparse(url)
    if world_size_opt is None:
        world_size = -1
        if result.scheme == "env":
            rank = int(os.environ.get("RANK", rank))
            # If the world_size env variable is not present then it is a dynamic group
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
    else:
        world_size = world_size_opt
    if rank != -1 or world_size != -1 or world_size_opt is None:
        query_dict = _query_to_dict(result.query)
        assert (
            "rank" not in query_dict and "world_size" not in query_dict
        ), f"The url: {url} has node-specific arguments(rank, world_size) already."
        if rank != -1:
            query_dict["rank"] = str(rank)
        if world_size != -1 or world_size_opt is None:
            query_dict["world_size"] = str(world_size)
        result = result._replace(
            query=f"{'&'.join([f'{k}={v}' for k, v in query_dict.items()])}"
        )
        url = urlunparse(result)

    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError(f"No rendezvous handler for {result.scheme}://")
    return _rendezvous_handlers[result.scheme](url, **kwargs)


def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    if not isinstance(url, (str, bytes)):
        raise RuntimeError(f"`url` must be a string. {type(url)}: {url}")

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError(f"`rank` must be an integer. {rank}")

    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError(f"`world_size` must be an integer. {world_size}")

    return _rendezvous_helper(url, rank, world_size, **kwargs)


def _create_store_from_options(backend_options, rank):
    store, _, _ = next(_rendezvous_helper(backend_options.init_method, rank, None))
    return store


def _rendezvous_error(msg):
    return ValueError("Error initializing torch.distributed using " + msg)


def _file_rendezvous_handler(url: str, **kwargs):
    def _error(msg):
        return _rendezvous_error("file:// rendezvous: " + msg)

    result = urlparse(url)
    path = result.path
    if sys.platform == "win32":
        import urllib.request

        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            # Normalizing an empty string produces ".", which is not expected.
            path = os.path.normpath(path)

    if not path:
        raise _error("path missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using file:// method")


def _torchelastic_use_agent_store() -> bool:
    return os.environ.get("TORCHELASTIC_USE_AGENT_STORE", None) == str(True)


def _create_c10d_store(hostname, port, rank, world_size, timeout, use_libuv=True) -> Store:
    """
    Smartly creates a c10d Store object on ``rank`` based on whether we need to re-use agent store.

    The TCPStore server is assumed to be hosted
    on ``hostname:port``.

    By default, the TCPStore server uses the asynchronous implementation
    ``LibUVStoreDaemon`` which utilizes libuv.

    If ``torchelastic_use_agent_store()`` is ``True``, then it is assumed that
    the agent leader (node rank 0) hosts the TCPStore server (for which the
    endpoint is specified by the given ``hostname:port``). Hence
    ALL ranks will create and return a TCPStore client (e.g. ``start_daemon=False``).

    If ``torchelastic_use_agent_store()`` is ``False``, then rank 0 will host
    the TCPStore (with multi-tenancy) and it is assumed that rank 0's hostname
    and port are correctly passed via ``hostname`` and ``port``. All
    non-zero ranks will create and return a TCPStore client.
    """
    # check if port is uint16_t
    if not 0 <= port < 2**16:
        raise ValueError(f"port must have value from 0 to 65535 but was {port}.")

    if _torchelastic_use_agent_store():
        attempt = os.environ["TORCHELASTIC_RESTART_COUNT"]
        tcp_store = TCPStore(hostname, port, world_size, False, timeout)
        return PrefixStore(f"/worker/attempt_{attempt}", tcp_store)
    else:
        start_daemon = rank == 0
        return TCPStore(
            hostname, port, world_size, start_daemon, timeout, multi_tenant=True, use_libuv=use_libuv
        )


def _tcp_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    def _error(msg):
        return _rendezvous_error("tcp:// rendezvous: " + msg)

    result = urlparse(url)
    if not result.port:
        raise _error("port number missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    assert result.hostname is not None

    store = _create_c10d_store(result.hostname, result.port, rank, world_size, timeout, use_libuv)

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using tcp:// method")


def _env_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    def _error(msg):
        return _rendezvous_error("env:// rendezvous: " + msg)

    def _env_error(var):
        return _error(f"environment variable {var} expected, but not set")

    def _get_env_or_raise(env_var: str) -> str:
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val

    result = urlparse(url)
    query_dict = _query_to_dict(result.query)

    rank: int
    world_size: int
    master_port: int
    master_addr: str

    if "rank" in query_dict:
        rank = int(query_dict["rank"])
    else:
        rank = int(_get_env_or_raise("RANK"))

    if "world_size" in query_dict:
        world_size = int(query_dict["world_size"])
    else:
        world_size = int(_get_env_or_raise("WORLD_SIZE"))


    master_addr = _get_env_or_raise("MASTER_ADDR")
    master_port = int(_get_env_or_raise("MASTER_PORT"))
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    store = _create_c10d_store(master_addr, master_port, rank, world_size, timeout, use_libuv)

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using env:// method")


register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
register_rendezvous_handler("file", _file_rendezvous_handler)
