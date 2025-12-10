import asyncio
import io
import logging
import re
import weakref
from copy import copy
from urllib.parse import urlparse

import aiohttp
import yarl

from fsspec.asyn import AbstractAsyncStreamedFile, AsyncFileSystem, sync, sync_wrapper
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.exceptions import FSTimeoutError
from fsspec.spec import AbstractBufferedFile
from fsspec.utils import (
    DEFAULT_BLOCK_SIZE,
    glob_translate,
    isfilelike,
    nullcontext,
    tokenize,
)

from ..caching import AllBytes

# https://stackoverflow.com/a/15926317/3821154
ex = re.compile(r"""<(a|A)\s+(?:[^>]*?\s+)?(href|HREF)=["'](?P<url>[^"']+)""")
ex2 = re.compile(r"""(?P<url>http[s]?://[-a-zA-Z0-9@:%_+.~#?&/=]+)""")
logger = logging.getLogger("fsspec.http")


async def get_client(**kwargs):
    return aiohttp.ClientSession(**kwargs)


class HTTPFileSystem(AsyncFileSystem):
    """
    Simple File-System for fetching data via HTTP(S)

    ``ls()`` is implemented by loading the parent page and doing a regex
    match on the result. If simple_link=True, anything of the form
    "http(s)://server.com/stuff?thing=other"; otherwise only links within
    HTML href tags will be used.
    """

    protocol = ("http", "https")
    sep = "/"

    def __init__(
        self,
        simple_links=True,
        block_size=None,
        same_scheme=True,
        size_policy=None,
        cache_type="bytes",
        cache_options=None,
        asynchronous=False,
        loop=None,
        client_kwargs=None,
        get_client=get_client,
        encoded=False,
        **storage_options,
    ):
        """
        NB: if this is called async, you must await set_client

        Parameters
        ----------
        block_size: int
            Blocks to read bytes; if 0, will default to raw requests file-like
            objects instead of HTTPFile instances
        simple_links: bool
            If True, will consider both HTML <a> tags and anything that looks
            like a URL; if False, will consider only the former.
        same_scheme: True
            When doing ls/glob, if this is True, only consider paths that have
            http/https matching the input URLs.
        size_policy: this argument is deprecated
        client_kwargs: dict
            Passed to aiohttp.ClientSession, see
            https://docs.aiohttp.org/en/stable/client_reference.html
            For example, ``{'auth': aiohttp.BasicAuth('user', 'pass')}``
        get_client: Callable[..., aiohttp.ClientSession]
            A callable, which takes keyword arguments and constructs
            an aiohttp.ClientSession. Its state will be managed by
            the HTTPFileSystem class.
        storage_options: key-value
            Any other parameters passed on to requests
        cache_type, cache_options: defaults used in open()
        """
        super().__init__(self, asynchronous=asynchronous, loop=loop, **storage_options)
        self.block_size = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
        self.simple_links = simple_links
        self.same_schema = same_scheme
        self.cache_type = cache_type
        self.cache_options = cache_options
        self.client_kwargs = client_kwargs or {}
        self.get_client = get_client
        self.encoded = encoded
        self.kwargs = storage_options
        self._session = None

        # Clean caching-related parameters from `storage_options`
        # before propagating them as `request_options` through `self.kwargs`.
        # TODO: Maybe rename `self.kwargs` to `self.request_options` to make
        #       it clearer.
        request_options = copy(storage_options)
        self.use_listings_cache = request_options.pop("use_listings_cache", False)
        request_options.pop("listings_expiry_time", None)
        request_options.pop("max_paths", None)
        request_options.pop("skip_instance_cache", None)
        self.kwargs = request_options

    @property
    def fsid(self):
        return "http"

    def encode_url(self, url):
        return yarl.URL(url, encoded=self.encoded)

    @staticmethod
    def close_session(loop, session):
        if loop is not None and loop.is_running():
            try:
                sync(loop, session.close, timeout=0.1)
                return
            except (TimeoutError, FSTimeoutError, NotImplementedError):
                pass
        connector = getattr(session, "_connector", None)
        if connector is not None:
            # close after loop is dead
            connector._close()

    async def set_session(self):
        if self._session is None:
            self._session = await self.get_client(loop=self.loop, **self.client_kwargs)
            if not self.asynchronous:
                weakref.finalize(self, self.close_session, self.loop, self._session)
        return self._session

    @classmethod
    def _strip_protocol(cls, path):
        """For HTTP, we always want to keep the full URL"""
        return path

    @classmethod
    def _parent(cls, path):
        # override, since _strip_protocol is different for URLs
        par = super()._parent(path)
        if len(par) > 7:  # "http://..."
            return par
        return ""

    async def _ls_real(self, url, detail=True, **kwargs):
        # ignoring URL-encoded arguments
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)
        session = await self.set_session()
        async with session.get(self.encode_url(url), **self.kwargs) as r:
            self._raise_not_found_for_status(r, url)

            if "Content-Type" in r.headers:
                mimetype = r.headers["Content-Type"].partition(";")[0]
            else:
                mimetype = None

            if mimetype in ("text/html", None):
                try:
                    text = await r.text(errors="ignore")
                    if self.simple_links:
                        links = ex2.findall(text) + [u[2] for u in ex.findall(text)]
                    else:
                        links = [u[2] for u in ex.findall(text)]
                except UnicodeDecodeError:
                    links = []  # binary, not HTML
            else:
                links = []

        out = set()
        parts = urlparse(url)
        for l in links:
            if isinstance(l, tuple):
                l = l[1]
            if l.startswith("/") and len(l) > 1:
                # absolute URL on this server
                l = f"{parts.scheme}://{parts.netloc}{l}"
            if l.startswith("http"):
                if self.same_schema and l.startswith(url.rstrip("/") + "/"):
                    out.add(l)
                elif l.replace("https", "http").startswith(
                    url.replace("https", "http").rstrip("/") + "/"
                ):
                    # allowed to cross http <-> https
                    out.add(l)
            else:
                if l not in ["..", "../"]:
                    # Ignore FTP-like "parent"
                    out.add("/".join([url.rstrip("/"), l.lstrip("/")]))
        if not out and url.endswith("/"):
            out = await self._ls_real(url.rstrip("/"), detail=False)
        if detail:
            return [
                {
                    "name": u,
                    "size": None,
                    "type": "directory" if u.endswith("/") else "file",
                }
                for u in out
            ]
        else:
            return sorted(out)

    async def _ls(self, url, detail=True, **kwargs):
        if self.use_listings_cache and url in self.dircache:
            out = self.dircache[url]
        else:
            out = await self._ls_real(url, detail=detail, **kwargs)
            self.dircache[url] = out
        return out

    ls = sync_wrapper(_ls)

    def _raise_not_found_for_status(self, response, url):
        """
        Raises FileNotFoundError for 404s, otherwise uses raise_for_status.
        """
        if response.status == 404:
            raise FileNotFoundError(url)
        response.raise_for_status()

    async def _cat_file(self, url, start=None, end=None, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)

        if start is not None or end is not None:
            if start == end:
                return b""
            headers = kw.pop("headers", {}).copy()

            headers["Range"] = await self._process_limits(url, start, end)
            kw["headers"] = headers
        session = await self.set_session()
        async with session.get(self.encode_url(url), **kw) as r:
            out = await r.read()
            self._raise_not_found_for_status(r, url)
        return out

    async def _get_file(
        self, rpath, lpath, chunk_size=5 * 2**20, callback=DEFAULT_CALLBACK, **kwargs
    ):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(rpath)
        session = await self.set_session()
        async with session.get(self.encode_url(rpath), **kw) as r:
            try:
                size = int(r.headers["content-length"])
            except (ValueError, KeyError):
                size = None

            callback.set_size(size)
            self._raise_not_found_for_status(r, rpath)
            if isfilelike(lpath):
                outfile = lpath
            else:
                outfile = open(lpath, "wb")  # noqa: ASYNC230

            try:
                chunk = True
                while chunk:
                    chunk = await r.content.read(chunk_size)
                    outfile.write(chunk)
                    callback.relative_update(len(chunk))
            finally:
                if not isfilelike(lpath):
                    outfile.close()

    async def _put_file(
        self,
        lpath,
        rpath,
        chunk_size=5 * 2**20,
        callback=DEFAULT_CALLBACK,
        method="post",
        mode="overwrite",
        **kwargs,
    ):
        if mode != "overwrite":
            raise NotImplementedError("Exclusive write")

        async def gen_chunks():
            # Support passing arbitrary file-like objects
            # and use them instead of streams.
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False  # might not support seeking
            else:
                context = open(lpath, "rb")  # noqa: ASYNC230
                use_seek = True

            with context as f:
                if use_seek:
                    callback.set_size(f.seek(0, 2))
                    f.seek(0)
                else:
                    callback.set_size(getattr(f, "size", None))

                chunk = f.read(chunk_size)
                while chunk:
                    yield chunk
                    callback.relative_update(len(chunk))
                    chunk = f.read(chunk_size)

        kw = self.kwargs.copy()
        kw.update(kwargs)
        session = await self.set_session()

        method = method.lower()
        if method not in ("post", "put"):
            raise ValueError(
                f"method has to be either 'post' or 'put', not: {method!r}"
            )

        meth = getattr(session, method)
        async with meth(self.encode_url(rpath), data=gen_chunks(), **kw) as resp:
            self._raise_not_found_for_status(resp, rpath)

    async def _exists(self, path, strict=False, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        try:
            logger.debug(path)
            session = await self.set_session()
            r = await session.get(self.encode_url(path), **kw)
            async with r:
                if strict:
                    self._raise_not_found_for_status(r, path)
                return r.status < 400
        except FileNotFoundError:
            return False
        except aiohttp.ClientError:
            if strict:
                raise
            return False

    async def _isfile(self, path, **kwargs):
        return await self._exists(path, **kwargs)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=None,  # XXX: This differs from the base class.
        cache_type=None,
        cache_options=None,
        size=None,
        **kwargs,
    ):
        """Make a file-like object

        Parameters
        ----------
        path: str
            Full URL with protocol
        mode: string
            must be "rb"
        block_size: int or None
            Bytes to download in one request; use instance value if None. If
            zero, will return a streaming Requests file-like instance.
        kwargs: key-value
            Any other parameters, passed to requests calls
        """
        if mode != "rb":
            raise NotImplementedError
        block_size = block_size if block_size is not None else self.block_size
        kw = self.kwargs.copy()
        kw["asynchronous"] = self.asynchronous
        kw.update(kwargs)
        info = {}
        size = size or info.update(self.info(path, **kwargs)) or info["size"]
        session = sync(self.loop, self.set_session)
        if block_size and size and info.get("partial", True):
            return HTTPFile(
                self,
                path,
                session=session,
                block_size=block_size,
                mode=mode,
                size=size,
                cache_type=cache_type or self.cache_type,
                cache_options=cache_options or self.cache_options,
                loop=self.loop,
                **kw,
            )
        else:
            return HTTPStreamFile(
                self,
                path,
                mode=mode,
                loop=self.loop,
                session=session,
                **kw,
            )

    async def open_async(self, path, mode="rb", size=None, **kwargs):
        session = await self.set_session()
        if size is None:
            try:
                size = (await self._info(path, **kwargs))["size"]
            except FileNotFoundError:
                pass
        return AsyncStreamFile(
            self,
            path,
            loop=self.loop,
            session=session,
            size=size,
            **kwargs,
        )

    def ukey(self, url):
        """Unique identifier; assume HTTP files are static, unchanging"""
        return tokenize(url, self.kwargs, self.protocol)

    async def _info(self, url, **kwargs):
        """Get info of URL

        Tries to access location via HEAD, and then GET methods, but does
        not fetch the data.

        It is possible that the server does not supply any size information, in
        which case size will be given as None (and certain operations on the
        corresponding file will not work).
        """
        info = {}
        session = await self.set_session()

        for policy in ["head", "get"]:
            try:
                info.update(
                    await _file_info(
                        self.encode_url(url),
                        size_policy=policy,
                        session=session,
                        **self.kwargs,
                        **kwargs,
                    )
                )
                if info.get("size") is not None:
                    break
            except Exception as exc:
                if policy == "get":
                    # If get failed, then raise a FileNotFoundError
                    raise FileNotFoundError(url) from exc
                logger.debug("", exc_info=exc)

        return {"name": url, "size": None, **info, "type": "file"}

    async def _glob(self, path, maxdepth=None, **kwargs):
        """
        Find files by glob-matching.

        This implementation is idntical to the one in AbstractFileSystem,
        but "?" is not considered as a character for globbing, because it is
        so common in URLs, often identifying the "query" part.
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")
        import re

        ends_with_slash = path.endswith("/")  # _strip_protocol strips trailing slash
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_slash or path.endswith(("/**", "/*"))
        idx_star = path.find("*") if path.find("*") >= 0 else len(path)
        idx_brace = path.find("[") if path.find("[") >= 0 else len(path)

        min_idx = min(idx_star, idx_brace)

        detail = kwargs.pop("detail", False)

        if not has_magic(path):
            if await self._exists(path, **kwargs):
                if not detail:
                    return [path]
                else:
                    return {path: await self._info(path, **kwargs)}
            else:
                if not detail:
                    return []  # glob of non-existent returns empty
                else:
                    return {}
        elif "/" in path[:min_idx]:
            min_idx = path[:min_idx].rindex("/")
            root = path[: min_idx + 1]
            depth = path[min_idx + 1 :].count("/") + 1
        else:
            root = ""
            depth = path[min_idx + 1 :].count("/") + 1

        if "**" in path:
            if maxdepth is not None:
                idx_double_stars = path.find("**")
                depth_double_stars = path[idx_double_stars:].count("/") + 1
                depth = depth - depth_double_stars + maxdepth
            else:
                depth = None

        allpaths = await self._find(
            root, maxdepth=depth, withdirs=True, detail=True, **kwargs
        )

        pattern = glob_translate(path + ("/" if ends_with_slash else ""))
        pattern = re.compile(pattern)

        out = {
            (
                p.rstrip("/")
                if not append_slash_to_dirname
                and info["type"] == "directory"
                and p.endswith("/")
                else p
            ): info
            for p, info in sorted(allpaths.items())
            if pattern.match(p.rstrip("/"))
        }

        if detail:
            return out
        else:
            return list(out)

    async def _isdir(self, path):
        # override, since all URLs are (also) files
        try:
            return bool(await self._ls(path))
        except (FileNotFoundError, ValueError):
            return False

    async def _pipe_file(self, path, value, mode="overwrite", **kwargs):
        """
        Write bytes to a remote file over HTTP.

        Parameters
        ----------
        path : str
            Target URL where the data should be written
        value : bytes
            Data to be written
        mode : str
            How to write to the file - 'overwrite' or 'append'
        **kwargs : dict
            Additional parameters to pass to the HTTP request
        """
        url = self._strip_protocol(path)
        headers = kwargs.pop("headers", {})
        headers["Content-Length"] = str(len(value))

        session = await self.set_session()

        async with session.put(url, data=value, headers=headers, **kwargs) as r:
            r.raise_for_status()


class HTTPFile(AbstractBufferedFile):
    """
    A file-like object pointing to a remote HTTP(S) resource

    Supports only reading, with read-ahead of a predetermined block-size.

    In the case that the server does not supply the filesize, only reading of
    the complete file in one go is supported.

    Parameters
    ----------
    url: str
        Full URL of the remote resource, including the protocol
    session: aiohttp.ClientSession or None
        All calls will be made within this session, to avoid restarting
        connections where the server allows this
    block_size: int or None
        The amount of read-ahead to do, in bytes. Default is 5MB, or the value
        configured for the FileSystem creating this file
    size: None or int
        If given, this is the size of the file in bytes, and we don't attempt
        to call the server to find the value.
    kwargs: all other key-values are passed to requests calls.
    """

    def __init__(
        self,
        fs,
        url,
        session=None,
        block_size=None,
        mode="rb",
        cache_type="bytes",
        cache_options=None,
        size=None,
        loop=None,
        asynchronous=False,
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError("File mode not supported")
        self.asynchronous = asynchronous
        self.loop = loop
        self.url = url
        self.session = session
        self.details = {"name": url, "size": size, "type": "file"}
        super().__init__(
            fs=fs,
            path=url,
            mode=mode,
            block_size=block_size,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )

    def read(self, length=-1):
        """Read bytes from file

        Parameters
        ----------
        length: int
            Read up to this many bytes. If negative, read all content to end of
            file. If the server has not supplied the filesize, attempting to
            read only part of the data will raise a ValueError.
        """
        if (
            (length < 0 and self.loc == 0)  # explicit read all
            # but not when the size is known and fits into a block anyways
            and not (self.size is not None and self.size <= self.blocksize)
        ):
            self._fetch_all()
        if self.size is None:
            if length < 0:
                self._fetch_all()
        else:
            length = min(self.size - self.loc, length)
        return super().read(length)

    async def async_fetch_all(self):
        """Read whole file in one shot, without caching

        This is only called when position is still at zero,
        and read() is called without a byte-count.
        """
        logger.debug(f"Fetch all for {self}")
        if not isinstance(self.cache, AllBytes):
            r = await self.session.get(self.fs.encode_url(self.url), **self.kwargs)
            async with r:
                r.raise_for_status()
                out = await r.read()
                self.cache = AllBytes(
                    size=len(out), fetcher=None, blocksize=None, data=out
                )
                self.size = len(out)

    _fetch_all = sync_wrapper(async_fetch_all)

    def _parse_content_range(self, headers):
        """Parse the Content-Range header"""
        s = headers.get("Content-Range", "")
        m = re.match(r"bytes (\d+-\d+|\*)/(\d+|\*)", s)
        if not m:
            return None, None, None

        if m[1] == "*":
            start = end = None
        else:
            start, end = [int(x) for x in m[1].split("-")]
        total = None if m[2] == "*" else int(m[2])
        return start, end, total

    async def async_fetch_range(self, start, end):
        """Download a block of data

        The expectation is that the server returns only the requested bytes,
        with HTTP code 206. If this is not the case, we first check the headers,
        and then stream the output - if the data size is bigger than we
        requested, an exception is raised.
        """
        logger.debug(f"Fetch range for {self}: {start}-{end}")
        kwargs = self.kwargs.copy()
        headers = kwargs.pop("headers", {}).copy()
        headers["Range"] = f"bytes={start}-{end - 1}"
        logger.debug(f"{self.url} : {headers['Range']}")
        r = await self.session.get(
            self.fs.encode_url(self.url), headers=headers, **kwargs
        )
        async with r:
            if r.status == 416:
                # range request outside file
                return b""
            r.raise_for_status()

            # If the server has handled the range request, it should reply
            # with status 206 (partial content). But we'll guess that a suitable
            # Content-Range header or a Content-Length no more than the
            # requested range also mean we have got the desired range.
            response_is_range = (
                r.status == 206
                or self._parse_content_range(r.headers)[0] == start
                or int(r.headers.get("Content-Length", end + 1)) <= end - start
            )

            if response_is_range:
                # partial content, as expected
                out = await r.read()
            elif start > 0:
                raise ValueError(
                    "The HTTP server doesn't appear to support range requests. "
                    "Only reading this file from the beginning is supported. "
                    "Open with block_size=0 for a streaming file interface."
                )
            else:
                # Response is not a range, but we want the start of the file,
                # so we can read the required amount anyway.
                cl = 0
                out = []
                while True:
                    chunk = await r.content.read(2**20)
                    # data size unknown, let's read until we have enough
                    if chunk:
                        out.append(chunk)
                        cl += len(chunk)
                        if cl > end - start:
                            break
                    else:
                        break
                out = b"".join(out)[: end - start]
            return out

    _fetch_range = sync_wrapper(async_fetch_range)


magic_check = re.compile("([*[])")


def has_magic(s):
    match = magic_check.search(s)
    return match is not None


class HTTPStreamFile(AbstractBufferedFile):
    def __init__(self, fs, url, mode="rb", loop=None, session=None, **kwargs):
        self.asynchronous = kwargs.pop("asynchronous", False)
        self.url = url
        self.loop = loop
        self.session = session
        if mode != "rb":
            raise ValueError
        self.details = {"name": url, "size": None}
        super().__init__(fs=fs, path=url, mode=mode, cache_type="none", **kwargs)

        async def cor():
            r = await self.session.get(self.fs.encode_url(url), **kwargs).__aenter__()
            self.fs._raise_not_found_for_status(r, url)
            return r

        self.r = sync(self.loop, cor)
        self.loop = fs.loop

    def seek(self, loc, whence=0):
        if loc == 0 and whence == 1:
            return
        if loc == self.loc and whence == 0:
            return
        raise ValueError("Cannot seek streaming HTTP file")

    async def _read(self, num=-1):
        out = await self.r.content.read(num)
        self.loc += len(out)
        return out

    read = sync_wrapper(_read)

    async def _close(self):
        self.r.close()

    def close(self):
        asyncio.run_coroutine_threadsafe(self._close(), self.loop)
        super().close()


class AsyncStreamFile(AbstractAsyncStreamedFile):
    def __init__(
        self, fs, url, mode="rb", loop=None, session=None, size=None, **kwargs
    ):
        self.url = url
        self.session = session
        self.r = None
        if mode != "rb":
            raise ValueError
        self.details = {"name": url, "size": None}
        self.kwargs = kwargs
        super().__init__(fs=fs, path=url, mode=mode, cache_type="none")
        self.size = size

    async def read(self, num=-1):
        if self.r is None:
            r = await self.session.get(
                self.fs.encode_url(self.url), **self.kwargs
            ).__aenter__()
            self.fs._raise_not_found_for_status(r, self.url)
            self.r = r
        out = await self.r.content.read(num)
        self.loc += len(out)
        return out

    async def close(self):
        if self.r is not None:
            self.r.close()
            self.r = None
        await super().close()


async def get_range(session, url, start, end, file=None, **kwargs):
    # explicit get a range when we know it must be safe
    kwargs = kwargs.copy()
    headers = kwargs.pop("headers", {}).copy()
    headers["Range"] = f"bytes={start}-{end - 1}"
    r = await session.get(url, headers=headers, **kwargs)
    r.raise_for_status()
    async with r:
        out = await r.read()
    if file:
        with open(file, "r+b") as f:  # noqa: ASYNC230
            f.seek(start)
            f.write(out)
    else:
        return out


async def _file_info(url, session, size_policy="head", **kwargs):
    """Call HEAD on the server to get details about the file (size/checksum etc.)

    Default operation is to explicitly allow redirects and use encoding
    'identity' (no compression) to get the true size of the target.
    """
    logger.debug("Retrieve file size for %s", url)
    kwargs = kwargs.copy()
    ar = kwargs.pop("allow_redirects", True)
    head = kwargs.get("headers", {}).copy()
    head["Accept-Encoding"] = "identity"
    kwargs["headers"] = head

    info = {}
    if size_policy == "head":
        r = await session.head(url, allow_redirects=ar, **kwargs)
    elif size_policy == "get":
        r = await session.get(url, allow_redirects=ar, **kwargs)
    else:
        raise TypeError(f'size_policy must be "head" or "get", got {size_policy}')
    async with r:
        r.raise_for_status()

        if "Content-Length" in r.headers:
            # Some servers may choose to ignore Accept-Encoding and return
            # compressed content, in which case the returned size is unreliable.
            if "Content-Encoding" not in r.headers or r.headers["Content-Encoding"] in [
                "identity",
                "",
            ]:
                info["size"] = int(r.headers["Content-Length"])
        elif "Content-Range" in r.headers:
            info["size"] = int(r.headers["Content-Range"].split("/")[1])

        if "Content-Type" in r.headers:
            info["mimetype"] = r.headers["Content-Type"].partition(";")[0]

        if r.headers.get("Accept-Ranges") == "none":
            # Some servers may explicitly discourage partial content requests, but
            # the lack of "Accept-Ranges" does not always indicate they would fail
            info["partial"] = False

        info["url"] = str(r.url)

        for checksum_field in ["ETag", "Content-MD5", "Digest", "Last-Modified"]:
            if r.headers.get(checksum_field):
                info[checksum_field] = r.headers[checksum_field]

    return info


async def _file_size(url, session=None, *args, **kwargs):
    if session is None:
        session = await get_client()
    info = await _file_info(url, session=session, *args, **kwargs)
    return info.get("size")


file_size = sync_wrapper(_file_size)
