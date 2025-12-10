"""This file is largely copied from http.py"""

import io
import logging
import re
import urllib.error
import urllib.parse
from copy import copy
from json import dumps, loads
from urllib.parse import urlparse

try:
    import yarl
except (ImportError, ModuleNotFoundError, OSError):
    yarl = False

from fsspec.callbacks import _DEFAULT_CALLBACK
from fsspec.registry import register_implementation
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE, isfilelike, nullcontext, tokenize

from ..caching import AllBytes

# https://stackoverflow.com/a/15926317/3821154
ex = re.compile(r"""<(a|A)\s+(?:[^>]*?\s+)?(href|HREF)=["'](?P<url>[^"']+)""")
ex2 = re.compile(r"""(?P<url>http[s]?://[-a-zA-Z0-9@:%_+.~#?&/=]+)""")
logger = logging.getLogger("fsspec.http")


class JsHttpException(urllib.error.HTTPError): ...


class StreamIO(io.BytesIO):
    # fake class, so you can set attributes on it
    # will eventually actually stream
    ...


class ResponseProxy:
    """Looks like a requests response"""

    def __init__(self, req, stream=False):
        self.request = req
        self.stream = stream
        self._data = None
        self._headers = None

    @property
    def raw(self):
        if self._data is None:
            b = self.request.response.to_bytes()
            if self.stream:
                self._data = StreamIO(b)
            else:
                self._data = b
        return self._data

    def close(self):
        if hasattr(self, "_data"):
            del self._data

    @property
    def headers(self):
        if self._headers is None:
            self._headers = dict(
                [
                    _.split(": ")
                    for _ in self.request.getAllResponseHeaders().strip().split("\r\n")
                ]
            )
        return self._headers

    @property
    def status_code(self):
        return int(self.request.status)

    def raise_for_status(self):
        if not self.ok:
            raise JsHttpException(
                self.url, self.status_code, self.reason, self.headers, None
            )

    def iter_content(self, chunksize, *_, **__):
        while True:
            out = self.raw.read(chunksize)
            if out:
                yield out
            else:
                break

    @property
    def reason(self):
        return self.request.statusText

    @property
    def ok(self):
        return self.status_code < 400

    @property
    def url(self):
        return self.request.response.responseURL

    @property
    def text(self):
        # TODO: encoding from headers
        return self.content.decode()

    @property
    def content(self):
        self.stream = False
        return self.raw

    def json(self):
        return loads(self.text)


class RequestsSessionShim:
    def __init__(self):
        self.headers = {}

    def request(
        self,
        method,
        url,
        params=None,
        data=None,
        headers=None,
        cookies=None,
        files=None,
        auth=None,
        timeout=None,
        allow_redirects=None,
        proxies=None,
        hooks=None,
        stream=None,
        verify=None,
        cert=None,
        json=None,
    ):
        from js import Blob, XMLHttpRequest

        logger.debug("JS request: %s %s", method, url)

        if cert or verify or proxies or files or cookies or hooks:
            raise NotImplementedError
        if data and json:
            raise ValueError("Use json= or data=, not both")
        req = XMLHttpRequest.new()
        extra = auth if auth else ()
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req.open(method, url, False, *extra)
        if timeout:
            req.timeout = timeout
        if headers:
            for k, v in headers.items():
                req.setRequestHeader(k, v)

        req.setRequestHeader("Accept", "application/octet-stream")
        req.responseType = "arraybuffer"
        if json:
            blob = Blob.new([dumps(data)], {type: "application/json"})
            req.send(blob)
        elif data:
            if isinstance(data, io.IOBase):
                data = data.read()
            blob = Blob.new([data], {type: "application/octet-stream"})
            req.send(blob)
        else:
            req.send(None)
        return ResponseProxy(req, stream=stream)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def head(self, url, **kwargs):
        return self.request("HEAD", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST}", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


class HTTPFileSystem(AbstractFileSystem):
    """
    Simple File-System for fetching data via HTTP(S)

    This is the BLOCKING version of the normal HTTPFileSystem. It uses
    requests in normal python and the JS runtime in pyodide.

    ***This implementation is extremely experimental, do not use unless
    you are testing pyodide/pyscript integration***
    """

    protocol = ("http", "https", "sync-http", "sync-https")
    sep = "/"

    def __init__(
        self,
        simple_links=True,
        block_size=None,
        same_scheme=True,
        cache_type="readahead",
        cache_options=None,
        client_kwargs=None,
        encoded=False,
        **storage_options,
    ):
        """

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
        storage_options: key-value
            Any other parameters passed on to requests
        cache_type, cache_options: defaults used in open
        """
        super().__init__(self, **storage_options)
        self.block_size = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
        self.simple_links = simple_links
        self.same_schema = same_scheme
        self.cache_type = cache_type
        self.cache_options = cache_options
        self.client_kwargs = client_kwargs or {}
        self.encoded = encoded
        self.kwargs = storage_options

        try:
            import js  # noqa: F401

            logger.debug("Starting JS session")
            self.session = RequestsSessionShim()
            self.js = True
        except Exception as e:
            import requests

            logger.debug("Starting cpython session because of: %s", e)
            self.session = requests.Session(**(client_kwargs or {}))
            self.js = False

        request_options = copy(storage_options)
        self.use_listings_cache = request_options.pop("use_listings_cache", False)
        request_options.pop("listings_expiry_time", None)
        request_options.pop("max_paths", None)
        request_options.pop("skip_instance_cache", None)
        self.kwargs = request_options

    @property
    def fsid(self):
        return "sync-http"

    def encode_url(self, url):
        if yarl:
            return yarl.URL(url, encoded=self.encoded)
        return url

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        """For HTTP, we always want to keep the full URL"""
        path = path.replace("sync-http://", "http://").replace(
            "sync-https://", "https://"
        )
        return path

    @classmethod
    def _parent(cls, path):
        # override, since _strip_protocol is different for URLs
        par = super()._parent(path)
        if len(par) > 7:  # "http://..."
            return par
        return ""

    def _ls_real(self, url, detail=True, **kwargs):
        # ignoring URL-encoded arguments
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)
        r = self.session.get(self.encode_url(url), **self.kwargs)
        self._raise_not_found_for_status(r, url)
        text = r.text
        if self.simple_links:
            links = ex2.findall(text) + [u[2] for u in ex.findall(text)]
        else:
            links = [u[2] for u in ex.findall(text)]
        out = set()
        parts = urlparse(url)
        for l in links:
            if isinstance(l, tuple):
                l = l[1]
            if l.startswith("/") and len(l) > 1:
                # absolute URL on this server
                l = parts.scheme + "://" + parts.netloc + l
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
            out = self._ls_real(url.rstrip("/"), detail=False)
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

    def ls(self, url, detail=True, **kwargs):
        if self.use_listings_cache and url in self.dircache:
            out = self.dircache[url]
        else:
            out = self._ls_real(url, detail=detail, **kwargs)
            self.dircache[url] = out
        return out

    def _raise_not_found_for_status(self, response, url):
        """
        Raises FileNotFoundError for 404s, otherwise uses raise_for_status.
        """
        if response.status_code == 404:
            raise FileNotFoundError(url)
        response.raise_for_status()

    def cat_file(self, url, start=None, end=None, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(url)

        if start is not None or end is not None:
            if start == end:
                return b""
            headers = kw.pop("headers", {}).copy()

            headers["Range"] = self._process_limits(url, start, end)
            kw["headers"] = headers
        r = self.session.get(self.encode_url(url), **kw)
        self._raise_not_found_for_status(r, url)
        return r.content

    def get_file(
        self, rpath, lpath, chunk_size=5 * 2**20, callback=_DEFAULT_CALLBACK, **kwargs
    ):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        logger.debug(rpath)
        r = self.session.get(self.encode_url(rpath), **kw)
        try:
            size = int(
                r.headers.get("content-length", None)
                or r.headers.get("Content-Length", None)
            )
        except (ValueError, KeyError, TypeError):
            size = None

        callback.set_size(size)
        self._raise_not_found_for_status(r, rpath)
        if not isfilelike(lpath):
            lpath = open(lpath, "wb")
        for chunk in r.iter_content(chunk_size, decode_unicode=False):
            lpath.write(chunk)
            callback.relative_update(len(chunk))

    def put_file(
        self,
        lpath,
        rpath,
        chunk_size=5 * 2**20,
        callback=_DEFAULT_CALLBACK,
        method="post",
        **kwargs,
    ):
        def gen_chunks():
            # Support passing arbitrary file-like objects
            # and use them instead of streams.
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False  # might not support seeking
            else:
                context = open(lpath, "rb")
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

        method = method.lower()
        if method not in ("post", "put"):
            raise ValueError(
                f"method has to be either 'post' or 'put', not: {method!r}"
            )

        meth = getattr(self.session, method)
        resp = meth(rpath, data=gen_chunks(), **kw)
        self._raise_not_found_for_status(resp, rpath)

    def _process_limits(self, url, start, end):
        """Helper for "Range"-based _cat_file"""
        size = None
        suff = False
        if start is not None and start < 0:
            # if start is negative and end None, end is the "suffix length"
            if end is None:
                end = -start
                start = ""
                suff = True
            else:
                size = size or self.info(url)["size"]
                start = size + start
        elif start is None:
            start = 0
        if not suff:
            if end is not None and end < 0:
                if start is not None:
                    size = size or self.info(url)["size"]
                    end = size + end
            elif end is None:
                end = ""
            if isinstance(end, int):
                end -= 1  # bytes range is inclusive
        return f"bytes={start}-{end}"

    def exists(self, path, strict=False, **kwargs):
        kw = self.kwargs.copy()
        kw.update(kwargs)
        try:
            logger.debug(path)
            r = self.session.get(self.encode_url(path), **kw)
            if strict:
                self._raise_not_found_for_status(r, path)
            return r.status_code < 400
        except FileNotFoundError:
            return False
        except Exception:
            if strict:
                raise
            return False

    def isfile(self, path, **kwargs):
        return self.exists(path, **kwargs)

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
        kw.update(kwargs)
        size = size or self.info(path, **kwargs)["size"]
        if block_size and size:
            return HTTPFile(
                self,
                path,
                session=self.session,
                block_size=block_size,
                mode=mode,
                size=size,
                cache_type=cache_type or self.cache_type,
                cache_options=cache_options or self.cache_options,
                **kw,
            )
        else:
            return HTTPStreamFile(
                self,
                path,
                mode=mode,
                session=self.session,
                **kw,
            )

    def ukey(self, url):
        """Unique identifier; assume HTTP files are static, unchanging"""
        return tokenize(url, self.kwargs, self.protocol)

    def info(self, url, **kwargs):
        """Get info of URL

        Tries to access location via HEAD, and then GET methods, but does
        not fetch the data.

        It is possible that the server does not supply any size information, in
        which case size will be given as None (and certain operations on the
        corresponding file will not work).
        """
        info = {}
        for policy in ["head", "get"]:
            try:
                info.update(
                    _file_info(
                        self.encode_url(url),
                        size_policy=policy,
                        session=self.session,
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
                logger.debug(str(exc))

        return {"name": url, "size": None, **info, "type": "file"}

    def glob(self, path, maxdepth=None, **kwargs):
        """
        Find files by glob-matching.

        This implementation is idntical to the one in AbstractFileSystem,
        but "?" is not considered as a character for globbing, because it is
        so common in URLs, often identifying the "query" part.
        """
        import re

        ends = path.endswith("/")
        path = self._strip_protocol(path)
        indstar = path.find("*") if path.find("*") >= 0 else len(path)
        indbrace = path.find("[") if path.find("[") >= 0 else len(path)

        ind = min(indstar, indbrace)

        detail = kwargs.pop("detail", False)

        if not has_magic(path):
            root = path
            depth = 1
            if ends:
                path += "/*"
            elif self.exists(path):
                if not detail:
                    return [path]
                else:
                    return {path: self.info(path)}
            else:
                if not detail:
                    return []  # glob of non-existent returns empty
                else:
                    return {}
        elif "/" in path[:ind]:
            ind2 = path[:ind].rindex("/")
            root = path[: ind2 + 1]
            depth = None if "**" in path else path[ind2 + 1 :].count("/") + 1
        else:
            root = ""
            depth = None if "**" in path else path[ind + 1 :].count("/") + 1

        allpaths = self.find(
            root, maxdepth=maxdepth or depth, withdirs=True, detail=True, **kwargs
        )
        # Escape characters special to python regex, leaving our supported
        # special characters in place.
        # See https://www.gnu.org/software/bash/manual/html_node/Pattern-Matching.html
        # for shell globbing details.
        pattern = (
            "^"
            + (
                path.replace("\\", r"\\")
                .replace(".", r"\.")
                .replace("+", r"\+")
                .replace("//", "/")
                .replace("(", r"\(")
                .replace(")", r"\)")
                .replace("|", r"\|")
                .replace("^", r"\^")
                .replace("$", r"\$")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .rstrip("/")
            )
            + "$"
        )
        pattern = re.sub("[*]{2}", "=PLACEHOLDER=", pattern)
        pattern = re.sub("[*]", "[^/]*", pattern)
        pattern = re.compile(pattern.replace("=PLACEHOLDER=", ".*"))
        out = {
            p: allpaths[p]
            for p in sorted(allpaths)
            if pattern.match(p.replace("//", "/").rstrip("/"))
        }
        if detail:
            return out
        else:
            return list(out)

    def isdir(self, path):
        # override, since all URLs are (also) files
        try:
            return bool(self.ls(path))
        except (FileNotFoundError, ValueError):
            return False


class HTTPFile(AbstractBufferedFile):
    """
    A file-like object pointing to a remove HTTP(S) resource

    Supports only reading, with read-ahead of a predermined block-size.

    In the case that the server does not supply the filesize, only reading of
    the complete file in one go is supported.

    Parameters
    ----------
    url: str
        Full URL of the remote resource, including the protocol
    session: requests.Session or None
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
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError("File mode not supported")
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

    def _fetch_all(self):
        """Read whole file in one shot, without caching

        This is only called when position is still at zero,
        and read() is called without a byte-count.
        """
        logger.debug(f"Fetch all for {self}")
        if not isinstance(self.cache, AllBytes):
            r = self.session.get(self.fs.encode_url(self.url), **self.kwargs)
            r.raise_for_status()
            out = r.content
            self.cache = AllBytes(size=len(out), fetcher=None, blocksize=None, data=out)
            self.size = len(out)

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

    def _fetch_range(self, start, end):
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
        logger.debug("%s : %s", self.url, headers["Range"])
        r = self.session.get(self.fs.encode_url(self.url), headers=headers, **kwargs)
        if r.status_code == 416:
            # range request outside file
            return b""
        r.raise_for_status()

        # If the server has handled the range request, it should reply
        # with status 206 (partial content). But we'll guess that a suitable
        # Content-Range header or a Content-Length no more than the
        # requested range also mean we have got the desired range.
        cl = r.headers.get("Content-Length", r.headers.get("content-length", end + 1))
        response_is_range = (
            r.status_code == 206
            or self._parse_content_range(r.headers)[0] == start
            or int(cl) <= end - start
        )

        if response_is_range:
            # partial content, as expected
            out = r.content
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
            for chunk in r.iter_content(2**20, False):
                out.append(chunk)
                cl += len(chunk)
            out = b"".join(out)[: end - start]
        return out


magic_check = re.compile("([*[])")


def has_magic(s):
    match = magic_check.search(s)
    return match is not None


class HTTPStreamFile(AbstractBufferedFile):
    def __init__(self, fs, url, mode="rb", session=None, **kwargs):
        self.url = url
        self.session = session
        if mode != "rb":
            raise ValueError
        self.details = {"name": url, "size": None}
        super().__init__(fs=fs, path=url, mode=mode, cache_type="readahead", **kwargs)

        r = self.session.get(self.fs.encode_url(url), stream=True, **kwargs)
        self.fs._raise_not_found_for_status(r, url)
        self.it = r.iter_content(1024, False)
        self.leftover = b""

        self.r = r

    def seek(self, *args, **kwargs):
        raise ValueError("Cannot seek streaming HTTP file")

    def read(self, num=-1):
        bufs = [self.leftover]
        leng = len(self.leftover)
        while leng < num or num < 0:
            try:
                out = self.it.__next__()
            except StopIteration:
                break
            if out:
                bufs.append(out)
            else:
                break
            leng += len(out)
        out = b"".join(bufs)
        if num >= 0:
            self.leftover = out[num:]
            out = out[:num]
        else:
            self.leftover = b""
        self.loc += len(out)
        return out

    def close(self):
        self.r.close()
        self.closed = True


def get_range(session, url, start, end, **kwargs):
    # explicit get a range when we know it must be safe
    kwargs = kwargs.copy()
    headers = kwargs.pop("headers", {}).copy()
    headers["Range"] = f"bytes={start}-{end - 1}"
    r = session.get(url, headers=headers, **kwargs)
    r.raise_for_status()
    return r.content


def _file_info(url, session, size_policy="head", **kwargs):
    """Call HEAD on the server to get details about the file (size/checksum etc.)

    Default operation is to explicitly allow redirects and use encoding
    'identity' (no compression) to get the true size of the target.
    """
    logger.debug("Retrieve file size for %s", url)
    kwargs = kwargs.copy()
    ar = kwargs.pop("allow_redirects", True)
    head = kwargs.get("headers", {}).copy()
    # TODO: not allowed in JS
    # head["Accept-Encoding"] = "identity"
    kwargs["headers"] = head

    info = {}
    if size_policy == "head":
        r = session.head(url, allow_redirects=ar, **kwargs)
    elif size_policy == "get":
        r = session.get(url, allow_redirects=ar, **kwargs)
    else:
        raise TypeError(f'size_policy must be "head" or "get", got {size_policy}')
    r.raise_for_status()

    # TODO:
    #  recognise lack of 'Accept-Ranges',
    #                 or 'Accept-Ranges': 'none' (not 'bytes')
    #  to mean streaming only, no random access => return None
    if "Content-Length" in r.headers:
        info["size"] = int(r.headers["Content-Length"])
    elif "Content-Range" in r.headers:
        info["size"] = int(r.headers["Content-Range"].split("/")[1])
    elif "content-length" in r.headers:
        info["size"] = int(r.headers["content-length"])
    elif "content-range" in r.headers:
        info["size"] = int(r.headers["content-range"].split("/")[1])

    for checksum_field in ["ETag", "Content-MD5", "Digest"]:
        if r.headers.get(checksum_field):
            info[checksum_field] = r.headers[checksum_field]

    return info


# importing this is enough to register it
def register():
    register_implementation("http", HTTPFileSystem, clobber=True)
    register_implementation("https", HTTPFileSystem, clobber=True)
    register_implementation("sync-http", HTTPFileSystem, clobber=True)
    register_implementation("sync-https", HTTPFileSystem, clobber=True)


register()


def unregister():
    from fsspec.implementations.http import HTTPFileSystem

    register_implementation("http", HTTPFileSystem, clobber=True)
    register_implementation("https", HTTPFileSystem, clobber=True)
