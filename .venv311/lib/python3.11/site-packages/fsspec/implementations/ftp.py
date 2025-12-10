import os
import uuid
from ftplib import FTP, FTP_TLS, Error, error_perm
from typing import Any

from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, isfilelike


class FTPFileSystem(AbstractFileSystem):
    """A filesystem over classic FTP"""

    root_marker = "/"
    cachable = False
    protocol = "ftp"

    def __init__(
        self,
        host,
        port=21,
        username=None,
        password=None,
        acct=None,
        block_size=None,
        tempdir=None,
        timeout=30,
        encoding="utf-8",
        tls=False,
        **kwargs,
    ):
        """
        You can use _get_kwargs_from_urls to get some kwargs from
        a reasonable FTP url.

        Authentication will be anonymous if username/password are not
        given.

        Parameters
        ----------
        host: str
            The remote server name/ip to connect to
        port: int
            Port to connect with
        username: str or None
            If authenticating, the user's identifier
        password: str of None
            User's password on the server, if using
        acct: str or None
            Some servers also need an "account" string for auth
        block_size: int or None
            If given, the read-ahead or write buffer size.
        tempdir: str
            Directory on remote to put temporary files when in a transaction
        timeout: int
            Timeout of the ftp connection in seconds
        encoding: str
            Encoding to use for directories and filenames in FTP connection
        tls: bool
            Use FTP-TLS, by default False
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.tempdir = tempdir or "/tmp"
        self.cred = username or "", password or "", acct or ""
        self.timeout = timeout
        self.encoding = encoding
        if block_size is not None:
            self.blocksize = block_size
        else:
            self.blocksize = 2**16
        self.tls = tls
        self._connect()
        if self.tls:
            self.ftp.prot_p()

    def _connect(self):
        if self.tls:
            ftp_cls = FTP_TLS
        else:
            ftp_cls = FTP
        self.ftp = ftp_cls(timeout=self.timeout, encoding=self.encoding)
        self.ftp.connect(self.host, self.port)
        self.ftp.login(*self.cred)

    @classmethod
    def _strip_protocol(cls, path):
        return "/" + infer_storage_options(path)["path"].lstrip("/").rstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(urlpath):
        out = infer_storage_options(urlpath)
        out.pop("path", None)
        out.pop("protocol", None)
        return out

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        out = []
        if path not in self.dircache:
            try:
                try:
                    out = [
                        (fn, details)
                        for (fn, details) in self.ftp.mlsd(path)
                        if fn not in [".", ".."]
                        and details["type"] not in ["pdir", "cdir"]
                    ]
                except error_perm:
                    out = _mlsd2(self.ftp, path)  # Not platform independent
                for fn, details in out:
                    details["name"] = "/".join(
                        ["" if path == "/" else path, fn.lstrip("/")]
                    )
                    if details["type"] == "file":
                        details["size"] = int(details["size"])
                    else:
                        details["size"] = 0
                    if details["type"] == "dir":
                        details["type"] = "directory"
                self.dircache[path] = out
            except Error:
                try:
                    info = self.info(path)
                    if info["type"] == "file":
                        out = [(path, info)]
                except (Error, IndexError) as exc:
                    raise FileNotFoundError(path) from exc
        files = self.dircache.get(path, out)
        if not detail:
            return sorted([fn for fn, details in files])
        return [details for fn, details in files]

    def info(self, path, **kwargs):
        # implement with direct method
        path = self._strip_protocol(path)
        if path == "/":
            # special case, since this dir has no real entry
            return {"name": "/", "size": 0, "type": "directory"}
        files = self.ls(self._parent(path).lstrip("/"), True)
        try:
            out = next(f for f in files if f["name"] == path)
        except StopIteration as exc:
            raise FileNotFoundError(path) from exc
        return out

    def get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            if not os.path.exists(lpath):
                os.mkdir(lpath)
            return
        if isfilelike(lpath):
            outfile = lpath
        else:
            outfile = open(lpath, "wb")

        def cb(x):
            outfile.write(x)

        self.ftp.retrbinary(
            f"RETR {rpath}",
            blocksize=self.blocksize,
            callback=cb,
        )
        if not isfilelike(lpath):
            outfile.close()

    def cat_file(self, path, start=None, end=None, **kwargs):
        if end is not None:
            return super().cat_file(path, start, end, **kwargs)
        out = []

        def cb(x):
            out.append(x)

        try:
            self.ftp.retrbinary(
                f"RETR {path}",
                blocksize=self.blocksize,
                rest=start,
                callback=cb,
            )
        except (Error, error_perm) as orig_exc:
            raise FileNotFoundError(path) from orig_exc
        return b"".join(out)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        autocommit=True,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        block_size = block_size or self.blocksize
        return FTPFile(
            self,
            path,
            mode=mode,
            block_size=block_size,
            tempdir=self.tempdir,
            autocommit=autocommit,
            cache_options=cache_options,
        )

    def _rm(self, path):
        path = self._strip_protocol(path)
        self.ftp.delete(path)
        self.invalidate_cache(self._parent(path))

    def rm(self, path, recursive=False, maxdepth=None):
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(paths):
            if self.isfile(p):
                self.rm_file(p)
            else:
                self.rmdir(p)

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        parent = self._parent(path)
        if parent != self.root_marker and not self.exists(parent) and create_parents:
            self.mkdir(parent, create_parents=create_parents)

        self.ftp.mkd(path)
        self.invalidate_cache(self._parent(path))

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        path = self._strip_protocol(path)
        if self.exists(path):
            # NB: "/" does not "exist" as it has no directory entry
            if not exist_ok:
                raise FileExistsError(f"{path} exists without `exist_ok`")
            # exists_ok=True -> no-op
        else:
            self.mkdir(path, create_parents=True)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        self.ftp.rmd(path)
        self.invalidate_cache(self._parent(path))

    def mv(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        self.ftp.rename(path1, path2)
        self.invalidate_cache(self._parent(path1))
        self.invalidate_cache(self._parent(path2))

    def __del__(self):
        self.ftp.close()

    def invalidate_cache(self, path=None):
        if path is None:
            self.dircache.clear()
        else:
            self.dircache.pop(path, None)
        super().invalidate_cache(path)


class TransferDone(Exception):
    """Internal exception to break out of transfer"""

    pass


class FTPFile(AbstractBufferedFile):
    """Interact with a remote FTP file with read/write buffering"""

    def __init__(
        self,
        fs,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        **kwargs,
    ):
        super().__init__(
            fs,
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )
        if not autocommit:
            self.target = self.path
            self.path = "/".join([kwargs["tempdir"], str(uuid.uuid4())])

    def commit(self):
        self.fs.mv(self.path, self.target)

    def discard(self):
        self.fs.rm(self.path)

    def _fetch_range(self, start, end):
        """Get bytes between given byte limits

        Implemented by raising an exception in the fetch callback when the
        number of bytes received reaches the requested amount.

        Will fail if the server does not respect the REST command on
        retrieve requests.
        """
        out = []
        total = [0]

        def callback(x):
            total[0] += len(x)
            if total[0] > end - start:
                out.append(x[: (end - start) - total[0]])
                if end < self.size:
                    raise TransferDone
            else:
                out.append(x)

            if total[0] == end - start and end < self.size:
                raise TransferDone

        try:
            self.fs.ftp.retrbinary(
                f"RETR {self.path}",
                blocksize=self.blocksize,
                rest=start,
                callback=callback,
            )
        except TransferDone:
            try:
                # stop transfer, we got enough bytes for this block
                self.fs.ftp.abort()
                self.fs.ftp.getmultiline()
            except Error:
                self.fs._connect()

        return b"".join(out)

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        self.fs.ftp.storbinary(
            f"STOR {self.path}", self.buffer, blocksize=self.blocksize, rest=self.offset
        )
        return True


def _mlsd2(ftp, path="."):
    """
    Fall back to using `dir` instead of `mlsd` if not supported.

    This parses a Linux style `ls -l` response to `dir`, but the response may
    be platform dependent.

    Parameters
    ----------
    ftp: ftplib.FTP
    path: str
        Expects to be given path, but defaults to ".".
    """
    lines = []
    minfo = []
    ftp.dir(path, lines.append)
    for line in lines:
        split_line = line.split()
        if len(split_line) < 9:
            continue
        this = (
            split_line[-1],
            {
                "modify": " ".join(split_line[5:8]),
                "unix.owner": split_line[2],
                "unix.group": split_line[3],
                "unix.mode": split_line[0],
                "size": split_line[4],
            },
        )
        if this[1]["unix.mode"][0] == "d":
            this[1]["type"] = "dir"
        else:
            this[1]["type"] = "file"
        minfo.append(this)
    return minfo
