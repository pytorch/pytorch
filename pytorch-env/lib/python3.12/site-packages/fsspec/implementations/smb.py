"""
This module contains SMBFileSystem class responsible for handling access to
Windows Samba network shares by using package smbprotocol
"""

import datetime
import re
import uuid
from stat import S_ISDIR, S_ISLNK

import smbclient
import smbprotocol.exceptions

from .. import AbstractFileSystem
from ..utils import infer_storage_options

# ! pylint: disable=bad-continuation


class SMBFileSystem(AbstractFileSystem):
    """Allow reading and writing to Windows and Samba network shares.

    When using `fsspec.open()` for getting a file-like object the URI
    should be specified as this format:
    ``smb://workgroup;user:password@server:port/share/folder/file.csv``.

    Example::

        >>> import fsspec
        >>> with fsspec.open(
        ...     'smb://myuser:mypassword@myserver.com/' 'share/folder/file.csv'
        ... ) as smbfile:
        ...     df = pd.read_csv(smbfile, sep='|', header=None)

    Note that you need to pass in a valid hostname or IP address for the host
    component of the URL. Do not use the Windows/NetBIOS machine name for the
    host component.

    The first component of the path in the URL points to the name of the shared
    folder. Subsequent path components will point to the directory/folder/file.

    The URL components ``workgroup`` , ``user``, ``password`` and ``port`` may be
    optional.

    .. note::

        For working this source require `smbprotocol`_ to be installed, e.g.::

            $ pip install smbprotocol
            # or
            # pip install smbprotocol[kerberos]

    .. _smbprotocol: https://github.com/jborean93/smbprotocol#requirements

    Note: if using this with the ``open`` or ``open_files``, with full URLs,
    there is no way to tell if a path is relative, so all paths are assumed
    to be absolute.
    """

    protocol = "smb"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        host,
        port=None,
        username=None,
        password=None,
        timeout=60,
        encrypt=None,
        share_access=None,
        register_session_retries=4,
        register_session_retry_wait=1,
        register_session_retry_factor=10,
        auto_mkdir=False,
        **kwargs,
    ):
        """
        You can use _get_kwargs_from_urls to get some kwargs from
        a reasonable SMB url.

        Authentication will be anonymous or integrated if username/password are not
        given.

        Parameters
        ----------
        host: str
            The remote server name/ip to connect to
        port: int or None
            Port to connect with. Usually 445, sometimes 139.
        username: str or None
            Username to connect with. Required if Kerberos auth is not being used.
        password: str or None
            User's password on the server, if using username
        timeout: int
            Connection timeout in seconds
        encrypt: bool
            Whether to force encryption or not, once this has been set to True
            the session cannot be changed back to False.
        share_access: str or None
            Specifies the default access applied to file open operations
            performed with this file system object.
            This affects whether other processes can concurrently open a handle
            to the same file.

            - None (the default): exclusively locks the file until closed.
            - 'r': Allow other handles to be opened with read access.
            - 'w': Allow other handles to be opened with write access.
            - 'd': Allow other handles to be opened with delete access.
        register_session_retries: int
            Number of retries to register a session with the server. Retries are not performed
            for authentication errors, as they are considered as invalid credentials and not network
            issues. If set to negative value, no register attempts will be performed.
        register_session_retry_wait: int
            Time in seconds to wait between each retry. Number must be non-negative.
        register_session_retry_factor: int
            Base factor for the wait time between each retry. The wait time
            is calculated using exponential function. For factor=1 all wait times
            will be equal to `register_session_retry_wait`. For any number of retries,
            the last wait time will be equal to `register_session_retry_wait` and for retries>1
            the first wait time will be equal to `register_session_retry_wait / factor`.
            Number must be equal to or greater than 1. Optimal factor is 10.
        auto_mkdir: bool
            Whether, when opening a file, the directory containing it should
            be created (if it doesn't already exist). This is assumed by pyarrow
            and zarr-python code.
        """
        super().__init__(**kwargs)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.encrypt = encrypt
        self.temppath = kwargs.pop("temppath", "")
        self.share_access = share_access
        self.register_session_retries = register_session_retries
        if register_session_retry_wait < 0:
            raise ValueError(
                "register_session_retry_wait must be a non-negative integer"
            )
        self.register_session_retry_wait = register_session_retry_wait
        if register_session_retry_factor < 1:
            raise ValueError(
                "register_session_retry_factor must be a positive "
                "integer equal to or greater than 1"
            )
        self.register_session_retry_factor = register_session_retry_factor
        self.auto_mkdir = auto_mkdir
        self._connect()

    @property
    def _port(self):
        return 445 if self.port is None else self.port

    def _connect(self):
        import time

        if self.register_session_retries <= -1:
            return

        retried_errors = []

        wait_time = self.register_session_retry_wait
        n_waits = (
            self.register_session_retries - 1
        )  # -1 = No wait time after the last retry
        factor = self.register_session_retry_factor

        # Generate wait times for each retry attempt.
        # Wait times are calculated using exponential function. For factor=1 all wait times
        # will be equal to `wait`. For any number of retries the last wait time will be
        # equal to `wait` and for retries>2 the first wait time will be equal to `wait / factor`.
        wait_times = iter(
            factor ** (n / n_waits - 1) * wait_time for n in range(0, n_waits + 1)
        )

        for attempt in range(self.register_session_retries + 1):
            try:
                smbclient.register_session(
                    self.host,
                    username=self.username,
                    password=self.password,
                    port=self._port,
                    encrypt=self.encrypt,
                    connection_timeout=self.timeout,
                )
                return
            except (
                smbprotocol.exceptions.SMBAuthenticationError,
                smbprotocol.exceptions.LogonFailure,
            ):
                # These exceptions should not be repeated, as they clearly indicate
                # that the credentials are invalid and not a network issue.
                raise
            except ValueError as exc:
                if re.findall(r"\[Errno -\d+]", str(exc)):
                    # This exception is raised by the smbprotocol.transport:Tcp.connect
                    # and originates from socket.gaierror (OSError). These exceptions might
                    # be raised due to network instability. We will retry to connect.
                    retried_errors.append(exc)
                else:
                    # All another ValueError exceptions should be raised, as they are not
                    # related to network issues.
                    raise
            except Exception as exc:
                # Save the exception and retry to connect. This except might be dropped
                # in the future, once all exceptions suited for retry are identified.
                retried_errors.append(exc)

            if attempt < self.register_session_retries:
                time.sleep(next(wait_times))

        # Raise last exception to inform user about the connection issues.
        # Note: Should we use ExceptionGroup to raise all exceptions?
        raise retried_errors[-1]

    @classmethod
    def _strip_protocol(cls, path):
        return infer_storage_options(path)["path"]

    @staticmethod
    def _get_kwargs_from_urls(path):
        # smb://workgroup;user:password@host:port/share/folder/file.csv
        out = infer_storage_options(path)
        out.pop("path", None)
        out.pop("protocol", None)
        return out

    def mkdir(self, path, create_parents=True, **kwargs):
        wpath = _as_unc_path(self.host, path)
        if create_parents:
            smbclient.makedirs(wpath, exist_ok=False, port=self._port, **kwargs)
        else:
            smbclient.mkdir(wpath, port=self._port, **kwargs)

    def makedirs(self, path, exist_ok=False):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            smbclient.makedirs(wpath, exist_ok=exist_ok, port=self._port)

    def rmdir(self, path):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            smbclient.rmdir(wpath, port=self._port)

    def info(self, path, **kwargs):
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port, **kwargs)
        if S_ISDIR(stats.st_mode):
            stype = "directory"
        elif S_ISLNK(stats.st_mode):
            stype = "link"
        else:
            stype = "file"
        res = {
            "name": path + "/" if stype == "directory" else path,
            "size": stats.st_size,
            "type": stype,
            "uid": stats.st_uid,
            "gid": stats.st_gid,
            "time": stats.st_atime,
            "mtime": stats.st_mtime,
        }
        return res

    def created(self, path):
        """Return the created timestamp of a file as a datetime.datetime"""
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port)
        return datetime.datetime.fromtimestamp(stats.st_ctime, tz=datetime.timezone.utc)

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        wpath = _as_unc_path(self.host, path)
        stats = smbclient.stat(wpath, port=self._port)
        return datetime.datetime.fromtimestamp(stats.st_mtime, tz=datetime.timezone.utc)

    def ls(self, path, detail=True, **kwargs):
        unc = _as_unc_path(self.host, path)
        listed = smbclient.listdir(unc, port=self._port, **kwargs)
        dirs = ["/".join([path.rstrip("/"), p]) for p in listed]
        if detail:
            dirs = [self.info(d) for d in dirs]
        return dirs

    # pylint: disable=too-many-arguments
    def _open(
        self,
        path,
        mode="rb",
        block_size=-1,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        """
        block_size: int or None
            If 0, no buffering, 1, line buffering, >1, buffer that many bytes

        Notes
        -----
        By specifying 'share_access' in 'kwargs' it is possible to override the
        default shared access setting applied in the constructor of this object.
        """
        if self.auto_mkdir and "w" in mode:
            self.makedirs(self._parent(path), exist_ok=True)
        bls = block_size if block_size is not None and block_size >= 0 else -1
        wpath = _as_unc_path(self.host, path)
        share_access = kwargs.pop("share_access", self.share_access)
        if "w" in mode and autocommit is False:
            temp = _as_temp_path(self.host, path, self.temppath)
            return SMBFileOpener(
                wpath, temp, mode, port=self._port, block_size=bls, **kwargs
            )
        return smbclient.open_file(
            wpath,
            mode,
            buffering=bls,
            share_access=share_access,
            port=self._port,
            **kwargs,
        )

    def copy(self, path1, path2, **kwargs):
        """Copy within two locations in the same filesystem"""
        wpath1 = _as_unc_path(self.host, path1)
        wpath2 = _as_unc_path(self.host, path2)
        if self.auto_mkdir:
            self.makedirs(self._parent(path2), exist_ok=True)
        smbclient.copyfile(wpath1, wpath2, port=self._port, **kwargs)

    def _rm(self, path):
        if _share_has_path(path):
            wpath = _as_unc_path(self.host, path)
            stats = smbclient.stat(wpath, port=self._port)
            if S_ISDIR(stats.st_mode):
                smbclient.rmdir(wpath, port=self._port)
            else:
                smbclient.remove(wpath, port=self._port)

    def mv(self, path1, path2, recursive=None, maxdepth=None, **kwargs):
        wpath1 = _as_unc_path(self.host, path1)
        wpath2 = _as_unc_path(self.host, path2)
        smbclient.rename(wpath1, wpath2, port=self._port, **kwargs)


def _as_unc_path(host, path):
    rpath = path.replace("/", "\\")
    unc = f"\\\\{host}{rpath}"
    return unc


def _as_temp_path(host, path, temppath):
    share = path.split("/")[1]
    temp_file = f"/{share}{temppath}/{uuid.uuid4()}"
    unc = _as_unc_path(host, temp_file)
    return unc


def _share_has_path(path):
    parts = path.count("/")
    if path.endswith("/"):
        return parts > 2
    return parts > 1


class SMBFileOpener:
    """writes to remote temporary file, move on commit"""

    def __init__(self, path, temp, mode, port=445, block_size=-1, **kwargs):
        self.path = path
        self.temp = temp
        self.mode = mode
        self.block_size = block_size
        self.kwargs = kwargs
        self.smbfile = None
        self._incontext = False
        self.port = port
        self._open()

    def _open(self):
        if self.smbfile is None or self.smbfile.closed:
            self.smbfile = smbclient.open_file(
                self.temp,
                self.mode,
                port=self.port,
                buffering=self.block_size,
                **self.kwargs,
            )

    def commit(self):
        """Move temp file to definitive on success."""
        # TODO: use transaction support in SMB protocol
        smbclient.replace(self.temp, self.path, port=self.port)

    def discard(self):
        """Remove the temp file on failure."""
        smbclient.remove(self.temp, port=self.port)

    def __fspath__(self):
        return self.path

    def __iter__(self):
        return self.smbfile.__iter__()

    def __getattr__(self, item):
        return getattr(self.smbfile, item)

    def __enter__(self):
        self._incontext = True
        return self.smbfile.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self._incontext = False
        self.smbfile.__exit__(exc_type, exc_value, traceback)
