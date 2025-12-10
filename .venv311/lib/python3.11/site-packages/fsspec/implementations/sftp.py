import datetime
import logging
import os
import types
import uuid
from stat import S_ISDIR, S_ISLNK

import paramiko

from .. import AbstractFileSystem
from ..utils import infer_storage_options

logger = logging.getLogger("fsspec.sftp")


class SFTPFileSystem(AbstractFileSystem):
    """Files over SFTP/SSH

    Peer-to-peer filesystem over SSH using paramiko.

    Note: if using this with the ``open`` or ``open_files``, with full URLs,
    there is no way to tell if a path is relative, so all paths are assumed
    to be absolute.
    """

    protocol = "sftp", "ssh"

    def __init__(self, host, **ssh_kwargs):
        """

        Parameters
        ----------
        host: str
            Hostname or IP as a string
        temppath: str
            Location on the server to put files, when within a transaction
        ssh_kwargs: dict
            Parameters passed on to connection. See details in
            https://docs.paramiko.org/en/3.3/api/client.html#paramiko.client.SSHClient.connect
            May include port, username, password...
        """
        if self._cached:
            return
        super().__init__(**ssh_kwargs)
        self.temppath = ssh_kwargs.pop("temppath", "/tmp")  # remote temp directory
        self.host = host
        self.ssh_kwargs = ssh_kwargs
        self._connect()

    def _connect(self):
        logger.debug("Connecting to SFTP server %s", self.host)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.host, **self.ssh_kwargs)
        self.ftp = self.client.open_sftp()

    @classmethod
    def _strip_protocol(cls, path):
        return infer_storage_options(path)["path"]

    @staticmethod
    def _get_kwargs_from_urls(urlpath):
        out = infer_storage_options(urlpath)
        out.pop("path", None)
        out.pop("protocol", None)
        return out

    def mkdir(self, path, create_parents=True, mode=511):
        path = self._strip_protocol(path)
        logger.debug("Creating folder %s", path)
        if self.exists(path):
            raise FileExistsError(f"File exists: {path}")

        if create_parents:
            self.makedirs(path)
        else:
            self.ftp.mkdir(path, mode)

    def makedirs(self, path, exist_ok=False, mode=511):
        if self.exists(path) and not exist_ok:
            raise FileExistsError(f"File exists: {path}")

        parts = path.split("/")
        new_path = "/" if path[:1] == "/" else ""

        for part in parts:
            if part:
                new_path = f"{new_path}/{part}" if new_path else part
                if not self.exists(new_path):
                    self.ftp.mkdir(new_path, mode)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        logger.debug("Removing folder %s", path)
        self.ftp.rmdir(path)

    def info(self, path):
        path = self._strip_protocol(path)
        stat = self._decode_stat(self.ftp.stat(path))
        stat["name"] = path
        return stat

    @staticmethod
    def _decode_stat(stat, parent_path=None):
        if S_ISDIR(stat.st_mode):
            t = "directory"
        elif S_ISLNK(stat.st_mode):
            t = "link"
        else:
            t = "file"
        out = {
            "name": "",
            "size": stat.st_size,
            "type": t,
            "uid": stat.st_uid,
            "gid": stat.st_gid,
            "time": datetime.datetime.fromtimestamp(
                stat.st_atime, tz=datetime.timezone.utc
            ),
            "mtime": datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ),
        }
        if parent_path:
            out["name"] = "/".join([parent_path.rstrip("/"), stat.filename])
        return out

    def ls(self, path, detail=False):
        path = self._strip_protocol(path)
        logger.debug("Listing folder %s", path)
        stats = [self._decode_stat(stat, path) for stat in self.ftp.listdir_iter(path)]
        if detail:
            return stats
        else:
            paths = [stat["name"] for stat in stats]
            return sorted(paths)

    def put(self, lpath, rpath, callback=None, **kwargs):
        rpath = self._strip_protocol(rpath)
        logger.debug("Put file %s into %s", lpath, rpath)
        self.ftp.put(lpath, rpath)

    def get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
        else:
            self.ftp.get(self._strip_protocol(rpath), lpath)

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        """
        block_size: int or None
            If 0, no buffering, if 1, line buffering, if >1, buffer that many
            bytes, if None use default from paramiko.
        """
        logger.debug("Opening file %s", path)
        if kwargs.get("autocommit", True) is False:
            # writes to temporary file, move on commit
            path2 = "/".join([self.temppath, str(uuid.uuid4())])
            f = self.ftp.open(path2, mode, bufsize=block_size if block_size else -1)
            f.temppath = path2
            f.targetpath = path
            f.fs = self
            f.commit = types.MethodType(commit_a_file, f)
            f.discard = types.MethodType(discard_a_file, f)
        else:
            f = self.ftp.open(path, mode, bufsize=block_size if block_size else -1)
        return f

    def _rm(self, path):
        if self.isdir(path):
            self.ftp.rmdir(path)
        else:
            self.ftp.remove(path)

    def mv(self, old, new):
        new = self._strip_protocol(new)
        old = self._strip_protocol(old)
        logger.debug("Renaming %s into %s", old, new)
        self.ftp.posix_rename(old, new)


def commit_a_file(self):
    self.fs.mv(self.temppath, self.targetpath)


def discard_a_file(self):
    self.fs._rm(self.temppath)
