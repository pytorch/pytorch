# https://hadoop.apache.org/docs/r1.0.4/webhdfs.html

import logging
import os
import secrets
import shutil
import tempfile
import uuid
from contextlib import suppress
from urllib.parse import quote

import requests

from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, tokenize

logger = logging.getLogger("webhdfs")


class WebHDFS(AbstractFileSystem):
    """
    Interface to HDFS over HTTP using the WebHDFS API. Supports also HttpFS gateways.

    Four auth mechanisms are supported:

    insecure: no auth is done, and the user is assumed to be whoever they
        say they are (parameter ``user``), or a predefined value such as
        "dr.who" if not given
    spnego: when kerberos authentication is enabled, auth is negotiated by
        requests_kerberos https://github.com/requests/requests-kerberos .
        This establishes a session based on existing kinit login and/or
        specified principal/password; parameters are passed with ``kerb_kwargs``
    token: uses an existing Hadoop delegation token from another secured
        service. Indeed, this client can also generate such tokens when
        not insecure. Note that tokens expire, but can be renewed (by a
        previously specified user) and may allow for proxying.
    basic-auth: used when both parameter ``user`` and parameter ``password``
        are provided.

    """

    tempdir = str(tempfile.gettempdir())
    protocol = "webhdfs", "webHDFS"

    def __init__(
        self,
        host,
        port=50070,
        kerberos=False,
        token=None,
        user=None,
        password=None,
        proxy_to=None,
        kerb_kwargs=None,
        data_proxy=None,
        use_https=False,
        session_cert=None,
        session_verify=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        host: str
            Name-node address
        port: int
            Port for webHDFS
        kerberos: bool
            Whether to authenticate with kerberos for this connection
        token: str or None
            If given, use this token on every call to authenticate. A user
            and user-proxy may be encoded in the token and should not be also
            given
        user: str or None
            If given, assert the user name to connect with
        password: str or None
            If given, assert the password to use for basic auth. If password
            is provided, user must be provided also
        proxy_to: str or None
            If given, the user has the authority to proxy, and this value is
            the user in who's name actions are taken
        kerb_kwargs: dict
            Any extra arguments for HTTPKerberosAuth, see
            `<https://github.com/requests/requests-kerberos/blob/master/requests_kerberos/kerberos_.py>`_
        data_proxy: dict, callable or None
            If given, map data-node addresses. This can be necessary if the
            HDFS cluster is behind a proxy, running on Docker or otherwise has
            a mismatch between the host-names given by the name-node and the
            address by which to refer to them from the client. If a dict,
            maps host names ``host->data_proxy[host]``; if a callable, full
            URLs are passed, and function must conform to
            ``url->data_proxy(url)``.
        use_https: bool
            Whether to connect to the Name-node using HTTPS instead of HTTP
        session_cert: str or Tuple[str, str] or None
            Path to a certificate file, or tuple of (cert, key) files to use
            for the requests.Session
        session_verify: str, bool or None
            Path to a certificate file to use for verifying the requests.Session.
        kwargs
        """
        if self._cached:
            return
        super().__init__(**kwargs)
        self.url = f"{'https' if use_https else 'http'}://{host}:{port}/webhdfs/v1"
        self.kerb = kerberos
        self.kerb_kwargs = kerb_kwargs or {}
        self.pars = {}
        self.proxy = data_proxy or {}
        if token is not None:
            if user is not None or proxy_to is not None:
                raise ValueError(
                    "If passing a delegation token, must not set "
                    "user or proxy_to, as these are encoded in the"
                    " token"
                )
            self.pars["delegation"] = token
        self.user = user
        self.password = password

        if password is not None:
            if user is None:
                raise ValueError(
                    "If passing a password, the user must also be"
                    "set in order to set up the basic-auth"
                )
        else:
            if user is not None:
                self.pars["user.name"] = user

        if proxy_to is not None:
            self.pars["doas"] = proxy_to
        if kerberos and user is not None:
            raise ValueError(
                "If using Kerberos auth, do not specify the "
                "user, this is handled by kinit."
            )

        self.session_cert = session_cert
        self.session_verify = session_verify

        self._connect()

        self._fsid = f"webhdfs_{tokenize(host, port)}"

    @property
    def fsid(self):
        return self._fsid

    def _connect(self):
        self.session = requests.Session()

        if self.session_cert:
            self.session.cert = self.session_cert

        self.session.verify = self.session_verify

        if self.kerb:
            from requests_kerberos import HTTPKerberosAuth

            self.session.auth = HTTPKerberosAuth(**self.kerb_kwargs)

        if self.user is not None and self.password is not None:
            from requests.auth import HTTPBasicAuth

            self.session.auth = HTTPBasicAuth(self.user, self.password)

    def _call(self, op, method="get", path=None, data=None, redirect=True, **kwargs):
        path = self._strip_protocol(path) if path is not None else ""
        url = self._apply_proxy(self.url + quote(path, safe="/="))
        args = kwargs.copy()
        args.update(self.pars)
        args["op"] = op.upper()
        logger.debug("sending %s with %s", url, method)
        out = self.session.request(
            method=method.upper(),
            url=url,
            params=args,
            data=data,
            allow_redirects=redirect,
        )
        if out.status_code in [400, 401, 403, 404, 500]:
            try:
                err = out.json()
                msg = err["RemoteException"]["message"]
                exp = err["RemoteException"]["exception"]
            except (ValueError, KeyError):
                pass
            else:
                if exp in ["IllegalArgumentException", "UnsupportedOperationException"]:
                    raise ValueError(msg)
                elif exp in ["SecurityException", "AccessControlException"]:
                    raise PermissionError(msg)
                elif exp in ["FileNotFoundException"]:
                    raise FileNotFoundError(msg)
                else:
                    raise RuntimeError(msg)
        out.raise_for_status()
        return out

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        replication=None,
        permissions=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        path: str
            File location
        mode: str
            'rb', 'wb', etc.
        block_size: int
            Client buffer size for read-ahead or write buffer
        autocommit: bool
            If False, writes to temporary file that only gets put in final
            location upon commit
        replication: int
            Number of copies of file on the cluster, write mode only
        permissions: str or int
            posix permissions, write mode only
        kwargs

        Returns
        -------
        WebHDFile instance
        """
        block_size = block_size or self.blocksize
        return WebHDFile(
            self,
            path,
            mode=mode,
            block_size=block_size,
            tempdir=self.tempdir,
            autocommit=autocommit,
            replication=replication,
            permissions=permissions,
        )

    @staticmethod
    def _process_info(info):
        info["type"] = info["type"].lower()
        info["size"] = info["length"]
        return info

    @classmethod
    def _strip_protocol(cls, path):
        return infer_storage_options(path)["path"]

    @staticmethod
    def _get_kwargs_from_urls(urlpath):
        out = infer_storage_options(urlpath)
        out.pop("path", None)
        out.pop("protocol", None)
        if "username" in out:
            out["user"] = out.pop("username")
        return out

    def info(self, path):
        out = self._call("GETFILESTATUS", path=path)
        info = out.json()["FileStatus"]
        info["name"] = path
        return self._process_info(info)

    def ls(self, path, detail=False, **kwargs):
        out = self._call("LISTSTATUS", path=path)
        infos = out.json()["FileStatuses"]["FileStatus"]
        for info in infos:
            self._process_info(info)
            info["name"] = path.rstrip("/") + "/" + info["pathSuffix"]
        if detail:
            return sorted(infos, key=lambda i: i["name"])
        else:
            return sorted(info["name"] for info in infos)

    def content_summary(self, path):
        """Total numbers of files, directories and bytes under path"""
        out = self._call("GETCONTENTSUMMARY", path=path)
        return out.json()["ContentSummary"]

    def ukey(self, path):
        """Checksum info of file, giving method and result"""
        out = self._call("GETFILECHECKSUM", path=path, redirect=False)
        if "Location" in out.headers:
            location = self._apply_proxy(out.headers["Location"])
            out2 = self.session.get(location)
            out2.raise_for_status()
            return out2.json()["FileChecksum"]
        else:
            out.raise_for_status()
            return out.json()["FileChecksum"]

    def home_directory(self):
        """Get user's home directory"""
        out = self._call("GETHOMEDIRECTORY")
        return out.json()["Path"]

    def get_delegation_token(self, renewer=None):
        """Retrieve token which can give the same authority to other uses

        Parameters
        ----------
        renewer: str or None
            User who may use this token; if None, will be current user
        """
        if renewer:
            out = self._call("GETDELEGATIONTOKEN", renewer=renewer)
        else:
            out = self._call("GETDELEGATIONTOKEN")
        t = out.json()["Token"]
        if t is None:
            raise ValueError("No token available for this user/security context")
        return t["urlString"]

    def renew_delegation_token(self, token):
        """Make token live longer. Returns new expiry time"""
        out = self._call("RENEWDELEGATIONTOKEN", method="put", token=token)
        return out.json()["long"]

    def cancel_delegation_token(self, token):
        """Stop the token from being useful"""
        self._call("CANCELDELEGATIONTOKEN", method="put", token=token)

    def chmod(self, path, mod):
        """Set the permission at path

        Parameters
        ----------
        path: str
            location to set (file or directory)
        mod: str or int
            posix epresentation or permission, give as oct string, e.g, '777'
            or 0o777
        """
        self._call("SETPERMISSION", method="put", path=path, permission=mod)

    def chown(self, path, owner=None, group=None):
        """Change owning user and/or group"""
        kwargs = {}
        if owner is not None:
            kwargs["owner"] = owner
        if group is not None:
            kwargs["group"] = group
        self._call("SETOWNER", method="put", path=path, **kwargs)

    def set_replication(self, path, replication):
        """
        Set file replication factor

        Parameters
        ----------
        path: str
            File location (not for directories)
        replication: int
            Number of copies of file on the cluster. Should be smaller than
            number of data nodes; normally 3 on most systems.
        """
        self._call("SETREPLICATION", path=path, method="put", replication=replication)

    def mkdir(self, path, **kwargs):
        self._call("MKDIRS", method="put", path=path)

    def makedirs(self, path, exist_ok=False):
        if exist_ok is False and self.exists(path):
            raise FileExistsError(path)
        self.mkdir(path)

    def mv(self, path1, path2, **kwargs):
        self._call("RENAME", method="put", path=path1, destination=path2)

    def rm(self, path, recursive=False, **kwargs):
        self._call(
            "DELETE",
            method="delete",
            path=path,
            recursive="true" if recursive else "false",
        )

    def rm_file(self, path, **kwargs):
        self.rm(path)

    def cp_file(self, lpath, rpath, **kwargs):
        with self.open(lpath) as lstream:
            tmp_fname = "/".join([self._parent(rpath), f".tmp.{secrets.token_hex(16)}"])
            # Perform an atomic copy (stream to a temporary file and
            # move it to the actual destination).
            try:
                with self.open(tmp_fname, "wb") as rstream:
                    shutil.copyfileobj(lstream, rstream)
                self.mv(tmp_fname, rpath)
            except BaseException:
                with suppress(FileNotFoundError):
                    self.rm(tmp_fname)
                raise

    def _apply_proxy(self, location):
        if self.proxy and callable(self.proxy):
            location = self.proxy(location)
        elif self.proxy:
            # as a dict
            for k, v in self.proxy.items():
                location = location.replace(k, v, 1)
        return location


class WebHDFile(AbstractBufferedFile):
    """A file living in HDFS over webHDFS"""

    def __init__(self, fs, path, **kwargs):
        super().__init__(fs, path, **kwargs)
        kwargs = kwargs.copy()
        if kwargs.get("permissions", None) is None:
            kwargs.pop("permissions", None)
        if kwargs.get("replication", None) is None:
            kwargs.pop("replication", None)
        self.permissions = kwargs.pop("permissions", 511)
        tempdir = kwargs.pop("tempdir")
        if kwargs.pop("autocommit", False) is False:
            self.target = self.path
            self.path = os.path.join(tempdir, str(uuid.uuid4()))

    def _upload_chunk(self, final=False):
        """Write one part of a multi-block file upload

        Parameters
        ==========
        final: bool
            This is the last block, so should complete file, if
            self.autocommit is True.
        """
        out = self.fs.session.post(
            self.location,
            data=self.buffer.getvalue(),
            headers={"content-type": "application/octet-stream"},
        )
        out.raise_for_status()
        return True

    def _initiate_upload(self):
        """Create remote file/upload"""
        kwargs = self.kwargs.copy()
        if "a" in self.mode:
            op, method = "APPEND", "POST"
        else:
            op, method = "CREATE", "PUT"
            kwargs["overwrite"] = "true"
        out = self.fs._call(op, method, self.path, redirect=False, **kwargs)
        location = self.fs._apply_proxy(out.headers["Location"])
        if "w" in self.mode:
            # create empty file to append to
            out2 = self.fs.session.put(
                location, headers={"content-type": "application/octet-stream"}
            )
            out2.raise_for_status()
            # after creating empty file, change location to append to
            out2 = self.fs._call("APPEND", "POST", self.path, redirect=False, **kwargs)
            self.location = self.fs._apply_proxy(out2.headers["Location"])

    def _fetch_range(self, start, end):
        start = max(start, 0)
        end = min(self.size, end)
        if start >= end or start >= self.size:
            return b""
        out = self.fs._call(
            "OPEN", path=self.path, offset=start, length=end - start, redirect=False
        )
        out.raise_for_status()
        if "Location" in out.headers:
            location = out.headers["Location"]
            out2 = self.fs.session.get(self.fs._apply_proxy(location))
            return out2.content
        else:
            return out.content

    def commit(self):
        self.fs.mv(self.path, self.target)

    def discard(self):
        self.fs.rm(self.path)
