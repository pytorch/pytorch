from __future__ import annotations

import base64
import urllib

import requests
from requests.adapters import HTTPAdapter, Retry
from typing_extensions import override

from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile


class DatabricksException(Exception):
    """
    Helper class for exceptions raised in this module.
    """

    def __init__(self, error_code, message, details=None):
        """Create a new DatabricksException"""
        super().__init__(message)

        self.error_code = error_code
        self.message = message
        self.details = details


class DatabricksFileSystem(AbstractFileSystem):
    """
    Get access to the Databricks filesystem implementation over HTTP.
    Can be used inside and outside of a databricks cluster.
    """

    def __init__(self, instance, token, **kwargs):
        """
        Create a new DatabricksFileSystem.

        Parameters
        ----------
        instance: str
            The instance URL of the databricks cluster.
            For example for an Azure databricks cluster, this
            has the form adb-<some-number>.<two digits>.azuredatabricks.net.
        token: str
            Your personal token. Find out more
            here: https://docs.databricks.com/dev-tools/api/latest/authentication.html
        """
        self.instance = instance
        self.token = token
        self.session = requests.Session()
        self.retries = Retry(
            total=10,
            backoff_factor=0.05,
            status_forcelist=[408, 429, 500, 502, 503, 504],
        )

        self.session.mount("https://", HTTPAdapter(max_retries=self.retries))
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

        super().__init__(**kwargs)

    @override
    def _ls_from_cache(self, path) -> list[dict[str, str | int]] | None:
        """Check cache for listing

        Returns listing, if found (may be empty list for a directory that
        exists but contains nothing), None if not in cache.
        """
        self.dircache.pop(path.rstrip("/"), None)

        parent = self._parent(path)
        if parent in self.dircache:
            for entry in self.dircache[parent]:
                if entry["name"] == path.rstrip("/"):
                    if entry["type"] != "directory":
                        return [entry]
                    return []
            raise FileNotFoundError(path)

    def ls(self, path, detail=True, **kwargs):
        """
        List the contents of the given path.

        Parameters
        ----------
        path: str
            Absolute path
        detail: bool
            Return not only the list of filenames,
            but also additional information on file sizes
            and types.
        """
        try:
            out = self._ls_from_cache(path)
        except FileNotFoundError:
            # This happens if the `path`'s parent was cached, but `path` is not
            # there. This suggests that `path` is new since the parent was
            # cached. Attempt to invalidate parent's cache before continuing.
            self.dircache.pop(self._parent(path), None)
            out = None

        if not out:
            try:
                r = self._send_to_api(
                    method="get", endpoint="list", json={"path": path}
                )
            except DatabricksException as e:
                if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                    raise FileNotFoundError(e.message) from e

                raise
            files = r.get("files", [])
            out = [
                {
                    "name": o["path"],
                    "type": "directory" if o["is_dir"] else "file",
                    "size": o["file_size"],
                }
                for o in files
            ]
            self.dircache[path] = out

        if detail:
            return out
        return [o["name"] for o in out]

    def makedirs(self, path, exist_ok=True):
        """
        Create a given absolute path and all of its parents.

        Parameters
        ----------
        path: str
            Absolute path to create
        exist_ok: bool
            If false, checks if the folder
            exists before creating it (and raises an
            Exception if this is the case)
        """
        if not exist_ok:
            try:
                # If the following succeeds, the path is already present
                self._send_to_api(
                    method="get", endpoint="get-status", json={"path": path}
                )
                raise FileExistsError(f"Path {path} already exists")
            except DatabricksException as e:
                if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                    pass

        try:
            self._send_to_api(method="post", endpoint="mkdirs", json={"path": path})
        except DatabricksException as e:
            if e.error_code == "RESOURCE_ALREADY_EXISTS":
                raise FileExistsError(e.message) from e

            raise
        self.invalidate_cache(self._parent(path))

    def mkdir(self, path, create_parents=True, **kwargs):
        """
        Create a given absolute path and all of its parents.

        Parameters
        ----------
        path: str
            Absolute path to create
        create_parents: bool
            Whether to create all parents or not.
            "False" is not implemented so far.
        """
        if not create_parents:
            raise NotImplementedError

        self.mkdirs(path, **kwargs)

    def rm(self, path, recursive=False, **kwargs):
        """
        Remove the file or folder at the given absolute path.

        Parameters
        ----------
        path: str
            Absolute path what to remove
        recursive: bool
            Recursively delete all files in a folder.
        """
        try:
            self._send_to_api(
                method="post",
                endpoint="delete",
                json={"path": path, "recursive": recursive},
            )
        except DatabricksException as e:
            # This is not really an exception, it just means
            # not everything was deleted so far
            if e.error_code == "PARTIAL_DELETE":
                self.rm(path=path, recursive=recursive)
            elif e.error_code == "IO_ERROR":
                # Using the same exception as the os module would use here
                raise OSError(e.message) from e

            raise
        self.invalidate_cache(self._parent(path))

    def mv(
        self, source_path, destination_path, recursive=False, maxdepth=None, **kwargs
    ):
        """
        Move a source to a destination path.

        A note from the original [databricks API manual]
        (https://docs.databricks.com/dev-tools/api/latest/dbfs.html#move).

        When moving a large number of files the API call will time out after
        approximately 60s, potentially resulting in partially moved data.
        Therefore, for operations that move more than 10k files, we strongly
        discourage using the DBFS REST API.

        Parameters
        ----------
        source_path: str
            From where to move (absolute path)
        destination_path: str
            To where to move (absolute path)
        recursive: bool
            Not implemented to far.
        maxdepth:
            Not implemented to far.
        """
        if recursive:
            raise NotImplementedError
        if maxdepth:
            raise NotImplementedError

        try:
            self._send_to_api(
                method="post",
                endpoint="move",
                json={"source_path": source_path, "destination_path": destination_path},
            )
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e
            elif e.error_code == "RESOURCE_ALREADY_EXISTS":
                raise FileExistsError(e.message) from e

            raise
        self.invalidate_cache(self._parent(source_path))
        self.invalidate_cache(self._parent(destination_path))

    def _open(self, path, mode="rb", block_size="default", **kwargs):
        """
        Overwrite the base class method to make sure to create a DBFile.
        All arguments are copied from the base method.

        Only the default blocksize is allowed.
        """
        return DatabricksFile(self, path, mode=mode, block_size=block_size, **kwargs)

    def _send_to_api(self, method, endpoint, json):
        """
        Send the given json to the DBFS API
        using a get or post request (specified by the argument `method`).

        Parameters
        ----------
        method: str
            Which http method to use for communication; "get" or "post".
        endpoint: str
            Where to send the request to (last part of the API URL)
        json: dict
            Dictionary of information to send
        """
        if method == "post":
            session_call = self.session.post
        elif method == "get":
            session_call = self.session.get
        else:
            raise ValueError(f"Do not understand method {method}")

        url = urllib.parse.urljoin(f"https://{self.instance}/api/2.0/dbfs/", endpoint)

        r = session_call(url, json=json)

        # The DBFS API will return a json, also in case of an exception.
        # We want to preserve this information as good as possible.
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # try to extract json error message
            # if that fails, fall back to the original exception
            try:
                exception_json = e.response.json()
            except Exception:
                raise e from None

            raise DatabricksException(**exception_json) from e

        return r.json()

    def _create_handle(self, path, overwrite=True):
        """
        Internal function to create a handle, which can be used to
        write blocks of a file to DBFS.
        A handle has a unique identifier which needs to be passed
        whenever written during this transaction.
        The handle is active for 10 minutes - after that a new
        write transaction needs to be created.
        Make sure to close the handle after you are finished.

        Parameters
        ----------
        path: str
            Absolute path for this file.
        overwrite: bool
            If a file already exist at this location, either overwrite
            it or raise an exception.
        """
        try:
            r = self._send_to_api(
                method="post",
                endpoint="create",
                json={"path": path, "overwrite": overwrite},
            )
            return r["handle"]
        except DatabricksException as e:
            if e.error_code == "RESOURCE_ALREADY_EXISTS":
                raise FileExistsError(e.message) from e

            raise

    def _close_handle(self, handle):
        """
        Close a handle, which was opened by :func:`_create_handle`.

        Parameters
        ----------
        handle: str
            Which handle to close.
        """
        try:
            self._send_to_api(method="post", endpoint="close", json={"handle": handle})
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e

            raise

    def _add_data(self, handle, data):
        """
        Upload data to an already opened file handle
        (opened by :func:`_create_handle`).
        The maximal allowed data size is 1MB after
        conversion to base64.
        Remember to close the handle when you are finished.

        Parameters
        ----------
        handle: str
            Which handle to upload data to.
        data: bytes
            Block of data to add to the handle.
        """
        data = base64.b64encode(data).decode()
        try:
            self._send_to_api(
                method="post",
                endpoint="add-block",
                json={"handle": handle, "data": data},
            )
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e
            elif e.error_code == "MAX_BLOCK_SIZE_EXCEEDED":
                raise ValueError(e.message) from e

            raise

    def _get_data(self, path, start, end):
        """
        Download data in bytes from a given absolute path in a block
        from [start, start+length].
        The maximum number of allowed bytes to read is 1MB.

        Parameters
        ----------
        path: str
            Absolute path to download data from
        start: int
            Start position of the block
        end: int
            End position of the block
        """
        try:
            r = self._send_to_api(
                method="get",
                endpoint="read",
                json={"path": path, "offset": start, "length": end - start},
            )
            return base64.b64decode(r["data"])
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e
            elif e.error_code in ["INVALID_PARAMETER_VALUE", "MAX_READ_SIZE_EXCEEDED"]:
                raise ValueError(e.message) from e

            raise

    def invalidate_cache(self, path=None):
        if path is None:
            self.dircache.clear()
        else:
            self.dircache.pop(path, None)
        super().invalidate_cache(path)


class DatabricksFile(AbstractBufferedFile):
    """
    Helper class for files referenced in the DatabricksFileSystem.
    """

    DEFAULT_BLOCK_SIZE = 1 * 2**20  # only allowed block size

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
        """
        Create a new instance of the DatabricksFile.

        The blocksize needs to be the default one.
        """
        if block_size is None or block_size == "default":
            block_size = self.DEFAULT_BLOCK_SIZE

        assert block_size == self.DEFAULT_BLOCK_SIZE, (
            f"Only the default block size is allowed, not {block_size}"
        )

        super().__init__(
            fs,
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_type=cache_type,
            cache_options=cache_options or {},
            **kwargs,
        )

    def _initiate_upload(self):
        """Internal function to start a file upload"""
        self.handle = self.fs._create_handle(self.path)

    def _upload_chunk(self, final=False):
        """Internal function to add a chunk of data to a started upload"""
        self.buffer.seek(0)
        data = self.buffer.getvalue()

        data_chunks = [
            data[start:end] for start, end in self._to_sized_blocks(len(data))
        ]

        for data_chunk in data_chunks:
            self.fs._add_data(handle=self.handle, data=data_chunk)

        if final:
            self.fs._close_handle(handle=self.handle)
            return True

    def _fetch_range(self, start, end):
        """Internal function to download a block of data"""
        return_buffer = b""
        length = end - start
        for chunk_start, chunk_end in self._to_sized_blocks(length, start):
            return_buffer += self.fs._get_data(
                path=self.path, start=chunk_start, end=chunk_end
            )

        return return_buffer

    def _to_sized_blocks(self, length, start=0):
        """Helper function to split a range from 0 to total_length into blocksizes"""
        end = start + length
        for data_chunk in range(start, end, self.blocksize):
            data_start = data_chunk
            data_end = min(end, data_chunk + self.blocksize)
            yield data_start, data_end
