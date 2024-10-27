import dask
from distributed.client import Client, _get_global_client
from distributed.worker import Worker

from fsspec import filesystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_storage_options


def _get_client(client):
    if client is None:
        return _get_global_client()
    elif isinstance(client, Client):
        return client
    else:
        # e.g., connection string
        return Client(client)


def _in_worker():
    return bool(Worker._instances)


class DaskWorkerFileSystem(AbstractFileSystem):
    """View files accessible to a worker as any other remote file-system

    When instances are run on the worker, uses the real filesystem. When
    run on the client, they call the worker to provide information or data.

    **Warning** this implementation is experimental, and read-only for now.
    """

    def __init__(
        self, target_protocol=None, target_options=None, fs=None, client=None, **kwargs
    ):
        super().__init__(**kwargs)
        if not (fs is None) ^ (target_protocol is None):
            raise ValueError(
                "Please provide one of filesystem instance (fs) or"
                " target_protocol, not both"
            )
        self.target_protocol = target_protocol
        self.target_options = target_options
        self.worker = None
        self.client = client
        self.fs = fs
        self._determine_worker()

    @staticmethod
    def _get_kwargs_from_urls(path):
        so = infer_storage_options(path)
        if "host" in so and "port" in so:
            return {"client": f"{so['host']}:{so['port']}"}
        else:
            return {}

    def _determine_worker(self):
        if _in_worker():
            self.worker = True
            if self.fs is None:
                self.fs = filesystem(
                    self.target_protocol, **(self.target_options or {})
                )
        else:
            self.worker = False
            self.client = _get_client(self.client)
            self.rfs = dask.delayed(self)

    def mkdir(self, *args, **kwargs):
        if self.worker:
            self.fs.mkdir(*args, **kwargs)
        else:
            self.rfs.mkdir(*args, **kwargs).compute()

    def rm(self, *args, **kwargs):
        if self.worker:
            self.fs.rm(*args, **kwargs)
        else:
            self.rfs.rm(*args, **kwargs).compute()

    def copy(self, *args, **kwargs):
        if self.worker:
            self.fs.copy(*args, **kwargs)
        else:
            self.rfs.copy(*args, **kwargs).compute()

    def mv(self, *args, **kwargs):
        if self.worker:
            self.fs.mv(*args, **kwargs)
        else:
            self.rfs.mv(*args, **kwargs).compute()

    def ls(self, *args, **kwargs):
        if self.worker:
            return self.fs.ls(*args, **kwargs)
        else:
            return self.rfs.ls(*args, **kwargs).compute()

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        if self.worker:
            return self.fs._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )
        else:
            return DaskFile(
                fs=self,
                path=path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )

    def fetch_range(self, path, mode, start, end):
        if self.worker:
            with self._open(path, mode) as f:
                f.seek(start)
                return f.read(end - start)
        else:
            return self.rfs.fetch_range(path, mode, start, end).compute()


class DaskFile(AbstractBufferedFile):
    def __init__(self, mode="rb", **kwargs):
        if mode != "rb":
            raise ValueError('Remote dask files can only be opened in "rb" mode')
        super().__init__(**kwargs)

    def _upload_chunk(self, final=False):
        pass

    def _initiate_upload(self):
        """Create remote file/upload"""
        pass

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        return self.fs.fetch_range(self.path, self.mode, start, end)
