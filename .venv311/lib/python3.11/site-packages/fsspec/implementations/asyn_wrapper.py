import asyncio
import functools
import inspect

import fsspec
from fsspec.asyn import AsyncFileSystem, running_async

from .chained import ChainedFileSystem


def async_wrapper(func, obj=None, semaphore=None):
    """
    Wraps a synchronous function to make it awaitable.

    Parameters
    ----------
    func : callable
        The synchronous function to wrap.
    obj : object, optional
        The instance to bind the function to, if applicable.
    semaphore : asyncio.Semaphore, optional
        A semaphore to limit concurrent calls.

    Returns
    -------
    coroutine
        An awaitable version of the function.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if semaphore:
            async with semaphore:
                return await asyncio.to_thread(func, *args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


class AsyncFileSystemWrapper(AsyncFileSystem, ChainedFileSystem):
    """
    A wrapper class to convert a synchronous filesystem into an asynchronous one.

    This class takes an existing synchronous filesystem implementation and wraps all
    its methods to provide an asynchronous interface.

    Parameters
    ----------
    sync_fs : AbstractFileSystem
        The synchronous filesystem instance to wrap.
    """

    protocol = "asyncwrapper", "async_wrapper"
    cachable = False

    def __init__(
        self,
        fs=None,
        asynchronous=None,
        target_protocol=None,
        target_options=None,
        semaphore=None,
        max_concurrent_tasks=None,
        **kwargs,
    ):
        if asynchronous is None:
            asynchronous = running_async()
        super().__init__(asynchronous=asynchronous, **kwargs)
        if fs is not None:
            self.sync_fs = fs
        else:
            self.sync_fs = fsspec.filesystem(target_protocol, **target_options)
        self.protocol = self.sync_fs.protocol
        self.semaphore = semaphore
        self._wrap_all_sync_methods()

    @property
    def fsid(self):
        return f"async_{self.sync_fs.fsid}"

    def _wrap_all_sync_methods(self):
        """
        Wrap all synchronous methods of the underlying filesystem with asynchronous versions.
        """
        excluded_methods = {"open"}
        for method_name in dir(self.sync_fs):
            if method_name.startswith("_") or method_name in excluded_methods:
                continue

            attr = inspect.getattr_static(self.sync_fs, method_name)
            if isinstance(attr, property):
                continue

            method = getattr(self.sync_fs, method_name)
            if callable(method) and not inspect.iscoroutinefunction(method):
                async_method = async_wrapper(method, obj=self, semaphore=self.semaphore)
                setattr(self, f"_{method_name}", async_method)

    @classmethod
    def wrap_class(cls, sync_fs_class):
        """
        Create a new class that can be used to instantiate an AsyncFileSystemWrapper
        with lazy instantiation of the underlying synchronous filesystem.

        Parameters
        ----------
        sync_fs_class : type
            The class of the synchronous filesystem to wrap.

        Returns
        -------
        type
            A new class that wraps the provided synchronous filesystem class.
        """

        class GeneratedAsyncFileSystemWrapper(cls):
            def __init__(self, *args, **kwargs):
                sync_fs = sync_fs_class(*args, **kwargs)
                super().__init__(sync_fs)

        GeneratedAsyncFileSystemWrapper.__name__ = (
            f"Async{sync_fs_class.__name__}Wrapper"
        )
        return GeneratedAsyncFileSystemWrapper
