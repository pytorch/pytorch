from __future__ import annotations

import io
import json
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import Any, ClassVar

from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
    _unstrip_protocol,
    glob_translate,
    isfilelike,
    other_paths,
    read_block,
    stringify_path,
    tokenize,
)

logger = logging.getLogger("fsspec")


def make_instance(cls, args, kwargs):
    return cls(*args, **kwargs)


class _Cached(type):
    """
    Metaclass for caching file system instances.

    Notes
    -----
    Instances are cached according to

    * The values of the class attributes listed in `_extra_tokenize_attributes`
    * The arguments passed to ``__init__``.

    This creates an additional reference to the filesystem, which prevents the
    filesystem from being garbage collected when all *user* references go away.
    A call to the :meth:`AbstractFileSystem.clear_instance_cache` must *also*
    be made for a filesystem instance to be garbage collected.
    """

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: we intentionally create a reference here, to avoid garbage
        # collecting instances when all other references are gone. To really
        # delete a FileSystem, the cache must be cleared.
        if conf.get("weakref_instance_cache"):  # pragma: no cover
            # debug option for analysing fork/spawn conditions
            cls._cache = weakref.WeakValueDictionary()
        else:
            cls._cache = {}
        cls._pid = os.getpid()

    def __call__(cls, *args, **kwargs):
        kwargs = apply_config(cls, kwargs)
        extra_tokens = tuple(
            getattr(cls, attr, None) for attr in cls._extra_tokenize_attributes
        )
        strip_tokenize_options = {
            k: kwargs.pop(k) for k in cls._strip_tokenize_options if k in kwargs
        }
        token = tokenize(
            cls, cls._pid, threading.get_ident(), *args, *extra_tokens, **kwargs
        )
        skip = kwargs.pop("skip_instance_cache", False)
        if os.getpid() != cls._pid:
            cls._cache.clear()
            cls._pid = os.getpid()
        if not skip and cls.cachable and token in cls._cache:
            cls._latest = token
            return cls._cache[token]
        else:
            obj = super().__call__(*args, **kwargs, **strip_tokenize_options)
            # Setting _fs_token here causes some static linters to complain.
            obj._fs_token_ = token
            obj.storage_args = args
            obj.storage_options = kwargs
            if obj.async_impl and obj.mirror_sync_methods:
                from .asyn import mirror_sync_methods

                mirror_sync_methods(obj)

            if cls.cachable and not skip:
                cls._latest = token
                cls._cache[token] = obj
            return obj


class AbstractFileSystem(metaclass=_Cached):
    """
    An abstract super-class for pythonic file-systems

    Implementations are expected to be compatible with or, better, subclass
    from here.
    """

    cachable = True  # this class can be cached, instances reused
    _cached = False
    blocksize = 2**22
    sep = "/"
    protocol: ClassVar[str | tuple[str, ...]] = "abstract"
    _latest = None
    async_impl = False
    mirror_sync_methods = False
    root_marker = ""  # For some FSs, may require leading '/' or other character
    transaction_type = Transaction

    #: Extra *class attributes* that should be considered when hashing.
    _extra_tokenize_attributes = ()
    #: *storage options* that should not be considered when hashing.
    _strip_tokenize_options = ()

    # Set by _Cached metaclass
    storage_args: tuple[Any, ...]
    storage_options: dict[str, Any]

    def __init__(self, *args, **storage_options):
        """Create and configure file-system instance

        Instances may be cachable, so if similar enough arguments are seen
        a new instance is not required. The token attribute exists to allow
        implementations to cache instances if they wish.

        A reasonable default should be provided if there are no arguments.

        Subclasses should call this method.

        Parameters
        ----------
        use_listings_cache, listings_expiry_time, max_paths:
            passed to ``DirCache``, if the implementation supports
            directory listing caching. Pass use_listings_cache=False
            to disable such caching.
        skip_instance_cache: bool
            If this is a cachable implementation, pass True here to force
            creating a new instance even if a matching instance exists, and prevent
            storing this instance.
        asynchronous: bool
        loop: asyncio-compatible IOLoop or None
        """
        if self._cached:
            # reusing instance, don't change
            return
        self._cached = True
        self._intrans = False
        self._transaction = None
        self._invalidated_caches_in_transaction = []
        self.dircache = DirCache(**storage_options)

        if storage_options.pop("add_docs", None):
            warnings.warn("add_docs is no longer supported.", FutureWarning)

        if storage_options.pop("add_aliases", None):
            warnings.warn("add_aliases has been removed.", FutureWarning)
        # This is set in _Cached
        self._fs_token_ = None

    @property
    def fsid(self):
        """Persistent filesystem id that can be used to compare filesystems
        across sessions.
        """
        raise NotImplementedError

    @property
    def _fs_token(self):
        return self._fs_token_

    def __dask_tokenize__(self):
        return self._fs_token

    def __hash__(self):
        return int(self._fs_token, 16)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self._fs_token == other._fs_token

    def __reduce__(self):
        return make_instance, (type(self), self.storage_args, self.storage_options)

    @classmethod
    def _strip_protocol(cls, path):
        """Turn path from fully-qualified to file-system-specific

        May require FS-specific handling, e.g., for relative paths or links.
        """
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        path = stringify_path(path)
        protos = (cls.protocol,) if isinstance(cls.protocol, str) else cls.protocol
        for protocol in protos:
            if path.startswith(protocol + "://"):
                path = path[len(protocol) + 3 :]
            elif path.startswith(protocol + "::"):
                path = path[len(protocol) + 2 :]
        path = path.rstrip("/")
        # use of root_marker to make minimum required path, e.g., "/"
        return path or cls.root_marker

    def unstrip_protocol(self, name: str) -> str:
        """Format FS-specific path to generic, including protocol"""
        protos = (self.protocol,) if isinstance(self.protocol, str) else self.protocol
        for protocol in protos:
            if name.startswith(f"{protocol}://"):
                return name
        return f"{protos[0]}://{name}"

    @staticmethod
    def _get_kwargs_from_urls(path):
        """If kwargs can be encoded in the paths, extract them here

        This should happen before instantiation of the class; incoming paths
        then should be amended to strip the options in methods.

        Examples may look like an sftp path "sftp://user@host:/my/path", where
        the user and host should become kwargs and later get stripped.
        """
        # by default, nothing happens
        return {}

    @classmethod
    def current(cls):
        """Return the most recently instantiated FileSystem

        If no instance has been created, then create one with defaults
        """
        if cls._latest in cls._cache:
            return cls._cache[cls._latest]
        return cls()

    @property
    def transaction(self):
        """A context within which files are committed together upon exit

        Requires the file class to implement `.commit()` and `.discard()`
        for the normal and exception cases.
        """
        if self._transaction is None:
            self._transaction = self.transaction_type(self)
        return self._transaction

    def start_transaction(self):
        """Begin write transaction for deferring files, non-context version"""
        self._intrans = True
        self._transaction = self.transaction_type(self)
        return self.transaction

    def end_transaction(self):
        """Finish write transaction, non-context version"""
        self.transaction.complete()
        self._transaction = None
        # The invalid cache must be cleared after the transaction is completed.
        for path in self._invalidated_caches_in_transaction:
            self.invalidate_cache(path)
        self._invalidated_caches_in_transaction.clear()

    def invalidate_cache(self, path=None):
        """
        Discard any cached directory information

        Parameters
        ----------
        path: string or None
            If None, clear all listings cached else listings at or under given
            path.
        """
        # Not necessary to implement invalidation mechanism, may have no cache.
        # But if have, you should call this method of parent class from your
        # subclass to ensure expiring caches after transacations correctly.
        # See the implementation of FTPFileSystem in ftp.py
        if self._intrans:
            self._invalidated_caches_in_transaction.append(path)

    def mkdir(self, path, create_parents=True, **kwargs):
        """
        Create directory entry at path

        For systems that don't have true directories, may create an for
        this instance only and not touch the real filesystem

        Parameters
        ----------
        path: str
            location
        create_parents: bool
            if True, this is equivalent to ``makedirs``
        kwargs:
            may be permissions, etc.
        """
        pass  # not necessary to implement, may not have directories

    def makedirs(self, path, exist_ok=False):
        """Recursively make directories

        Creates directory at path and any intervening required directories.
        Raises exception if, for instance, the path already exists but is a
        file.

        Parameters
        ----------
        path: str
            leaf directory name
        exist_ok: bool (False)
            If False, will error if the target already exists
        """
        pass  # not necessary to implement, may not have directories

    def rmdir(self, path):
        """Remove a directory, if empty"""
        pass  # not necessary to implement, may not have directories

    def ls(self, path, detail=True, **kwargs):
        """List objects at path.

        This should include subdirectories and files at that location. The
        difference between a file and a directory must be clear when details
        are requested.

        The specific keys, or perhaps a FileInfo class, or similar, is TBD,
        but must be consistent across implementations.
        Must include:

        - full path to the entry (without protocol)
        - size of the entry, in bytes. If the value cannot be determined, will
          be ``None``.
        - type of entry, "file", "directory" or other

        Additional information
        may be present, appropriate to the file-system, e.g., generation,
        checksum, etc.

        May use refresh=True|False to allow use of self._ls_from_cache to
        check for a saved listing and avoid calling the backend. This would be
        common where listing may be expensive.

        Parameters
        ----------
        path: str
        detail: bool
            if True, gives a list of dictionaries, where each is the same as
            the result of ``info(path)``. If False, gives a list of paths
            (str).
        kwargs: may have additional backend-specific options, such as version
            information

        Returns
        -------
        List of strings if detail is False, or list of directory information
        dicts if detail is True.
        """
        raise NotImplementedError

    def _ls_from_cache(self, path):
        """Check cache for listing

        Returns listing, if found (may be empty list for a directly that exists
        but contains nothing), None if not in cache.
        """
        parent = self._parent(path)
        try:
            return self.dircache[path.rstrip("/")]
        except KeyError:
            pass
        try:
            files = [
                f
                for f in self.dircache[parent]
                if f["name"] == path
                or (f["name"] == path.rstrip("/") and f["type"] == "directory")
            ]
            if len(files) == 0:
                # parent dir was listed but did not contain this file
                raise FileNotFoundError(path)
            return files
        except KeyError:
            pass

    def walk(self, path, maxdepth=None, topdown=True, on_error="omit", **kwargs):
        """Return all files under the given path.

        List all files, recursing into subdirectories; output is iterator-style,
        like ``os.walk()``. For a simple list of files, ``find()`` is available.

        When topdown is True, the caller can modify the dirnames list in-place (perhaps
        using del or slice assignment), and walk() will
        only recurse into the subdirectories whose names remain in dirnames;
        this can be used to prune the search, impose a specific order of visiting,
        or even to inform walk() about directories the caller creates or renames before
        it resumes walk() again.
        Modifying dirnames when topdown is False has no effect. (see os.walk)

        Note that the "files" outputted will include anything that is not
        a directory, such as links.

        Parameters
        ----------
        path: str
            Root to recurse into
        maxdepth: int
            Maximum recursion depth. None means limitless, but not recommended
            on link-based file-systems.
        topdown: bool (True)
            Whether to walk the directory tree from the top downwards or from
            the bottom upwards.
        on_error: "omit", "raise", a callable
            if omit (default), path with exception will simply be empty;
            If raise, an underlying exception will be raised;
            if callable, it will be called with a single OSError instance as argument
        kwargs: passed to ``ls``
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        path = self._strip_protocol(path)
        full_dirs = {}
        dirs = {}
        files = {}

        detail = kwargs.pop("detail", False)
        try:
            listing = self.ls(path, detail=True, **kwargs)
        except (FileNotFoundError, OSError) as e:
            if on_error == "raise":
                raise
            if callable(on_error):
                on_error(e)
            return

        for info in listing:
            # each info name must be at least [path]/part , but here
            # we check also for names like [path]/part/
            pathname = info["name"].rstrip("/")
            name = pathname.rsplit("/", 1)[-1]
            if info["type"] == "directory" and pathname != path:
                # do not include "self" path
                full_dirs[name] = pathname
                dirs[name] = info
            elif pathname == path:
                # file-like with same name as give path
                files[""] = info
            else:
                files[name] = info

        if not detail:
            dirs = list(dirs)
            files = list(files)

        if topdown:
            # Yield before recursion if walking top down
            yield path, dirs, files

        if maxdepth is not None:
            maxdepth -= 1
            if maxdepth < 1:
                if not topdown:
                    yield path, dirs, files
                return

        for d in dirs:
            yield from self.walk(
                full_dirs[d],
                maxdepth=maxdepth,
                detail=detail,
                topdown=topdown,
                **kwargs,
            )

        if not topdown:
            # Yield after recursion if walking bottom up
            yield path, dirs, files

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        """List all files below path.

        Like posix ``find`` command without conditions

        Parameters
        ----------
        path : str
        maxdepth: int or None
            If not None, the maximum number of levels to descend
        withdirs: bool
            Whether to include directory paths in the output. This is True
            when used by glob, but users usually only want files.
        kwargs are passed to ``ls``.
        """
        # TODO: allow equivalent of -name parameter
        path = self._strip_protocol(path)
        out = {}

        # Add the root directory if withdirs is requested
        # This is needed for posix glob compliance
        if withdirs and path != "" and self.isdir(path):
            out[path] = self.info(path)

        for _, dirs, files in self.walk(path, maxdepth, detail=True, **kwargs):
            if withdirs:
                files.update(dirs)
            out.update({info["name"]: info for name, info in files.items()})
        if not out and self.isfile(path):
            # walk works on directories, but find should also return [path]
            # when path happens to be a file
            out[path] = {}
        names = sorted(out)
        if not detail:
            return names
        else:
            return {name: out[name] for name in names}

    def du(self, path, total=True, maxdepth=None, withdirs=False, **kwargs):
        """Space used by files and optionally directories within a path

        Directory size does not include the size of its contents.

        Parameters
        ----------
        path: str
        total: bool
            Whether to sum all the file sizes
        maxdepth: int or None
            Maximum number of directory levels to descend, None for unlimited.
        withdirs: bool
            Whether to include directory paths in the output.
        kwargs: passed to ``find``

        Returns
        -------
        Dict of {path: size} if total=False, or int otherwise, where numbers
        refer to bytes used.
        """
        sizes = {}
        if withdirs and self.isdir(path):
            # Include top-level directory in output
            info = self.info(path)
            sizes[info["name"]] = info["size"]
        for f in self.find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs):
            info = self.info(f)
            sizes[info["name"]] = info["size"]
        if total:
            return sum(sizes.values())
        else:
            return sizes

    def glob(self, path, maxdepth=None, **kwargs):
        """Find files by glob-matching.

        Pattern matching capabilities for finding files that match the given pattern.

        Parameters
        ----------
        path: str
            The glob pattern to match against
        maxdepth: int or None
            Maximum depth for ``'**'`` patterns. Applied on the first ``'**'`` found.
            Must be at least 1 if provided.
        kwargs:
            Additional arguments passed to ``find`` (e.g., detail=True)

        Returns
        -------
        List of matched paths, or dict of paths and their info if detail=True

        Notes
        -----
        Supported patterns:
        - '*': Matches any sequence of characters within a single directory level
        - ``'**'``: Matches any number of directory levels (must be an entire path component)
        - '?': Matches exactly one character
        - '[abc]': Matches any character in the set
        - '[a-z]': Matches any character in the range
        - '[!abc]': Matches any character NOT in the set

        Special behaviors:
        - If the path ends with '/', only folders are returned
        - Consecutive '*' characters are compressed into a single '*'
        - Empty brackets '[]' never match anything
        - Negated empty brackets '[!]' match any single character
        - Special characters in character classes are escaped properly

        Limitations:
        - ``'**'`` must be a complete path component (e.g., ``'a/**/b'``, not ``'a**b'``)
        - No brace expansion ('{a,b}.txt')
        - No extended glob patterns ('+(pattern)', '!(pattern)')
        """
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        import re

        seps = (os.path.sep, os.path.altsep) if os.path.altsep else (os.path.sep,)
        ends_with_sep = path.endswith(seps)  # _strip_protocol strips trailing slash
        path = self._strip_protocol(path)
        append_slash_to_dirname = ends_with_sep or path.endswith(
            tuple(sep + "**" for sep in seps)
        )
        idx_star = path.find("*") if path.find("*") >= 0 else len(path)
        idx_qmark = path.find("?") if path.find("?") >= 0 else len(path)
        idx_brace = path.find("[") if path.find("[") >= 0 else len(path)

        min_idx = min(idx_star, idx_qmark, idx_brace)

        detail = kwargs.pop("detail", False)

        if not has_magic(path):
            if self.exists(path, **kwargs):
                if not detail:
                    return [path]
                else:
                    return {path: self.info(path, **kwargs)}
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

        allpaths = self.find(root, maxdepth=depth, withdirs=True, detail=True, **kwargs)

        pattern = glob_translate(path + ("/" if ends_with_sep else ""))
        pattern = re.compile(pattern)

        out = {
            p: info
            for p, info in sorted(allpaths.items())
            if pattern.match(
                p + "/"
                if append_slash_to_dirname and info["type"] == "directory"
                else p
            )
        }

        if detail:
            return out
        else:
            return list(out)

    def exists(self, path, **kwargs):
        """Is there a file at the given path"""
        try:
            self.info(path, **kwargs)
            return True
        except:  # noqa: E722
            # any exception allowed bar FileNotFoundError?
            return False

    def lexists(self, path, **kwargs):
        """If there is a file at the given path (including
        broken links)"""
        return self.exists(path)

    def info(self, path, **kwargs):
        """Give details of entry at path

        Returns a single dictionary, with exactly the same information as ``ls``
        would with ``detail=True``.

        The default implementation calls ls and could be overridden by a
        shortcut. kwargs are passed on to ```ls()``.

        Some file systems might not be able to measure the file's size, in
        which case, the returned dict will include ``'size': None``.

        Returns
        -------
        dict with keys: name (full path in the FS), size (in bytes), type (file,
        directory, or something else) and other FS-specific keys.
        """
        path = self._strip_protocol(path)
        out = self.ls(self._parent(path), detail=True, **kwargs)
        out = [o for o in out if o["name"].rstrip("/") == path]
        if out:
            return out[0]
        out = self.ls(path, detail=True, **kwargs)
        path = path.rstrip("/")
        out1 = [o for o in out if o["name"].rstrip("/") == path]
        if len(out1) == 1:
            if "size" not in out1[0]:
                out1[0]["size"] = None
            return out1[0]
        elif len(out1) > 1 or out:
            return {"name": path, "size": 0, "type": "directory"}
        else:
            raise FileNotFoundError(path)

    def checksum(self, path):
        """Unique value for current version of file

        If the checksum is the same from one moment to another, the contents
        are guaranteed to be the same. If the checksum changes, the contents
        *might* have changed.

        This should normally be overridden; default will probably capture
        creation/modification timestamp (which would be good) or maybe
        access timestamp (which would be bad)
        """
        return int(tokenize(self.info(path)), 16)

    def size(self, path):
        """Size in bytes of file"""
        return self.info(path).get("size", None)

    def sizes(self, paths):
        """Size in bytes of each file in a list of paths"""
        return [self.size(p) for p in paths]

    def isdir(self, path):
        """Is this entry directory-like?"""
        try:
            return self.info(path)["type"] == "directory"
        except OSError:
            return False

    def isfile(self, path):
        """Is this entry file-like?"""
        try:
            return self.info(path)["type"] == "file"
        except:  # noqa: E722
            return False

    def read_text(self, path, encoding=None, errors=None, newline=None, **kwargs):
        """Get the contents of the file as a string.

        Parameters
        ----------
        path: str
            URL of file on this filesystems
        encoding, errors, newline: same as `open`.
        """
        with self.open(
            path,
            mode="r",
            encoding=encoding,
            errors=errors,
            newline=newline,
            **kwargs,
        ) as f:
            return f.read()

    def write_text(
        self, path, value, encoding=None, errors=None, newline=None, **kwargs
    ):
        """Write the text to the given file.

        An existing file will be overwritten.

        Parameters
        ----------
        path: str
            URL of file on this filesystems
        value: str
            Text to write.
        encoding, errors, newline: same as `open`.
        """
        with self.open(
            path,
            mode="w",
            encoding=encoding,
            errors=errors,
            newline=newline,
            **kwargs,
        ) as f:
            return f.write(value)

    def cat_file(self, path, start=None, end=None, **kwargs):
        """Get the content of a file

        Parameters
        ----------
        path: URL of file on this filesystems
        start, end: int
            Bytes limits of the read. If negative, backwards from end,
            like usual python slices. Either can be None for start or
            end of file, respectively
        kwargs: passed to ``open()``.
        """
        # explicitly set buffering off?
        with self.open(path, "rb", **kwargs) as f:
            if start is not None:
                if start >= 0:
                    f.seek(start)
                else:
                    f.seek(max(0, f.size + start))
            if end is not None:
                if end < 0:
                    end = f.size + end
                return f.read(end - f.tell())
            return f.read()

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        """Set the bytes of given file"""
        if mode == "create" and self.exists(path):
            # non-atomic but simple way; or could use "xb" in open(), which is likely
            # not as well supported
            raise FileExistsError
        with self.open(path, "wb", **kwargs) as f:
            f.write(value)

    def pipe(self, path, value=None, **kwargs):
        """Put value into path

        (counterpart to ``cat``)

        Parameters
        ----------
        path: string or dict(str, bytes)
            If a string, a single remote location to put ``value`` bytes; if a dict,
            a mapping of {path: bytesvalue}.
        value: bytes, optional
            If using a single path, these are the bytes to put there. Ignored if
            ``path`` is a dict
        """
        if isinstance(path, str):
            self.pipe_file(self._strip_protocol(path), value, **kwargs)
        elif isinstance(path, dict):
            for k, v in path.items():
                self.pipe_file(self._strip_protocol(k), v, **kwargs)
        else:
            raise ValueError("path must be str or dict")

    def cat_ranges(
        self, paths, starts, ends, max_gap=None, on_error="return", **kwargs
    ):
        """Get the contents of byte ranges from one or more files

        Parameters
        ----------
        paths: list
            A list of of filepaths on this filesystems
        starts, ends: int or list
            Bytes limits of the read. If using a single int, the same value will be
            used to read all the specified files.
        """
        if max_gap is not None:
            raise NotImplementedError
        if not isinstance(paths, list):
            raise TypeError
        if not isinstance(starts, list):
            starts = [starts] * len(paths)
        if not isinstance(ends, list):
            ends = [ends] * len(paths)
        if len(starts) != len(paths) or len(ends) != len(paths):
            raise ValueError
        out = []
        for p, s, e in zip(paths, starts, ends):
            try:
                out.append(self.cat_file(p, s, e))
            except Exception as e:
                if on_error == "return":
                    out.append(e)
                else:
                    raise
        return out

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        """Fetch (potentially multiple) paths' contents

        Parameters
        ----------
        recursive: bool
            If True, assume the path(s) are directories, and get all the
            contained files
        on_error : "raise", "omit", "return"
            If raise, an underlying exception will be raised (converted to KeyError
            if the type is in self.missing_exceptions); if omit, keys with exception
            will simply not be included in the output; if "return", all keys are
            included in the output, but the value will be bytes or an exception
            instance.
        kwargs: passed to cat_file

        Returns
        -------
        dict of {path: contents} if there are multiple paths
        or the path has been otherwise expanded
        """
        paths = self.expand_path(path, recursive=recursive, **kwargs)
        if (
            len(paths) > 1
            or isinstance(path, list)
            or paths[0] != self._strip_protocol(path)
        ):
            out = {}
            for path in paths:
                try:
                    out[path] = self.cat_file(path, **kwargs)
                except Exception as e:
                    if on_error == "raise":
                        raise
                    if on_error == "return":
                        out[path] = e
            return out
        else:
            return self.cat_file(paths[0], **kwargs)

    def get_file(self, rpath, lpath, callback=DEFAULT_CALLBACK, outfile=None, **kwargs):
        """Copy single remote file to local"""
        from .implementations.local import LocalFileSystem

        if isfilelike(lpath):
            outfile = lpath
        elif self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return None

        fs = LocalFileSystem(auto_mkdir=True)
        fs.makedirs(fs._parent(lpath), exist_ok=True)

        with self.open(rpath, "rb", **kwargs) as f1:
            if outfile is None:
                outfile = open(lpath, "wb")

            try:
                callback.set_size(getattr(f1, "size", None))
                data = True
                while data:
                    data = f1.read(self.blocksize)
                    segment_len = outfile.write(data)
                    if segment_len is None:
                        segment_len = len(data)
                    callback.relative_update(segment_len)
            finally:
                if not isfilelike(lpath):
                    outfile.close()

    def get(
        self,
        rpath,
        lpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        maxdepth=None,
        **kwargs,
    ):
        """Copy file(s) to local.

        Copies a specific file or tree of files (if recursive=True). If lpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within. Can submit a list of paths, which may be glob-patterns
        and will be expanded.

        Calls get_file for each source.
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            rpaths = rpath
            lpaths = lpath
        else:
            from .implementations.local import (
                LocalFileSystem,
                make_path_posix,
                trailing_sep,
            )

            source_is_str = isinstance(rpath, str)
            rpaths = self.expand_path(
                rpath, recursive=recursive, maxdepth=maxdepth, **kwargs
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                rpaths = [p for p in rpaths if not (trailing_sep(p) or self.isdir(p))]
                if not rpaths:
                    return

            if isinstance(lpath, str):
                lpath = make_path_posix(lpath)

            source_is_file = len(rpaths) == 1
            dest_is_dir = isinstance(lpath, str) and (
                trailing_sep(lpath) or LocalFileSystem().isdir(lpath)
            )

            exists = source_is_str and (
                (has_magic(rpath) and source_is_file)
                or (not has_magic(rpath) and dest_is_dir and not trailing_sep(rpath))
            )
            lpaths = other_paths(
                rpaths,
                lpath,
                exists=exists,
                flatten=not source_is_str,
            )

        callback.set_size(len(lpaths))
        for lpath, rpath in callback.wrap(zip(lpaths, rpaths)):
            with callback.branched(rpath, lpath) as child:
                self.get_file(rpath, lpath, callback=child, **kwargs)

    def put_file(
        self, lpath, rpath, callback=DEFAULT_CALLBACK, mode="overwrite", **kwargs
    ):
        """Copy single file to remote"""
        if mode == "create" and self.exists(rpath):
            raise FileExistsError
        if os.path.isdir(lpath):
            self.makedirs(rpath, exist_ok=True)
            return None

        with open(lpath, "rb") as f1:
            size = f1.seek(0, 2)
            callback.set_size(size)
            f1.seek(0)

            self.mkdirs(self._parent(os.fspath(rpath)), exist_ok=True)
            with self.open(rpath, "wb", **kwargs) as f2:
                while f1.tell() < size:
                    data = f1.read(self.blocksize)
                    segment_len = f2.write(data)
                    if segment_len is None:
                        segment_len = len(data)
                    callback.relative_update(segment_len)

    def put(
        self,
        lpath,
        rpath,
        recursive=False,
        callback=DEFAULT_CALLBACK,
        maxdepth=None,
        **kwargs,
    ):
        """Copy file(s) from local.

        Copies a specific file or tree of files (if recursive=True). If rpath
        ends with a "/", it will be assumed to be a directory, and target files
        will go within.

        Calls put_file for each source.
        """
        if isinstance(lpath, list) and isinstance(rpath, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            rpaths = rpath
            lpaths = lpath
        else:
            from .implementations.local import (
                LocalFileSystem,
                make_path_posix,
                trailing_sep,
            )

            source_is_str = isinstance(lpath, str)
            if source_is_str:
                lpath = make_path_posix(lpath)
            fs = LocalFileSystem()
            lpaths = fs.expand_path(
                lpath, recursive=recursive, maxdepth=maxdepth, **kwargs
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                lpaths = [p for p in lpaths if not (trailing_sep(p) or fs.isdir(p))]
                if not lpaths:
                    return

            source_is_file = len(lpaths) == 1
            dest_is_dir = isinstance(rpath, str) and (
                trailing_sep(rpath) or self.isdir(rpath)
            )

            rpath = (
                self._strip_protocol(rpath)
                if isinstance(rpath, str)
                else [self._strip_protocol(p) for p in rpath]
            )
            exists = source_is_str and (
                (has_magic(lpath) and source_is_file)
                or (not has_magic(lpath) and dest_is_dir and not trailing_sep(lpath))
            )
            rpaths = other_paths(
                lpaths,
                rpath,
                exists=exists,
                flatten=not source_is_str,
            )

        callback.set_size(len(rpaths))
        for lpath, rpath in callback.wrap(zip(lpaths, rpaths)):
            with callback.branched(lpath, rpath) as child:
                self.put_file(lpath, rpath, callback=child, **kwargs)

    def head(self, path, size=1024):
        """Get the first ``size`` bytes from file"""
        with self.open(path, "rb") as f:
            return f.read(size)

    def tail(self, path, size=1024):
        """Get the last ``size`` bytes from file"""
        with self.open(path, "rb") as f:
            f.seek(max(-size, -f.size), 2)
            return f.read()

    def cp_file(self, path1, path2, **kwargs):
        raise NotImplementedError

    def copy(
        self, path1, path2, recursive=False, maxdepth=None, on_error=None, **kwargs
    ):
        """Copy within two locations in the filesystem

        on_error : "raise", "ignore"
            If raise, any not-found exceptions will be raised; if ignore any
            not-found exceptions will cause the path to be skipped; defaults to
            raise unless recursive is true, where the default is ignore
        """
        if on_error is None and recursive:
            on_error = "ignore"
        elif on_error is None:
            on_error = "raise"

        if isinstance(path1, list) and isinstance(path2, list):
            # No need to expand paths when both source and destination
            # are provided as lists
            paths1 = path1
            paths2 = path2
        else:
            from .implementations.local import trailing_sep

            source_is_str = isinstance(path1, str)
            paths1 = self.expand_path(
                path1, recursive=recursive, maxdepth=maxdepth, **kwargs
            )
            if source_is_str and (not recursive or maxdepth is not None):
                # Non-recursive glob does not copy directories
                paths1 = [p for p in paths1 if not (trailing_sep(p) or self.isdir(p))]
                if not paths1:
                    return

            source_is_file = len(paths1) == 1
            dest_is_dir = isinstance(path2, str) and (
                trailing_sep(path2) or self.isdir(path2)
            )

            exists = source_is_str and (
                (has_magic(path1) and source_is_file)
                or (not has_magic(path1) and dest_is_dir and not trailing_sep(path1))
            )
            paths2 = other_paths(
                paths1,
                path2,
                exists=exists,
                flatten=not source_is_str,
            )

        for p1, p2 in zip(paths1, paths2):
            try:
                self.cp_file(p1, p2, **kwargs)
            except FileNotFoundError:
                if on_error == "raise":
                    raise

    def expand_path(self, path, recursive=False, maxdepth=None, **kwargs):
        """Turn one or more globs or directories into a list of all matching paths
        to files or directories.

        kwargs are passed to ``glob`` or ``find``, which may in turn call ``ls``
        """

        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        if isinstance(path, (str, os.PathLike)):
            out = self.expand_path([path], recursive, maxdepth, **kwargs)
        else:
            out = set()
            path = [self._strip_protocol(p) for p in path]
            for p in path:
                if has_magic(p):
                    bit = set(self.glob(p, maxdepth=maxdepth, **kwargs))
                    out |= bit
                    if recursive:
                        # glob call above expanded one depth so if maxdepth is defined
                        # then decrement it in expand_path call below. If it is zero
                        # after decrementing then avoid expand_path call.
                        if maxdepth is not None and maxdepth <= 1:
                            continue
                        out |= set(
                            self.expand_path(
                                list(bit),
                                recursive=recursive,
                                maxdepth=maxdepth - 1 if maxdepth is not None else None,
                                **kwargs,
                            )
                        )
                    continue
                elif recursive:
                    rec = set(
                        self.find(
                            p, maxdepth=maxdepth, withdirs=True, detail=False, **kwargs
                        )
                    )
                    out |= rec
                if p not in out and (recursive is False or self.exists(p)):
                    # should only check once, for the root
                    out.add(p)
        if not out:
            raise FileNotFoundError(path)
        return sorted(out)

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        """Move file(s) from one location to another"""
        if path1 == path2:
            logger.debug("%s mv: The paths are the same, so no files were moved.", self)
        else:
            # explicitly raise exception to prevent data corruption
            self.copy(
                path1, path2, recursive=recursive, maxdepth=maxdepth, onerror="raise"
            )
            self.rm(path1, recursive=recursive)

    def rm_file(self, path):
        """Delete a file"""
        self._rm(path)

    def _rm(self, path):
        """Delete one file"""
        # this is the old name for the method, prefer rm_file
        raise NotImplementedError

    def rm(self, path, recursive=False, maxdepth=None):
        """Delete files.

        Parameters
        ----------
        path: str or list of str
            File(s) to delete.
        recursive: bool
            If file(s) are directories, recursively delete contents and then
            also remove the directory
        maxdepth: int or None
            Depth to pass to walk for finding files to delete, if recursive.
            If None, there will be no limit and infinite recursion may be
            possible.
        """
        path = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(path):
            self.rm_file(p)

    @classmethod
    def _parent(cls, path):
        path = cls._strip_protocol(path)
        if "/" in path:
            parent = path.rsplit("/", 1)[0].lstrip(cls.root_marker)
            return cls.root_marker + parent
        else:
            return cls.root_marker

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        """Return raw bytes-mode file-like from the file-system"""
        return AbstractBufferedFile(
            self,
            path,
            mode,
            block_size,
            autocommit,
            cache_options=cache_options,
            **kwargs,
        )

    def open(
        self,
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        compression=None,
        **kwargs,
    ):
        """
        Return a file-like object from the filesystem

        The resultant instance must function correctly in a context ``with``
        block.

        Parameters
        ----------
        path: str
            Target file
        mode: str like 'rb', 'w'
            See builtin ``open()``
            Mode "x" (exclusive write) may be implemented by the backend. Even if
            it is, whether  it is checked up front or on commit, and whether it is
            atomic is implementation-dependent.
        block_size: int
            Some indication of buffering - this is a value in bytes
        cache_options : dict, optional
            Extra arguments to pass through to the cache.
        compression: string or None
            If given, open file using compression codec. Can either be a compression
            name (a key in ``fsspec.compression.compr``) or "infer" to guess the
            compression from the filename suffix.
        encoding, errors, newline: passed on to TextIOWrapper for text mode
        """
        import io

        path = self._strip_protocol(path)
        if "b" not in mode:
            mode = mode.replace("t", "") + "b"

            text_kwargs = {
                k: kwargs.pop(k)
                for k in ["encoding", "errors", "newline"]
                if k in kwargs
            }
            return io.TextIOWrapper(
                self.open(
                    path,
                    mode,
                    block_size=block_size,
                    cache_options=cache_options,
                    compression=compression,
                    **kwargs,
                ),
                **text_kwargs,
            )
        else:
            ac = kwargs.pop("autocommit", not self._intrans)
            f = self._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=ac,
                cache_options=cache_options,
                **kwargs,
            )
            if compression is not None:
                from fsspec.compression import compr
                from fsspec.core import get_compression

                compression = get_compression(path, compression)
                compress = compr[compression]
                f = compress(f, mode=mode[0])

            if not ac and "r" not in mode:
                self.transaction.files.append(f)
            return f

    def touch(self, path, truncate=True, **kwargs):
        """Create empty file, or update timestamp

        Parameters
        ----------
        path: str
            file location
        truncate: bool
            If True, always set file size to 0; if False, update timestamp and
            leave file unchanged, if backend allows this
        """
        if truncate or not self.exists(path):
            with self.open(path, "wb", **kwargs):
                pass
        else:
            raise NotImplementedError  # update timestamp, if possible

    def ukey(self, path):
        """Hash of file properties, to tell if it has changed"""
        return sha256(str(self.info(path)).encode()).hexdigest()

    def read_block(self, fn, offset, length, delimiter=None):
        """Read a block of bytes from

        Starting at ``offset`` of the file, read ``length`` bytes.  If
        ``delimiter`` is set then we ensure that the read starts and stops at
        delimiter boundaries that follow the locations ``offset`` and ``offset
        + length``.  If ``offset`` is zero then we start at zero.  The
        bytestring returned WILL include the end delimiter string.

        If offset+length is beyond the eof, reads to eof.

        Parameters
        ----------
        fn: string
            Path to filename
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read. If None, read to end.
        delimiter: bytes (optional)
            Ensure reading starts and stops at delimiter bytestring

        Examples
        --------
        >>> fs.read_block('data/file.csv', 0, 13)  # doctest: +SKIP
        b'Alice, 100\\nBo'
        >>> fs.read_block('data/file.csv', 0, 13, delimiter=b'\\n')  # doctest: +SKIP
        b'Alice, 100\\nBob, 200\\n'

        Use ``length=None`` to read to the end of the file.
        >>> fs.read_block('data/file.csv', 0, None, delimiter=b'\\n')  # doctest: +SKIP
        b'Alice, 100\\nBob, 200\\nCharlie, 300'

        See Also
        --------
        :func:`fsspec.utils.read_block`
        """
        with self.open(fn, "rb") as f:
            size = f.size
            if length is None:
                length = size
            if size is not None and offset + length > size:
                length = size - offset
            return read_block(f, offset, length, delimiter)

    def to_json(self, *, include_password: bool = True) -> str:
        """
        JSON representation of this filesystem instance.

        Parameters
        ----------
        include_password: bool, default True
            Whether to include the password (if any) in the output.

        Returns
        -------
        JSON string with keys ``cls`` (the python location of this class),
        protocol (text name of this class's protocol, first one in case of
        multiple), ``args`` (positional args, usually empty), and all other
        keyword arguments as their own keys.

        Warnings
        --------
        Serialized filesystems may contain sensitive information which have been
        passed to the constructor, such as passwords and tokens. Make sure you
        store and send them in a secure environment!
        """
        from .json import FilesystemJSONEncoder

        return json.dumps(
            self,
            cls=type(
                "_FilesystemJSONEncoder",
                (FilesystemJSONEncoder,),
                {"include_password": include_password},
            ),
        )

    @staticmethod
    def from_json(blob: str) -> AbstractFileSystem:
        """
        Recreate a filesystem instance from JSON representation.

        See ``.to_json()`` for the expected structure of the input.

        Parameters
        ----------
        blob: str

        Returns
        -------
        file system instance, not necessarily of this particular class.

        Warnings
        --------
        This can import arbitrary modules (as determined by the ``cls`` key).
        Make sure you haven't installed any modules that may execute malicious code
        at import time.
        """
        from .json import FilesystemJSONDecoder

        return json.loads(blob, cls=FilesystemJSONDecoder)

    def to_dict(self, *, include_password: bool = True) -> dict[str, Any]:
        """
        JSON-serializable dictionary representation of this filesystem instance.

        Parameters
        ----------
        include_password: bool, default True
            Whether to include the password (if any) in the output.

        Returns
        -------
        Dictionary with keys ``cls`` (the python location of this class),
        protocol (text name of this class's protocol, first one in case of
        multiple), ``args`` (positional args, usually empty), and all other
        keyword arguments as their own keys.

        Warnings
        --------
        Serialized filesystems may contain sensitive information which have been
        passed to the constructor, such as passwords and tokens. Make sure you
        store and send them in a secure environment!
        """
        from .json import FilesystemJSONEncoder

        json_encoder = FilesystemJSONEncoder()

        cls = type(self)
        proto = self.protocol

        storage_options = dict(self.storage_options)
        if not include_password:
            storage_options.pop("password", None)

        return dict(
            cls=f"{cls.__module__}:{cls.__name__}",
            protocol=proto[0] if isinstance(proto, (tuple, list)) else proto,
            args=json_encoder.make_serializable(self.storage_args),
            **json_encoder.make_serializable(storage_options),
        )

    @staticmethod
    def from_dict(dct: dict[str, Any]) -> AbstractFileSystem:
        """
        Recreate a filesystem instance from dictionary representation.

        See ``.to_dict()`` for the expected structure of the input.

        Parameters
        ----------
        dct: Dict[str, Any]

        Returns
        -------
        file system instance, not necessarily of this particular class.

        Warnings
        --------
        This can import arbitrary modules (as determined by the ``cls`` key).
        Make sure you haven't installed any modules that may execute malicious code
        at import time.
        """
        from .json import FilesystemJSONDecoder

        json_decoder = FilesystemJSONDecoder()

        dct = dict(dct)  # Defensive copy

        cls = FilesystemJSONDecoder.try_resolve_fs_cls(dct)
        if cls is None:
            raise ValueError("Not a serialized AbstractFileSystem")

        dct.pop("cls", None)
        dct.pop("protocol", None)

        return cls(
            *json_decoder.unmake_serializable(dct.pop("args", ())),
            **json_decoder.unmake_serializable(dct),
        )

    def _get_pyarrow_filesystem(self):
        """
        Make a version of the FS instance which will be acceptable to pyarrow
        """
        # all instances already also derive from pyarrow
        return self

    def get_mapper(self, root="", check=False, create=False, missing_exceptions=None):
        """Create key/value store based on this file-system

        Makes a MutableMapping interface to the FS at the given root path.
        See ``fsspec.mapping.FSMap`` for further details.
        """
        from .mapping import FSMap

        return FSMap(
            root,
            self,
            check=check,
            create=create,
            missing_exceptions=missing_exceptions,
        )

    @classmethod
    def clear_instance_cache(cls):
        """
        Clear the cache of filesystem instances.

        Notes
        -----
        Unless overridden by setting the ``cachable`` class attribute to False,
        the filesystem class stores a reference to newly created instances. This
        prevents Python's normal rules around garbage collection from working,
        since the instances refcount will not drop to zero until
        ``clear_instance_cache`` is called.
        """
        cls._cache.clear()

    def created(self, path):
        """Return the created timestamp of a file as a datetime.datetime"""
        raise NotImplementedError

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        raise NotImplementedError

    def tree(
        self,
        path: str = "/",
        recursion_limit: int = 2,
        max_display: int = 25,
        display_size: bool = False,
        prefix: str = "",
        is_last: bool = True,
        first: bool = True,
        indent_size: int = 4,
    ) -> str:
        """
        Return a tree-like structure of the filesystem starting from the given path as a string.

        Parameters
        ----------
            path: Root path to start traversal from
            recursion_limit: Maximum depth of directory traversal
            max_display: Maximum number of items to display per directory
            display_size: Whether to display file sizes
            prefix: Current line prefix for visual tree structure
            is_last: Whether current item is last in its level
            first: Whether this is the first call (displays root path)
            indent_size: Number of spaces by indent

        Returns
        -------
            str: A string representing the tree structure.

        Example
        -------
            >>> from fsspec import filesystem

            >>> fs = filesystem('ftp', host='test.rebex.net', user='demo', password='password')
            >>> tree = fs.tree(display_size=True, recursion_limit=3, indent_size=8, max_display=10)
            >>> print(tree)
        """

        def format_bytes(n: int) -> str:
            """Format bytes as text."""
            for prefix, k in (
                ("P", 2**50),
                ("T", 2**40),
                ("G", 2**30),
                ("M", 2**20),
                ("k", 2**10),
            ):
                if n >= 0.9 * k:
                    return f"{n / k:.2f} {prefix}b"
            return f"{n}B"

        result = []

        if first:
            result.append(path)

        if recursion_limit:
            indent = " " * indent_size
            contents = self.ls(path, detail=True)
            contents.sort(
                key=lambda x: (x.get("type") != "directory", x.get("name", ""))
            )

            if max_display is not None and len(contents) > max_display:
                displayed_contents = contents[:max_display]
                remaining_count = len(contents) - max_display
            else:
                displayed_contents = contents
                remaining_count = 0

            for i, item in enumerate(displayed_contents):
                is_last_item = (i == len(displayed_contents) - 1) and (
                    remaining_count == 0
                )

                branch = (
                    "└" + ("─" * (indent_size - 2))
                    if is_last_item
                    else "├" + ("─" * (indent_size - 2))
                )
                branch += " "
                new_prefix = prefix + (
                    indent if is_last_item else "│" + " " * (indent_size - 1)
                )

                name = os.path.basename(item.get("name", ""))

                if display_size and item.get("type") == "directory":
                    sub_contents = self.ls(item.get("name", ""), detail=True)
                    num_files = sum(
                        1 for sub_item in sub_contents if sub_item.get("type") == "file"
                    )
                    num_folders = sum(
                        1
                        for sub_item in sub_contents
                        if sub_item.get("type") == "directory"
                    )

                    if num_files == 0 and num_folders == 0:
                        size = " (empty folder)"
                    elif num_files == 0:
                        size = f" ({num_folders} subfolder{'s' if num_folders > 1 else ''})"
                    elif num_folders == 0:
                        size = f" ({num_files} file{'s' if num_files > 1 else ''})"
                    else:
                        size = f" ({num_files} file{'s' if num_files > 1 else ''}, {num_folders} subfolder{'s' if num_folders > 1 else ''})"
                elif display_size and item.get("type") == "file":
                    size = f" ({format_bytes(item.get('size', 0))})"
                else:
                    size = ""

                result.append(f"{prefix}{branch}{name}{size}")

                if item.get("type") == "directory" and recursion_limit > 0:
                    result.append(
                        self.tree(
                            path=item.get("name", ""),
                            recursion_limit=recursion_limit - 1,
                            max_display=max_display,
                            display_size=display_size,
                            prefix=new_prefix,
                            is_last=is_last_item,
                            first=False,
                            indent_size=indent_size,
                        )
                    )

            if remaining_count > 0:
                more_message = f"{remaining_count} more item(s) not displayed."
                result.append(
                    f"{prefix}{'└' + ('─' * (indent_size - 2))} {more_message}"
                )

        return "\n".join(_ for _ in result if _)

    # ------------------------------------------------------------------------
    # Aliases

    def read_bytes(self, path, start=None, end=None, **kwargs):
        """Alias of `AbstractFileSystem.cat_file`."""
        return self.cat_file(path, start=start, end=end, **kwargs)

    def write_bytes(self, path, value, **kwargs):
        """Alias of `AbstractFileSystem.pipe_file`."""
        self.pipe_file(path, value, **kwargs)

    def makedir(self, path, create_parents=True, **kwargs):
        """Alias of `AbstractFileSystem.mkdir`."""
        return self.mkdir(path, create_parents=create_parents, **kwargs)

    def mkdirs(self, path, exist_ok=False):
        """Alias of `AbstractFileSystem.makedirs`."""
        return self.makedirs(path, exist_ok=exist_ok)

    def listdir(self, path, detail=True, **kwargs):
        """Alias of `AbstractFileSystem.ls`."""
        return self.ls(path, detail=detail, **kwargs)

    def cp(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.copy`."""
        return self.copy(path1, path2, **kwargs)

    def move(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.mv`."""
        return self.mv(path1, path2, **kwargs)

    def stat(self, path, **kwargs):
        """Alias of `AbstractFileSystem.info`."""
        return self.info(path, **kwargs)

    def disk_usage(self, path, total=True, maxdepth=None, **kwargs):
        """Alias of `AbstractFileSystem.du`."""
        return self.du(path, total=total, maxdepth=maxdepth, **kwargs)

    def rename(self, path1, path2, **kwargs):
        """Alias of `AbstractFileSystem.mv`."""
        return self.mv(path1, path2, **kwargs)

    def delete(self, path, recursive=False, maxdepth=None):
        """Alias of `AbstractFileSystem.rm`."""
        return self.rm(path, recursive=recursive, maxdepth=maxdepth)

    def upload(self, lpath, rpath, recursive=False, **kwargs):
        """Alias of `AbstractFileSystem.put`."""
        return self.put(lpath, rpath, recursive=recursive, **kwargs)

    def download(self, rpath, lpath, recursive=False, **kwargs):
        """Alias of `AbstractFileSystem.get`."""
        return self.get(rpath, lpath, recursive=recursive, **kwargs)

    def sign(self, path, expiration=100, **kwargs):
        """Create a signed URL representing the given path

        Some implementations allow temporary URLs to be generated, as a
        way of delegating credentials.

        Parameters
        ----------
        path : str
             The path on the filesystem
        expiration : int
            Number of seconds to enable the URL for (if supported)

        Returns
        -------
        URL : str
            The signed URL

        Raises
        ------
        NotImplementedError : if method is not implemented for a filesystem
        """
        raise NotImplementedError("Sign is not implemented for this filesystem")

    def _isfilestore(self):
        # Originally inherited from pyarrow DaskFileSystem. Keeping this
        # here for backwards compatibility as long as pyarrow uses its
        # legacy fsspec-compatible filesystems and thus accepts fsspec
        # filesystems as well
        return False


class AbstractBufferedFile(io.IOBase):
    """Convenient class to derive from to provide buffering

    In the case that the backend does not provide a pythonic file-like object
    already, this class contains much of the logic to build one. The only
    methods that need to be overridden are ``_upload_chunk``,
    ``_initiate_upload`` and ``_fetch_range``.
    """

    DEFAULT_BLOCK_SIZE = 5 * 2**20
    _details = None

    def __init__(
        self,
        fs,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        """
        Template for files with buffered reading and writing

        Parameters
        ----------
        fs: instance of FileSystem
        path: str
            location in file-system
        mode: str
            Normal file modes. Currently only 'wb', 'ab' or 'rb'. Some file
            systems may be read-only, and some may not support append.
        block_size: int
            Buffer size for reading or writing, 'default' for class default
        autocommit: bool
            Whether to write to final destination; may only impact what
            happens when file is being closed.
        cache_type: {"readahead", "none", "mmap", "bytes"}, default "readahead"
            Caching policy in read mode. See the definitions in ``core``.
        cache_options : dict
            Additional options passed to the constructor for the cache specified
            by `cache_type`.
        size: int
            If given and in read mode, suppressed having to look up the file size
        kwargs:
            Gets stored as self.kwargs
        """
        from .core import caches

        self.path = path
        self.fs = fs
        self.mode = mode
        self.blocksize = (
            self.DEFAULT_BLOCK_SIZE if block_size in ["default", None] else block_size
        )
        self.loc = 0
        self.autocommit = autocommit
        self.end = None
        self.start = None
        self.closed = False

        if cache_options is None:
            cache_options = {}

        if "trim" in kwargs:
            warnings.warn(
                "Passing 'trim' to control the cache behavior has been deprecated. "
                "Specify it within the 'cache_options' argument instead.",
                FutureWarning,
            )
            cache_options["trim"] = kwargs.pop("trim")

        self.kwargs = kwargs

        if mode not in {"ab", "rb", "wb", "xb"}:
            raise NotImplementedError("File mode not supported")
        if mode == "rb":
            if size is not None:
                self.size = size
            else:
                self.size = self.details["size"]
            self.cache = caches[cache_type](
                self.blocksize, self._fetch_range, self.size, **cache_options
            )
        else:
            self.buffer = io.BytesIO()
            self.offset = None
            self.forced = False
            self.location = None

    @property
    def details(self):
        if self._details is None:
            self._details = self.fs.info(self.path)
        return self._details

    @details.setter
    def details(self, value):
        self._details = value
        self.size = value["size"]

    @property
    def full_name(self):
        return _unstrip_protocol(self.path, self.fs)

    @property
    def closed(self):
        # get around this attr being read-only in IOBase
        # use getattr here, since this can be called during del
        return getattr(self, "_closed", True)

    @closed.setter
    def closed(self, c):
        self._closed = c

    def __hash__(self):
        if "w" in self.mode:
            return id(self)
        else:
            return int(tokenize(self.details), 16)

    def __eq__(self, other):
        """Files are equal if they have the same checksum, only in read mode"""
        if self is other:
            return True
        return (
            isinstance(other, type(self))
            and self.mode == "rb"
            and other.mode == "rb"
            and hash(self) == hash(other)
        )

    def commit(self):
        """Move from temp to final destination"""

    def discard(self):
        """Throw away temporary file"""

    def info(self):
        """File information about this path"""
        if self.readable():
            return self.details
        else:
            raise ValueError("Info not available while writing")

    def tell(self):
        """Current file location"""
        return self.loc

    def seek(self, loc, whence=0):
        """Set current file location

        Parameters
        ----------
        loc: int
            byte location
        whence: {0, 1, 2}
            from start of file, current location or end of file, resp.
        """
        loc = int(loc)
        if not self.mode == "rb":
            raise OSError(ESPIPE, "Seek only available in read mode")
        if whence == 0:
            nloc = loc
        elif whence == 1:
            nloc = self.loc + loc
        elif whence == 2:
            nloc = self.size + loc
        else:
            raise ValueError(f"invalid whence ({whence}, should be 0, 1 or 2)")
        if nloc < 0:
            raise ValueError("Seek before start of file")
        self.loc = nloc
        return self.loc

    def write(self, data):
        """
        Write data to buffer.

        Buffer only sent on flush() or if buffer is greater than
        or equal to blocksize.

        Parameters
        ----------
        data: bytes
            Set of bytes to be written.
        """
        if not self.writable():
            raise ValueError("File not in write mode")
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if self.forced:
            raise ValueError("This file has been force-flushed, can only close")
        out = self.buffer.write(data)
        self.loc += out
        if self.buffer.tell() >= self.blocksize:
            self.flush()
        return out

    def flush(self, force=False):
        """
        Write buffered data to backend store.

        Writes the current buffer, if it is larger than the block-size, or if
        the file is being closed.

        Parameters
        ----------
        force: bool
            When closing, write the last block even if it is smaller than
            blocks are allowed to be. Disallows further writing to this file.
        """

        if self.closed:
            raise ValueError("Flush on closed file")
        if force and self.forced:
            raise ValueError("Force flush cannot be called more than once")
        if force:
            self.forced = True

        if self.readable():
            # no-op to flush on read-mode
            return

        if not force and self.buffer.tell() < self.blocksize:
            # Defer write on small block
            return

        if self.offset is None:
            # Initialize a multipart upload
            self.offset = 0
            try:
                self._initiate_upload()
            except:
                self.closed = True
                raise

        if self._upload_chunk(final=force) is not False:
            self.offset += self.buffer.seek(0, 2)
            self.buffer = io.BytesIO()

    def _upload_chunk(self, final=False):
        """Write one part of a multi-block file upload

        Parameters
        ==========
        final: bool
            This is the last block, so should complete file, if
            self.autocommit is True.
        """
        # may not yet have been initialized, may need to call _initialize_upload

    def _initiate_upload(self):
        """Create remote file/upload"""
        pass

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        return self.fs.cat_file(self.path, start=start, end=end)

    def read(self, length=-1):
        """
        Return data from cache, or fetch pieces as necessary

        Parameters
        ----------
        length: int (-1)
            Number of bytes to read; if <0, all remaining bytes.
        """
        length = -1 if length is None else int(length)
        if self.mode != "rb":
            raise ValueError("File not in read mode")
        if length < 0:
            length = self.size - self.loc
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if length == 0:
            # don't even bother calling fetch
            return b""
        out = self.cache._fetch(self.loc, self.loc + length)

        logger.debug(
            "%s read: %i - %i %s",
            self,
            self.loc,
            self.loc + length,
            self.cache._log_stats(),
        )
        self.loc += len(out)
        return out

    def readinto(self, b):
        """mirrors builtin file's readinto method

        https://docs.python.org/3/library/io.html#io.RawIOBase.readinto
        """
        out = memoryview(b).cast("B")
        data = self.read(out.nbytes)
        out[: len(data)] = data
        return len(data)

    def readuntil(self, char=b"\n", blocks=None):
        """Return data between current position and first occurrence of char

        char is included in the output, except if the end of the tile is
        encountered first.

        Parameters
        ----------
        char: bytes
            Thing to find
        blocks: None or int
            How much to read in each go. Defaults to file blocksize - which may
            mean a new read on every call.
        """
        out = []
        while True:
            start = self.tell()
            part = self.read(blocks or self.blocksize)
            if len(part) == 0:
                break
            found = part.find(char)
            if found > -1:
                out.append(part[: found + len(char)])
                self.seek(start + found + len(char))
                break
            out.append(part)
        return b"".join(out)

    def readline(self):
        """Read until and including the first occurrence of newline character

        Note that, because of character encoding, this is not necessarily a
        true line ending.
        """
        return self.readuntil(b"\n")

    def __next__(self):
        out = self.readline()
        if out:
            return out
        raise StopIteration

    def __iter__(self):
        return self

    def readlines(self):
        """Return all data, split by the newline character, including the newline character"""
        data = self.read()
        lines = data.split(b"\n")
        out = [l + b"\n" for l in lines[:-1]]
        if data.endswith(b"\n"):
            return out
        else:
            return out + [lines[-1]]
        # return list(self)  ???

    def readinto1(self, b):
        return self.readinto(b)

    def close(self):
        """Close file

        Finalizes writes, discards cache
        """
        if getattr(self, "_unclosable", False):
            return
        if self.closed:
            return
        try:
            if self.mode == "rb":
                self.cache = None
            else:
                if not self.forced:
                    self.flush(force=True)

                if self.fs is not None:
                    self.fs.invalidate_cache(self.path)
                    self.fs.invalidate_cache(self.fs._parent(self.path))
        finally:
            self.closed = True

    def readable(self):
        """Whether opened for reading"""
        return "r" in self.mode and not self.closed

    def seekable(self):
        """Whether is seekable (only in read mode)"""
        return self.readable()

    def writable(self):
        """Whether opened for writing"""
        return self.mode in {"wb", "ab", "xb"} and not self.closed

    def __reduce__(self):
        if self.mode != "rb":
            raise RuntimeError("Pickling a writeable file is not supported")

        return reopen, (
            self.fs,
            self.path,
            self.mode,
            self.blocksize,
            self.loc,
            self.size,
            self.autocommit,
            self.cache.name if self.cache else "none",
            self.kwargs,
        )

    def __del__(self):
        if not self.closed:
            self.close()

    def __str__(self):
        return f"<File-like object {type(self.fs).__name__}, {self.path}>"

    __repr__ = __str__

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def reopen(fs, path, mode, blocksize, loc, size, autocommit, cache_type, kwargs):
    file = fs.open(
        path,
        mode=mode,
        block_size=blocksize,
        autocommit=autocommit,
        cache_type=cache_type,
        size=size,
        **kwargs,
    )
    if loc > 0:
        file.seek(loc)
    return file
