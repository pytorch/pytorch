import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Literal

import fsspec.core
from fsspec.spec import AbstractBufferedFile

try:
    import ujson as json
except ImportError:
    if not TYPE_CHECKING:
        import json

from fsspec.asyn import AsyncFileSystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.core import filesystem, open, split_protocol
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.utils import (
    isfilelike,
    merge_offset_ranges,
    other_paths,
)

logger = logging.getLogger("fsspec.reference")


class ReferenceNotReachable(RuntimeError):
    def __init__(self, reference, target, *args):
        super().__init__(*args)
        self.reference = reference
        self.target = target

    def __str__(self):
        return f'Reference "{self.reference}" failed to fetch target {self.target}'


def _first(d):
    return next(iter(d.values()))


def _prot_in_references(path, references):
    ref = references.get(path)
    if isinstance(ref, (list, tuple)) and isinstance(ref[0], str):
        return split_protocol(ref[0])[0] if ref[0] else ref[0]


def _protocol_groups(paths, references):
    if isinstance(paths, str):
        return {_prot_in_references(paths, references): [paths]}
    out = {}
    for path in paths:
        protocol = _prot_in_references(path, references)
        out.setdefault(protocol, []).append(path)
    return out


class RefsValuesView(collections.abc.ValuesView):
    def __iter__(self):
        for val in self._mapping.zmetadata.values():
            yield json.dumps(val).encode()
        yield from self._mapping._items.values()
        for field in self._mapping.listdir():
            chunk_sizes = self._mapping._get_chunk_sizes(field)
            if len(chunk_sizes) == 0:
                yield self._mapping[field + "/0"]
                continue
            yield from self._mapping._generate_all_records(field)


class RefsItemsView(collections.abc.ItemsView):
    def __iter__(self):
        return zip(self._mapping.keys(), self._mapping.values())


def ravel_multi_index(idx, sizes):
    val = 0
    mult = 1
    for i, s in zip(idx[::-1], sizes[::-1]):
        val += i * mult
        mult *= s
    return val


class LazyReferenceMapper(collections.abc.MutableMapping):
    """This interface can be used to read/write references from Parquet stores.
    It is not intended for other types of references.
    It can be used with Kerchunk's MultiZarrToZarr method to combine
    references into a parquet store.
    Examples of this use-case can be found here:
    https://fsspec.github.io/kerchunk/advanced.html?highlight=parquet#parquet-storage"""

    # import is class level to prevent numpy dep requirement for fsspec
    @property
    def np(self):
        import numpy as np

        return np

    @property
    def pd(self):
        import pandas as pd

        return pd

    def __init__(
        self,
        root,
        fs=None,
        out_root=None,
        cache_size=128,
        categorical_threshold=10,
        engine: Literal["fastparquet", "pyarrow"] = "fastparquet",
    ):
        """

        This instance will be writable, storing changes in memory until full partitions
        are accumulated or .flush() is called.

        To create an empty lazy store, use .create()

        Parameters
        ----------
        root : str
            Root of parquet store
        fs : fsspec.AbstractFileSystem
            fsspec filesystem object, default is local filesystem.
        cache_size : int, default=128
            Maximum size of LRU cache, where cache_size*record_size denotes
            the total number of references that can be loaded in memory at once.
        categorical_threshold : int
            Encode urls as pandas.Categorical to reduce memory footprint if the ratio
            of the number of unique urls to total number of refs for each variable
            is greater than or equal to this number. (default 10)
        engine: Literal["fastparquet","pyarrow"]
            Engine choice for reading parquet files. (default is "fastparquet")
        """

        self.root = root
        self.chunk_sizes = {}
        self.cat_thresh = categorical_threshold
        self.engine = engine
        self.cache_size = cache_size
        self.url = self.root + "/{field}/refs.{record}.parq"
        # TODO: derive fs from `root`
        self.fs = fsspec.filesystem("file") if fs is None else fs
        self.out_root = self.fs.unstrip_protocol(out_root or self.root)

        from importlib.util import find_spec

        if self.engine == "pyarrow" and find_spec("pyarrow") is None:
            raise ImportError("engine choice `pyarrow` is not installed.")

    def __getattr__(self, item):
        if item in ("_items", "record_size", "zmetadata"):
            self.setup()
            # avoid possible recursion if setup fails somehow
            return self.__dict__[item]
        raise AttributeError(item)

    def setup(self):
        self._items = {}
        self._items[".zmetadata"] = self.fs.cat_file(
            "/".join([self.root, ".zmetadata"])
        )
        met = json.loads(self._items[".zmetadata"])
        self.record_size = met["record_size"]
        self.zmetadata = met["metadata"]

        # Define function to open and decompress refs
        @lru_cache(maxsize=self.cache_size)
        def open_refs(field, record):
            """cached parquet file loader"""
            path = self.url.format(field=field, record=record)
            data = io.BytesIO(self.fs.cat_file(path))
            try:
                df = self.pd.read_parquet(data, engine=self.engine)
                refs = {c: df[c].to_numpy() for c in df.columns}
            except OSError:
                refs = None
            return refs

        self.open_refs = open_refs

    @staticmethod
    def create(root, storage_options=None, fs=None, record_size=10000, **kwargs):
        """Make empty parquet reference set

        First deletes the contents of the given directory, if it exists.

        Parameters
        ----------
        root: str
            Directory to contain the output; will be created
        storage_options: dict | None
            For making the filesystem to use for writing is fs is None
        fs: FileSystem | None
            Filesystem for writing
        record_size: int
            Number of references per parquet file
        kwargs: passed to __init__

        Returns
        -------
        LazyReferenceMapper instance
        """
        met = {"metadata": {}, "record_size": record_size}
        if fs is None:
            fs, root = fsspec.core.url_to_fs(root, **(storage_options or {}))
        if fs.exists(root):
            fs.rm(root, recursive=True)
        fs.makedirs(root, exist_ok=True)
        fs.pipe("/".join([root, ".zmetadata"]), json.dumps(met).encode())
        return LazyReferenceMapper(root, fs, **kwargs)

    @lru_cache
    def listdir(self):
        """List top-level directories"""
        dirs = (p.rsplit("/", 1)[0] for p in self.zmetadata if not p.startswith(".z"))
        return set(dirs)

    def ls(self, path="", detail=True):
        """Shortcut file listings"""
        path = path.rstrip("/")
        pathdash = path + "/" if path else ""
        dirnames = self.listdir()
        dirs = [
            d
            for d in dirnames
            if d.startswith(pathdash) and "/" not in d.lstrip(pathdash)
        ]
        if dirs:
            others = {
                f
                for f in chain(
                    [".zmetadata"],
                    (name for name in self.zmetadata),
                    (name for name in self._items),
                )
                if f.startswith(pathdash) and "/" not in f.lstrip(pathdash)
            }
            if detail is False:
                others.update(dirs)
                return sorted(others)
            dirinfo = [{"name": name, "type": "directory", "size": 0} for name in dirs]
            fileinfo = [
                {
                    "name": name,
                    "type": "file",
                    "size": len(
                        json.dumps(self.zmetadata[name])
                        if name in self.zmetadata
                        else self._items[name]
                    ),
                }
                for name in others
            ]
            return sorted(dirinfo + fileinfo, key=lambda s: s["name"])
        field = path
        others = set(
            [name for name in self.zmetadata if name.startswith(f"{path}/")]
            + [name for name in self._items if name.startswith(f"{path}/")]
        )
        fileinfo = [
            {
                "name": name,
                "type": "file",
                "size": len(
                    json.dumps(self.zmetadata[name])
                    if name in self.zmetadata
                    else self._items[name]
                ),
            }
            for name in others
        ]
        keys = self._keys_in_field(field)

        if detail is False:
            return list(others) + list(keys)
        recs = self._generate_all_records(field)
        recinfo = [
            {"name": name, "type": "file", "size": rec[-1]}
            for name, rec in zip(keys, recs)
            if rec[0]  # filters out path==None, deleted/missing
        ]
        return fileinfo + recinfo

    def _load_one_key(self, key):
        """Get the reference for one key

        Returns bytes, one-element list or three-element list.
        """
        if key in self._items:
            return self._items[key]
        elif key in self.zmetadata:
            return json.dumps(self.zmetadata[key]).encode()
        elif "/" not in key or self._is_meta(key):
            raise KeyError(key)
        field, _ = key.rsplit("/", 1)
        record, ri, chunk_size = self._key_to_record(key)
        maybe = self._items.get((field, record), {}).get(ri, False)
        if maybe is None:
            # explicitly deleted
            raise KeyError
        elif maybe:
            return maybe
        elif chunk_size == 0:
            return b""

        # Chunk keys can be loaded from row group and cached in LRU cache
        try:
            refs = self.open_refs(field, record)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            raise KeyError(key) from exc
        columns = ["path", "offset", "size", "raw"]
        selection = [refs[c][ri] if c in refs else None for c in columns]
        raw = selection[-1]
        if raw is not None:
            return raw
        if selection[0] is None:
            raise KeyError("This reference does not exist or has been deleted")
        if selection[1:3] == [0, 0]:
            # URL only
            return selection[:1]
        # URL, offset, size
        return selection[:3]

    @lru_cache(4096)
    def _key_to_record(self, key):
        """Details needed to construct a reference for one key"""
        field, chunk = key.rsplit("/", 1)
        chunk_sizes = self._get_chunk_sizes(field)
        if len(chunk_sizes) == 0:
            return 0, 0, 0
        chunk_idx = [int(c) for c in chunk.split(".")]
        chunk_number = ravel_multi_index(chunk_idx, chunk_sizes)
        record = chunk_number // self.record_size
        ri = chunk_number % self.record_size
        return record, ri, len(chunk_sizes)

    def _get_chunk_sizes(self, field):
        """The number of chunks along each axis for a given field"""
        if field not in self.chunk_sizes:
            zarray = self.zmetadata[f"{field}/.zarray"]
            size_ratio = [
                math.ceil(s / c) for s, c in zip(zarray["shape"], zarray["chunks"])
            ]
            self.chunk_sizes[field] = size_ratio or [1]
        return self.chunk_sizes[field]

    def _generate_record(self, field, record):
        """The references for a given parquet file of a given field"""
        refs = self.open_refs(field, record)
        it = iter(zip(*refs.values()))
        if len(refs) == 3:
            # All urls
            return (list(t) for t in it)
        elif len(refs) == 1:
            # All raws
            return refs["raw"]
        else:
            # Mix of urls and raws
            return (list(t[:3]) if not t[3] else t[3] for t in it)

    def _generate_all_records(self, field):
        """Load all the references within a field by iterating over the parquet files"""
        nrec = 1
        for ch in self._get_chunk_sizes(field):
            nrec *= ch
        nrec = math.ceil(nrec / self.record_size)
        for record in range(nrec):
            yield from self._generate_record(field, record)

    def values(self):
        return RefsValuesView(self)

    def items(self):
        return RefsItemsView(self)

    def __hash__(self):
        return id(self)

    def __getitem__(self, key):
        return self._load_one_key(key)

    def __setitem__(self, key, value):
        if "/" in key and not self._is_meta(key):
            field, chunk = key.rsplit("/", 1)
            record, i, _ = self._key_to_record(key)
            subdict = self._items.setdefault((field, record), {})
            subdict[i] = value
            if len(subdict) == self.record_size:
                self.write(field, record)
        else:
            # metadata or top-level
            if hasattr(value, "to_bytes"):
                val = value.to_bytes().decode()
            elif isinstance(value, bytes):
                val = value.decode()
            else:
                val = value
            self._items[key] = val
            new_value = json.loads(val)
            self.zmetadata[key] = {**self.zmetadata.get(key, {}), **new_value}

    @staticmethod
    def _is_meta(key):
        return key.startswith(".z") or "/.z" in key

    def __delitem__(self, key):
        if key in self._items:
            del self._items[key]
        elif key in self.zmetadata:
            del self.zmetadata[key]
        else:
            if "/" in key and not self._is_meta(key):
                field, _ = key.rsplit("/", 1)
                record, i, _ = self._key_to_record(key)
                subdict = self._items.setdefault((field, record), {})
                subdict[i] = None
                if len(subdict) == self.record_size:
                    self.write(field, record)
            else:
                # metadata or top-level
                self._items[key] = None

    def write(self, field, record, base_url=None, storage_options=None):
        # extra requirements if writing
        import kerchunk.df
        import numpy as np
        import pandas as pd

        partition = self._items[(field, record)]
        original = False
        if len(partition) < self.record_size:
            try:
                original = self.open_refs(field, record)
            except OSError:
                pass

        if original:
            paths = original["path"]
            offsets = original["offset"]
            sizes = original["size"]
            raws = original["raw"]
        else:
            paths = np.full(self.record_size, np.nan, dtype="O")
            offsets = np.zeros(self.record_size, dtype="int64")
            sizes = np.zeros(self.record_size, dtype="int64")
            raws = np.full(self.record_size, np.nan, dtype="O")
        for j, data in partition.items():
            if isinstance(data, list):
                if (
                    str(paths.dtype) == "category"
                    and data[0] not in paths.dtype.categories
                ):
                    paths = paths.add_categories(data[0])
                paths[j] = data[0]
                if len(data) > 1:
                    offsets[j] = data[1]
                    sizes[j] = data[2]
            elif data is None:
                # delete
                paths[j] = None
                offsets[j] = 0
                sizes[j] = 0
                raws[j] = None
            else:
                # this is the only call into kerchunk, could remove
                raws[j] = kerchunk.df._proc_raw(data)
        # TODO: only save needed columns
        df = pd.DataFrame(
            {
                "path": paths,
                "offset": offsets,
                "size": sizes,
                "raw": raws,
            },
            copy=False,
        )
        if df.path.count() / (df.path.nunique() or 1) > self.cat_thresh:
            df["path"] = df["path"].astype("category")
        object_encoding = {"raw": "bytes", "path": "utf8"}
        has_nulls = ["path", "raw"]

        fn = f"{base_url or self.out_root}/{field}/refs.{record}.parq"
        self.fs.mkdirs(f"{base_url or self.out_root}/{field}", exist_ok=True)

        if self.engine == "pyarrow":
            df_backend_kwargs = {"write_statistics": False}
        elif self.engine == "fastparquet":
            df_backend_kwargs = {
                "stats": False,
                "object_encoding": object_encoding,
                "has_nulls": has_nulls,
            }
        else:
            raise NotImplementedError(f"{self.engine} not supported")
        df.to_parquet(
            fn,
            engine=self.engine,
            storage_options=storage_options
            or getattr(self.fs, "storage_options", None),
            compression="zstd",
            index=False,
            **df_backend_kwargs,
        )

        partition.clear()
        self._items.pop((field, record))

    def flush(self, base_url=None, storage_options=None):
        """Output any modified or deleted keys

        Parameters
        ----------
        base_url: str
            Location of the output
        """

        # write what we have so far and clear sub chunks
        for thing in list(self._items):
            if isinstance(thing, tuple):
                field, record = thing
                self.write(
                    field,
                    record,
                    base_url=base_url,
                    storage_options=storage_options,
                )

        # gather .zmetadata from self._items and write that too
        for k in list(self._items):
            if k != ".zmetadata" and ".z" in k:
                self.zmetadata[k] = json.loads(self._items.pop(k))
        met = {"metadata": self.zmetadata, "record_size": self.record_size}
        self._items.clear()
        self._items[".zmetadata"] = json.dumps(met).encode()
        self.fs.pipe(
            "/".join([base_url or self.out_root, ".zmetadata"]),
            self._items[".zmetadata"],
        )

        # TODO: only clear those that we wrote to?
        self.open_refs.cache_clear()

    def __len__(self):
        # Caveat: This counts expected references, not actual - but is fast
        count = 0
        for field in self.listdir():
            if field.startswith("."):
                count += 1
            else:
                count += math.prod(self._get_chunk_sizes(field))
        count += len(self.zmetadata)  # all metadata keys
        # any other files not in reference partitions
        count += sum(1 for _ in self._items if not isinstance(_, tuple))
        return count

    def __iter__(self):
        # Caveat: returns only existing keys, so the number of these does not
        #  match len(self)
        metas = set(self.zmetadata)
        metas.update(self._items)
        for bit in metas:
            if isinstance(bit, str):
                yield bit
        for field in self.listdir():
            for k in self._keys_in_field(field):
                if k in self:
                    yield k

    def __contains__(self, item):
        try:
            self._load_one_key(item)
            return True
        except KeyError:
            return False

    def _keys_in_field(self, field):
        """List key names in given field

        Produces strings like "field/x.y" appropriate from the chunking of the array
        """
        chunk_sizes = self._get_chunk_sizes(field)
        if len(chunk_sizes) == 0:
            yield field + "/0"
            return
        inds = itertools.product(*(range(i) for i in chunk_sizes))
        for ind in inds:
            yield field + "/" + ".".join([str(c) for c in ind])


class ReferenceFileSystem(AsyncFileSystem):
    """View byte ranges of some other file as a file system
    Initial version: single file system target, which must support
    async, and must allow start and end args in _cat_file. Later versions
    may allow multiple arbitrary URLs for the targets.
    This FileSystem is read-only. It is designed to be used with async
    targets (for now). We do not get original file details from the target FS.
    Configuration is by passing a dict of references at init, or a URL to
    a JSON file containing the same; this dict
    can also contain concrete data for some set of paths.
    Reference dict format:
    {path0: bytes_data, path1: (target_url, offset, size)}
    https://github.com/fsspec/kerchunk/blob/main/README.md
    """

    protocol = "reference"
    cachable = False

    def __init__(
        self,
        fo,
        target=None,
        ref_storage_args=None,
        target_protocol=None,
        target_options=None,
        remote_protocol=None,
        remote_options=None,
        fs=None,
        template_overrides=None,
        simple_templates=True,
        max_gap=64_000,
        max_block=256_000_000,
        cache_size=128,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fo : dict or str
            The set of references to use for this instance, with a structure as above.
            If str referencing a JSON file, will use fsspec.open, in conjunction
            with target_options and target_protocol to open and parse JSON at this
            location. If a directory, then assume references are a set of parquet
            files to be loaded lazily.
        target : str
            For any references having target_url as None, this is the default file
            target to use
        ref_storage_args : dict
            If references is a str, use these kwargs for loading the JSON file.
            Deprecated: use target_options instead.
        target_protocol : str
            Used for loading the reference file, if it is a path. If None, protocol
            will be derived from the given path
        target_options : dict
            Extra FS options for loading the reference file ``fo``, if given as a path
        remote_protocol : str
            The protocol of the filesystem on which the references will be evaluated
            (unless fs is provided). If not given, will be derived from the first
            URL that has a protocol in the templates or in the references, in that
            order.
        remote_options : dict
            kwargs to go with remote_protocol
        fs : AbstractFileSystem | dict(str, (AbstractFileSystem | dict))
            Directly provide a file system(s):
                - a single filesystem instance
                - a dict of protocol:filesystem, where each value is either a filesystem
                  instance, or a dict of kwargs that can be used to create in
                  instance for the given protocol

            If this is given, remote_options and remote_protocol are ignored.
        template_overrides : dict
            Swap out any templates in the references file with these - useful for
            testing.
        simple_templates: bool
            Whether templates can be processed with simple replace (True) or if
            jinja  is needed (False, much slower). All reference sets produced by
            ``kerchunk`` are simple in this sense, but the spec allows for complex.
        max_gap, max_block: int
            For merging multiple concurrent requests to the same remote file.
            Neighboring byte ranges will only be merged when their
            inter-range gap is <= ``max_gap``. Default is 64KB. Set to 0
            to only merge when it requires no extra bytes. Pass a negative
            number to disable merging, appropriate for local target files.
            Neighboring byte ranges will only be merged when the size of
            the aggregated range is <= ``max_block``. Default is 256MB.
        cache_size : int
            Maximum size of LRU cache, where cache_size*record_size denotes
            the total number of references that can be loaded in memory at once.
            Only used for lazily loaded references.
        kwargs : passed to parent class
        """
        super().__init__(**kwargs)
        self.target = target
        self.template_overrides = template_overrides
        self.simple_templates = simple_templates
        self.templates = {}
        self.fss = {}
        self._dircache = {}
        self.max_gap = max_gap
        self.max_block = max_block
        if isinstance(fo, str):
            dic = dict(
                **(ref_storage_args or target_options or {}), protocol=target_protocol
            )
            ref_fs, fo2 = fsspec.core.url_to_fs(fo, **dic)
            if ".json" not in fo2 and (
                fo.endswith(("parq", "parquet", "/")) or ref_fs.isdir(fo2)
            ):
                # Lazy parquet refs
                logger.info("Open lazy reference dict from URL %s", fo)
                self.references = LazyReferenceMapper(
                    fo2,
                    fs=ref_fs,
                    cache_size=cache_size,
                )
            else:
                # text JSON
                with fsspec.open(fo, "rb", **dic) as f:
                    logger.info("Read reference from URL %s", fo)
                    text = json.load(f)
                self._process_references(text, template_overrides)
        else:
            # dictionaries
            self._process_references(fo, template_overrides)
        if isinstance(fs, dict):
            self.fss = {
                k: (
                    fsspec.filesystem(k.split(":", 1)[0], **opts)
                    if isinstance(opts, dict)
                    else opts
                )
                for k, opts in fs.items()
            }
            if None not in self.fss:
                self.fss[None] = filesystem("file")
            return
        if fs is not None:
            # single remote FS
            remote_protocol = (
                fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
            )
            self.fss[remote_protocol] = fs

        if remote_protocol is None:
            # get single protocol from any templates
            for ref in self.templates.values():
                if callable(ref):
                    ref = ref()
                protocol, _ = fsspec.core.split_protocol(ref)
                if protocol and protocol not in self.fss:
                    fs = filesystem(protocol, **(remote_options or {}))
                    self.fss[protocol] = fs
        if remote_protocol is None:
            # get single protocol from references
            # TODO: warning here, since this can be very expensive?
            for ref in self.references.values():
                if callable(ref):
                    ref = ref()
                if isinstance(ref, list) and ref[0]:
                    protocol, _ = fsspec.core.split_protocol(ref[0])
                    if protocol not in self.fss:
                        fs = filesystem(protocol, **(remote_options or {}))
                        self.fss[protocol] = fs
                        # only use first remote URL
                        break

        if remote_protocol and remote_protocol not in self.fss:
            fs = filesystem(remote_protocol, **(remote_options or {}))
            self.fss[remote_protocol] = fs

        self.fss[None] = fs or filesystem("file")  # default one
        # Wrap any non-async filesystems to ensure async methods are available below
        for k, f in self.fss.items():
            if not f.async_impl:
                self.fss[k] = AsyncFileSystemWrapper(f, asynchronous=self.asynchronous)
            elif self.asynchronous ^ f.asynchronous:
                raise ValueError(
                    "Reference-FS's target filesystem must have same value "
                    "of asynchronous"
                )

    def _cat_common(self, path, start=None, end=None):
        path = self._strip_protocol(path)
        logger.debug(f"cat: {path}")
        try:
            part = self.references[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        if isinstance(part, str):
            part = part.encode()
        if hasattr(part, "to_bytes"):
            part = part.to_bytes()
        if isinstance(part, bytes):
            logger.debug(f"Reference: {path}, type bytes")
            if part.startswith(b"base64:"):
                part = base64.b64decode(part[7:])
            return part, None, None

        if len(part) == 1:
            logger.debug(f"Reference: {path}, whole file => {part}")
            url = part[0]
            start1, end1 = start, end
        else:
            url, start0, size = part
            logger.debug(f"Reference: {path} => {url}, offset {start0}, size {size}")
            end0 = start0 + size

            if start is not None:
                if start >= 0:
                    start1 = start0 + start
                else:
                    start1 = end0 + start
            else:
                start1 = start0
            if end is not None:
                if end >= 0:
                    end1 = start0 + end
                else:
                    end1 = end0 + end
            else:
                end1 = end0
        if url is None:
            url = self.target
        return url, start1, end1

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path, start=start, end=end)
        if isinstance(part_or_url, bytes):
            return part_or_url[start:end]
        protocol, _ = split_protocol(part_or_url)
        try:
            return await self.fss[protocol]._cat_file(
                part_or_url, start=start0, end=end0
            )
        except Exception as e:
            raise ReferenceNotReachable(path, part_or_url) from e

    def cat_file(self, path, start=None, end=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path, start=start, end=end)
        if isinstance(part_or_url, bytes):
            return part_or_url[start:end]
        protocol, _ = split_protocol(part_or_url)
        try:
            return self.fss[protocol].cat_file(part_or_url, start=start0, end=end0)
        except Exception as e:
            raise ReferenceNotReachable(path, part_or_url) from e

    def pipe_file(self, path, value, **_):
        """Temporarily add binary data or reference as a file"""
        self.references[path] = value

    async def _get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            return os.makedirs(lpath, exist_ok=True)
        data = await self._cat_file(rpath)
        with open(lpath, "wb") as f:
            f.write(data)

    def get_file(self, rpath, lpath, callback=DEFAULT_CALLBACK, **kwargs):
        if self.isdir(rpath):
            return os.makedirs(lpath, exist_ok=True)
        data = self.cat_file(rpath, **kwargs)
        callback.set_size(len(data))
        if isfilelike(lpath):
            lpath.write(data)
        else:
            with open(lpath, "wb") as f:
                f.write(data)
        callback.absolute_update(len(data))

    def get(self, rpath, lpath, recursive=False, **kwargs):
        if recursive:
            # trigger directory build
            self.ls("")
        rpath = self.expand_path(rpath, recursive=recursive)
        fs = fsspec.filesystem("file", auto_mkdir=True)
        targets = other_paths(rpath, lpath)
        if recursive:
            data = self.cat([r for r in rpath if not self.isdir(r)])
        else:
            data = self.cat(rpath)
        for remote, local in zip(rpath, targets):
            if remote in data:
                fs.pipe_file(local, data[remote])

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        if isinstance(path, str) and recursive:
            raise NotImplementedError
        if isinstance(path, list) and (recursive or any("*" in p for p in path)):
            raise NotImplementedError
        # TODO: if references is lazy, pre-fetch all paths in batch before access
        proto_dict = _protocol_groups(path, self.references)
        out = {}
        for proto, paths in proto_dict.items():
            fs = self.fss[proto]
            urls, starts, ends, valid_paths = [], [], [], []
            for p in paths:
                # find references or label not-found. Early exit if any not
                # found and on_error is "raise"
                try:
                    u, s, e = self._cat_common(p)
                    if not isinstance(u, (bytes, str)):
                        # nan/None from parquet
                        continue
                except FileNotFoundError as err:
                    if on_error == "raise":
                        raise
                    if on_error != "omit":
                        out[p] = err
                else:
                    urls.append(u)
                    starts.append(s)
                    ends.append(e)
                    valid_paths.append(p)

            # process references into form for merging
            urls2 = []
            starts2 = []
            ends2 = []
            paths2 = []
            whole_files = set()
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                if isinstance(u, bytes):
                    # data
                    out[p] = u
                elif s is None:
                    # whole file - limits are None, None, but no further
                    # entries take for this file
                    whole_files.add(u)
                    urls2.append(u)
                    starts2.append(s)
                    ends2.append(e)
                    paths2.append(p)
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                # second run to account for files that are to be loaded whole
                if s is not None and u not in whole_files:
                    urls2.append(u)
                    starts2.append(s)
                    ends2.append(e)
                    paths2.append(p)

            # merge and fetch consolidated ranges
            new_paths, new_starts, new_ends = merge_offset_ranges(
                list(urls2),
                list(starts2),
                list(ends2),
                sort=True,
                max_gap=self.max_gap,
                max_block=self.max_block,
            )
            bytes_out = fs.cat_ranges(new_paths, new_starts, new_ends)

            # unbundle from merged bytes - simple approach
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                if p in out:
                    continue  # was bytes, already handled
                for np, ns, ne, b in zip(new_paths, new_starts, new_ends, bytes_out):
                    if np == u and (ns is None or ne is None):
                        if isinstance(b, Exception):
                            out[p] = b
                        else:
                            out[p] = b[s:e]
                    elif np == u and s >= ns and e <= ne:
                        if isinstance(b, Exception):
                            out[p] = b
                        else:
                            out[p] = b[s - ns : (e - ne) or None]

        for k, v in out.copy().items():
            # these were valid references, but fetch failed, so transform exc
            if isinstance(v, Exception) and k in self.references:
                ex = out[k]
                new_ex = ReferenceNotReachable(k, self.references[k])
                new_ex.__cause__ = ex
                if on_error == "raise":
                    raise new_ex
                elif on_error != "omit":
                    out[k] = new_ex

        if len(out) == 1 and isinstance(path, str) and "*" not in path:
            return _first(out)
        return out

    def _process_references(self, references, template_overrides=None):
        vers = references.get("version", None)
        if vers is None:
            self._process_references0(references)
        elif vers == 1:
            self._process_references1(references, template_overrides=template_overrides)
        else:
            raise ValueError(f"Unknown reference spec version: {vers}")
        # TODO: we make dircache by iterating over all entries, but for Spec >= 1,
        #  can replace with programmatic. Is it even needed for mapper interface?

    def _process_references0(self, references):
        """Make reference dict for Spec Version 0"""
        if isinstance(references, dict):
            # do not do this for lazy/parquet backend, which will not make dicts,
            # but must remain writable in the original object
            references = {
                key: json.dumps(val) if isinstance(val, dict) else val
                for key, val in references.items()
            }
        self.references = references

    def _process_references1(self, references, template_overrides=None):
        if not self.simple_templates or self.templates:
            import jinja2
        self.references = {}
        self._process_templates(references.get("templates", {}))

        @lru_cache(1000)
        def _render_jinja(u):
            return jinja2.Template(u).render(**self.templates)

        for k, v in references.get("refs", {}).items():
            if isinstance(v, str):
                if v.startswith("base64:"):
                    self.references[k] = base64.b64decode(v[7:])
                self.references[k] = v
            elif isinstance(v, dict):
                self.references[k] = json.dumps(v)
            elif self.templates:
                u = v[0]
                if "{{" in u:
                    if self.simple_templates:
                        u = (
                            u.replace("{{", "{")
                            .replace("}}", "}")
                            .format(**self.templates)
                        )
                    else:
                        u = _render_jinja(u)
                self.references[k] = [u] if len(v) == 1 else [u, v[1], v[2]]
            else:
                self.references[k] = v
        self.references.update(self._process_gen(references.get("gen", [])))

    def _process_templates(self, tmp):
        self.templates = {}
        if self.template_overrides is not None:
            tmp.update(self.template_overrides)
        for k, v in tmp.items():
            if "{{" in v:
                import jinja2

                self.templates[k] = lambda temp=v, **kwargs: jinja2.Template(
                    temp
                ).render(**kwargs)
            else:
                self.templates[k] = v

    def _process_gen(self, gens):
        out = {}
        for gen in gens:
            dimension = {
                k: (
                    v
                    if isinstance(v, list)
                    else range(v.get("start", 0), v["stop"], v.get("step", 1))
                )
                for k, v in gen["dimensions"].items()
            }
            products = (
                dict(zip(dimension.keys(), values))
                for values in itertools.product(*dimension.values())
            )
            for pr in products:
                import jinja2

                key = jinja2.Template(gen["key"]).render(**pr, **self.templates)
                url = jinja2.Template(gen["url"]).render(**pr, **self.templates)
                if ("offset" in gen) and ("length" in gen):
                    offset = int(
                        jinja2.Template(gen["offset"]).render(**pr, **self.templates)
                    )
                    length = int(
                        jinja2.Template(gen["length"]).render(**pr, **self.templates)
                    )
                    out[key] = [url, offset, length]
                elif ("offset" in gen) ^ ("length" in gen):
                    raise ValueError(
                        "Both 'offset' and 'length' are required for a "
                        "reference generator entry if either is provided."
                    )
                else:
                    out[key] = [url]
        return out

    def _dircache_from_items(self):
        self.dircache = {"": []}
        it = self.references.items()
        for path, part in it:
            if isinstance(part, (bytes, str)) or hasattr(part, "to_bytes"):
                size = len(part)
            elif len(part) == 1:
                size = None
            else:
                _, _, size = part
            par = path.rsplit("/", 1)[0] if "/" in path else ""
            par0 = par
            subdirs = [par0]
            while par0 and par0 not in self.dircache:
                # collect parent directories
                par0 = self._parent(par0)
                subdirs.append(par0)

            subdirs.reverse()
            for parent, child in zip(subdirs, subdirs[1:]):
                # register newly discovered directories
                assert child not in self.dircache
                assert parent in self.dircache
                self.dircache[parent].append(
                    {"name": child, "type": "directory", "size": 0}
                )
                self.dircache[child] = []

            self.dircache[par].append({"name": path, "type": "file", "size": size})

    def _open(self, path, mode="rb", block_size=None, cache_options=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path)
        # This logic is kept outside `ReferenceFile` to avoid unnecessary redirection.
        # That does mean `_cat_common` gets called twice if it eventually reaches `ReferenceFile`.
        if isinstance(part_or_url, bytes):
            return io.BytesIO(part_or_url[start0:end0])

        protocol, _ = split_protocol(part_or_url)
        if start0 is None and end0 is None:
            return self.fss[protocol]._open(
                part_or_url,
                mode,
                block_size=block_size,
                cache_options=cache_options,
                **kwargs,
            )

        return ReferenceFile(
            self,
            path,
            mode,
            block_size=block_size,
            cache_options=cache_options,
            **kwargs,
        )

    def ls(self, path, detail=True, **kwargs):
        logger.debug("list %s", path)
        path = self._strip_protocol(path)
        if isinstance(self.references, LazyReferenceMapper):
            try:
                return self.references.ls(path, detail)
            except KeyError:
                pass
            raise FileNotFoundError(f"'{path}' is not a known key")
        if not self.dircache:
            self._dircache_from_items()
        out = self._ls_from_cache(path)
        if out is None:
            raise FileNotFoundError(path)
        if detail:
            return out
        return [o["name"] for o in out]

    def exists(self, path, **kwargs):  # overwrite auto-sync version
        return self.isdir(path) or self.isfile(path)

    def isdir(self, path):  # overwrite auto-sync version
        if self.dircache:
            return path in self.dircache
        elif isinstance(self.references, LazyReferenceMapper):
            return path in self.references.listdir()
        else:
            # this may be faster than building dircache for single calls, but
            # by looping will be slow for many calls; could cache it?
            return any(_.startswith(f"{path}/") for _ in self.references)

    def isfile(self, path):  # overwrite auto-sync version
        return path in self.references

    async def _ls(self, path, detail=True, **kwargs):  # calls fast sync code
        return self.ls(path, detail, **kwargs)

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        if withdirs:
            return super().find(
                path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs
            )
        if path:
            path = self._strip_protocol(path)
            r = sorted(k for k in self.references if k.startswith(path))
        else:
            r = sorted(self.references)
        if detail:
            if not self.dircache:
                self._dircache_from_items()
            return {k: self._ls_from_cache(k)[0] for k in r}
        else:
            return r

    def info(self, path, **kwargs):
        out = self.references.get(path)
        if out is not None:
            if isinstance(out, (str, bytes)):
                # decode base64 here
                return {"name": path, "type": "file", "size": len(out)}
            elif len(out) > 1:
                return {"name": path, "type": "file", "size": out[2]}
            else:
                out0 = [{"name": path, "type": "file", "size": None}]
        else:
            out = self.ls(path, True)
            out0 = [o for o in out if o["name"] == path]
            if not out0:
                return {"name": path, "type": "directory", "size": 0}
        if out0[0]["size"] is None:
            # if this is a whole remote file, update size using remote FS
            prot, _ = split_protocol(self.references[path][0])
            out0[0]["size"] = self.fss[prot].size(self.references[path][0])
        return out0[0]

    async def _info(self, path, **kwargs):  # calls fast sync code
        return self.info(path)

    async def _rm_file(self, path, **kwargs):
        self.references.pop(
            path, None
        )  # ignores FileNotFound, just as well for directories
        self.dircache.clear()  # this is a bit heavy handed

    async def _pipe_file(self, path, data, mode="overwrite", **kwargs):
        if mode == "create" and self.exists(path):
            raise FileExistsError
        # can be str or bytes
        self.references[path] = data
        self.dircache.clear()  # this is a bit heavy handed

    async def _put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        # puts binary
        if mode == "create" and self.exists(rpath):
            raise FileExistsError
        with open(lpath, "rb") as f:
            self.references[rpath] = f.read()
        self.dircache.clear()  # this is a bit heavy handed

    def save_json(self, url, **storage_options):
        """Write modified references into new location"""
        out = {}
        for k, v in self.references.items():
            if isinstance(v, bytes):
                try:
                    out[k] = v.decode("ascii")
                except UnicodeDecodeError:
                    out[k] = (b"base64:" + base64.b64encode(v)).decode()
            else:
                out[k] = v
        with fsspec.open(url, "wb", **storage_options) as f:
            f.write(json.dumps({"version": 1, "refs": out}).encode())


class ReferenceFile(AbstractBufferedFile):
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
        super().__init__(
            fs,
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            size=size,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )
        part_or_url, self.start, self.end = self.fs._cat_common(self.path)
        protocol, _ = split_protocol(part_or_url)
        self.src_fs = self.fs.fss[protocol]
        self.src_path = part_or_url
        self._f = None

    @property
    def f(self):
        if self._f is None or self._f.closed:
            self._f = self.src_fs._open(
                self.src_path,
                mode=self.mode,
                block_size=self.blocksize,
                autocommit=self.autocommit,
                cache_type="none",
                **self.kwargs,
            )
        return self._f

    def close(self):
        if self._f is not None:
            self._f.close()
        return super().close()

    def _fetch_range(self, start, end):
        start = start + self.start
        end = min(end + self.start, self.end)
        self.f.seek(start)
        return self.f.read(end - start)
