from __future__ import annotations

import collections
import functools
import logging
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING, Any, ClassVar, Generic, NamedTuple, TypeVar

if TYPE_CHECKING:
    import mmap

    from typing_extensions import ParamSpec

    P = ParamSpec("P")
else:
    P = TypeVar("P")

T = TypeVar("T")


logger = logging.getLogger("fsspec")

Fetcher = Callable[[int, int], bytes]  # Maps (start, end) to bytes
MultiFetcher = Callable[[list[int, int]], bytes]  # Maps [(start, end)] to bytes


class BaseCache:
    """Pass-though cache: doesn't keep anything, calls every time

    Acts as base class for other cachers

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: func
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    """

    name: ClassVar[str] = "none"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        self.blocksize = blocksize
        self.nblocks = 0
        self.fetcher = fetcher
        self.size = size
        self.hit_count = 0
        self.miss_count = 0
        # the bytes that we actually requested
        self.total_requested_bytes = 0

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        if start >= self.size or start >= stop:
            return b""
        return self.fetcher(start, stop)

    def _reset_stats(self) -> None:
        """Reset hit and miss counts for a more ganular report e.g. by file."""
        self.hit_count = 0
        self.miss_count = 0
        self.total_requested_bytes = 0

    def _log_stats(self) -> str:
        """Return a formatted string of the cache statistics."""
        if self.hit_count == 0 and self.miss_count == 0:
            # a cache that does nothing, this is for logs only
            return ""
        return f" , {self.name}: {self.hit_count} hits, {self.miss_count} misses, {self.total_requested_bytes} total requested bytes"

    def __repr__(self) -> str:
        # TODO: use rich for better formatting
        return f"""
        <{self.__class__.__name__}:
            block size  :   {self.blocksize}
            block count :   {self.nblocks}
            file size   :   {self.size}
            cache hits  :   {self.hit_count}
            cache misses:   {self.miss_count}
            total requested bytes: {self.total_requested_bytes}>
        """


class MMapCache(BaseCache):
    """memory-mapped sparse file cache

    Opens temporary file, which is filled blocks-wise when data is requested.
    Ensure there is enough disc space in the temporary location.

    This cache method might only work on posix

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: Fetcher
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    location: str
        Where to create the temporary file. If None, a temporary file is
        created using tempfile.TemporaryFile().
    blocks: set[int]
        Set of block numbers that have already been fetched. If None, an empty
        set is created.
    multi_fetcher: MultiFetcher
        Function of the form f([(start, end)]) which gets bytes from remote
        as specified. This function is used to fetch multiple blocks at once.
        If not specified, the fetcher function is used instead.
    """

    name = "mmap"

    def __init__(
        self,
        blocksize: int,
        fetcher: Fetcher,
        size: int,
        location: str | None = None,
        blocks: set[int] | None = None,
        multi_fetcher: MultiFetcher | None = None,
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.blocks = set() if blocks is None else blocks
        self.location = location
        self.multi_fetcher = multi_fetcher
        self.cache = self._makefile()

    def _makefile(self) -> mmap.mmap | bytearray:
        import mmap
        import tempfile

        if self.size == 0:
            return bytearray()

        # posix version
        if self.location is None or not os.path.exists(self.location):
            if self.location is None:
                fd = tempfile.TemporaryFile()
                self.blocks = set()
            else:
                fd = open(self.location, "wb+")
            fd.seek(self.size - 1)
            fd.write(b"1")
            fd.flush()
        else:
            fd = open(self.location, "r+b")

        return mmap.mmap(fd.fileno(), self.size)

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        logger.debug(f"MMap cache fetching {start}-{end}")
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        start_block = start // self.blocksize
        end_block = end // self.blocksize
        block_range = range(start_block, end_block + 1)
        # Determine which blocks need to be fetched. This sequence is sorted by construction.
        need = (i for i in block_range if i not in self.blocks)
        # Count the number of blocks already cached
        self.hit_count += sum(1 for i in block_range if i in self.blocks)

        ranges = []

        # Consolidate needed blocks.
        # Algorithm adapted from Python 2.x itertools documentation.
        # We are grouping an enumerated sequence of blocks. By comparing when the difference
        # between an ascending range (provided by enumerate) and the needed block numbers
        # we can detect when the block number skips values. The key computes this difference.
        # Whenever the difference changes, we know that we have previously cached block(s),
        # and a new group is started. In other words, this algorithm neatly groups
        # runs of consecutive block numbers so they can be fetched together.
        for _, _blocks in groupby(enumerate(need), key=lambda x: x[0] - x[1]):
            # Extract the blocks from the enumerated sequence
            _blocks = tuple(map(itemgetter(1), _blocks))
            # Compute start of first block
            sstart = _blocks[0] * self.blocksize
            # Compute the end of the last block. Last block may not be full size.
            send = min(_blocks[-1] * self.blocksize + self.blocksize, self.size)

            # Fetch bytes (could be multiple consecutive blocks)
            self.total_requested_bytes += send - sstart
            logger.debug(
                f"MMap get blocks {_blocks[0]}-{_blocks[-1]} ({sstart}-{send})"
            )
            ranges.append((sstart, send))

            # Update set of cached blocks
            self.blocks.update(_blocks)
            # Update cache statistics with number of blocks we had to cache
            self.miss_count += len(_blocks)

        if not ranges:
            return self.cache[start:end]

        if self.multi_fetcher:
            logger.debug(f"MMap get blocks {ranges}")
            for idx, r in enumerate(self.multi_fetcher(ranges)):
                (sstart, send) = ranges[idx]
                logger.debug(f"MMap copy block ({sstart}-{send}")
                self.cache[sstart:send] = r
        else:
            for sstart, send in ranges:
                logger.debug(f"MMap get block ({sstart}-{send}")
                self.cache[sstart:send] = self.fetcher(sstart, send)

        return self.cache[start:end]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["cache"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore instance attributes
        self.__dict__.update(state)
        self.cache = self._makefile()


class ReadAheadCache(BaseCache):
    """Cache which reads only when we get beyond a block of data

    This is a much simpler version of BytesCache, and does not attempt to
    fill holes in the cache or keep fragments alive. It is best suited to
    many small reads in a sequential order (e.g., reading lines from a file).
    """

    name = "readahead"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        super().__init__(blocksize, fetcher, size)
        self.cache = b""
        self.start = 0
        self.end = 0

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        l = end - start
        if start >= self.start and end <= self.end:
            # cache hit
            self.hit_count += 1
            return self.cache[start - self.start : end - self.start]
        elif self.start <= start < self.end:
            # partial hit
            self.miss_count += 1
            part = self.cache[start - self.start :]
            l -= len(part)
            start = self.end
        else:
            # miss
            self.miss_count += 1
            part = b""
        end = min(self.size, end + self.blocksize)
        self.total_requested_bytes += end - start
        self.cache = self.fetcher(start, end)  # new block replaces old
        self.start = start
        self.end = self.start + len(self.cache)
        return part + self.cache[:l]


class FirstChunkCache(BaseCache):
    """Caches the first block of a file only

    This may be useful for file types where the metadata is stored in the header,
    but is randomly accessed.
    """

    name = "first"

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        if blocksize > size:
            # this will buffer the whole thing
            blocksize = size
        super().__init__(blocksize, fetcher, size)
        self.cache: bytes | None = None

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        start = start or 0
        if start > self.size:
            logger.debug("FirstChunkCache: requested start > file size")
            return b""

        end = min(end, self.size)

        if start < self.blocksize:
            if self.cache is None:
                self.miss_count += 1
                if end > self.blocksize:
                    self.total_requested_bytes += end
                    data = self.fetcher(0, end)
                    self.cache = data[: self.blocksize]
                    return data[start:]
                self.cache = self.fetcher(0, self.blocksize)
                self.total_requested_bytes += self.blocksize
            part = self.cache[start:end]
            if end > self.blocksize:
                self.total_requested_bytes += end - self.blocksize
                part += self.fetcher(self.blocksize, end)
            self.hit_count += 1
            return part
        else:
            self.miss_count += 1
            self.total_requested_bytes += end - start
            return self.fetcher(start, end)


class BlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored.

    Parameters
    ----------
    blocksize : int
        The number of bytes to store in each block.
        Requests are only ever made for ``blocksize``, so this
        should balance the overhead of making a request against
        the granularity of the blocks.
    fetcher : Callable
    size : int
        The total size of the file being cached.
    maxblocks : int
        The maximum number of blocks to cache for. The maximum memory
        use for this cache is then ``blocksize * maxblocks``.
    """

    name = "blockcache"

    def __init__(
        self, blocksize: int, fetcher: Fetcher, size: int, maxblocks: int = 32
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = functools.lru_cache(maxblocks)(self._fetch_block)

    def cache_info(self):
        """
        The statistics on the block cache.

        Returns
        -------
        NamedTuple
            Returned directly from the LRU Cache used internally.
        """
        return self._fetch_block_cached.cache_info()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__
        del state["_fetch_block_cached"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._fetch_block_cached = functools.lru_cache(state["maxblocks"])(
            self._fetch_block
        )

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""

        # byte position -> block numbers
        start_block_number = start // self.blocksize
        end_block_number = end // self.blocksize

        # these are cached, so safe to do multiple calls for the same start and end.
        for block_number in range(start_block_number, end_block_number + 1):
            self._fetch_block_cached(block_number)

        return self._read_cache(
            start,
            end,
            start_block_number=start_block_number,
            end_block_number=end_block_number,
        )

    def _fetch_block(self, block_number: int) -> bytes:
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(
                f"'block_number={block_number}' is greater than "
                f"the number of blocks ({self.nblocks})"
            )

        start = block_number * self.blocksize
        end = start + self.blocksize
        self.total_requested_bytes += end - start
        self.miss_count += 1
        logger.info("BlockCache fetching block %d", block_number)
        block_contents = super()._fetch(start, end)
        return block_contents

    def _read_cache(
        self, start: int, end: int, start_block_number: int, end_block_number: int
    ) -> bytes:
        """
        Read from our block cache.

        Parameters
        ----------
        start, end : int
            The start and end byte positions.
        start_block_number, end_block_number : int
            The start and end block numbers.
        """
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize

        self.hit_count += 1
        if start_block_number == end_block_number:
            block: bytes = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]

        else:
            # read from the initial
            out = [self._fetch_block_cached(start_block_number)[start_pos:]]

            # intermediate blocks
            # Note: it'd be nice to combine these into one big request. However
            # that doesn't play nicely with our LRU cache.
            out.extend(
                map(
                    self._fetch_block_cached,
                    range(start_block_number + 1, end_block_number),
                )
            )

            # final block
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])

            return b"".join(out)


class BytesCache(BaseCache):
    """Cache which holds data in a in-memory bytes object

    Implements read-ahead by the block size, for semi-random reads progressing
    through the file.

    Parameters
    ----------
    trim: bool
        As we read more data, whether to discard the start of the buffer when
        we are more than a blocksize ahead of it.
    """

    name: ClassVar[str] = "bytes"

    def __init__(
        self, blocksize: int, fetcher: Fetcher, size: int, trim: bool = True
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.cache = b""
        self.start: int | None = None
        self.end: int | None = None
        self.trim = trim

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        # TODO: only set start/end after fetch, in case it fails?
        # is this where retry logic might go?
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        if (
            self.start is not None
            and start >= self.start
            and self.end is not None
            and end < self.end
        ):
            # cache hit: we have all the required data
            offset = start - self.start
            self.hit_count += 1
            return self.cache[offset : offset + end - start]

        if self.blocksize:
            bend = min(self.size, end + self.blocksize)
        else:
            bend = end

        if bend == start or start > self.size:
            return b""

        if (self.start is None or start < self.start) and (
            self.end is None or end > self.end
        ):
            # First read, or extending both before and after
            self.total_requested_bytes += bend - start
            self.miss_count += 1
            self.cache = self.fetcher(start, bend)
            self.start = start
        else:
            assert self.start is not None
            assert self.end is not None
            self.miss_count += 1

            if start < self.start:
                if self.end is None or self.end - end > self.blocksize:
                    self.total_requested_bytes += bend - start
                    self.cache = self.fetcher(start, bend)
                    self.start = start
                else:
                    self.total_requested_bytes += self.start - start
                    new = self.fetcher(start, self.start)
                    self.start = start
                    self.cache = new + self.cache
            elif self.end is not None and bend > self.end:
                if self.end > self.size:
                    pass
                elif end - self.end > self.blocksize:
                    self.total_requested_bytes += bend - start
                    self.cache = self.fetcher(start, bend)
                    self.start = start
                else:
                    self.total_requested_bytes += bend - self.end
                    new = self.fetcher(self.end, bend)
                    self.cache = self.cache + new

        self.end = self.start + len(self.cache)
        offset = start - self.start
        out = self.cache[offset : offset + end - start]
        if self.trim:
            num = (self.end - self.start) // (self.blocksize + 1)
            if num > 1:
                self.start += self.blocksize * num
                self.cache = self.cache[self.blocksize * num :]
        return out

    def __len__(self) -> int:
        return len(self.cache)


class AllBytes(BaseCache):
    """Cache entire contents of the file"""

    name: ClassVar[str] = "all"

    def __init__(
        self,
        blocksize: int | None = None,
        fetcher: Fetcher | None = None,
        size: int | None = None,
        data: bytes | None = None,
    ) -> None:
        super().__init__(blocksize, fetcher, size)  # type: ignore[arg-type]
        if data is None:
            self.miss_count += 1
            self.total_requested_bytes += self.size
            data = self.fetcher(0, self.size)
        self.data = data

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        self.hit_count += 1
        return self.data[start:stop]


class KnownPartsOfAFile(BaseCache):
    """
    Cache holding known file parts.

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: func
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    data: dict
        A dictionary mapping explicit `(start, stop)` file-offset tuples
        with known bytes.
    strict: bool, default True
        Whether to fetch reads that go beyond a known byte-range boundary.
        If `False`, any read that ends outside a known part will be zero
        padded. Note that zero padding will not be used for reads that
        begin outside a known byte-range.
    """

    name: ClassVar[str] = "parts"

    def __init__(
        self,
        blocksize: int,
        fetcher: Fetcher,
        size: int,
        data: dict[tuple[int, int], bytes] | None = None,
        strict: bool = False,
        **_: Any,
    ):
        super().__init__(blocksize, fetcher, size)
        self.strict = strict

        # simple consolidation of contiguous blocks
        if data:
            old_offsets = sorted(data.keys())
            offsets = [old_offsets[0]]
            blocks = [data.pop(old_offsets[0])]
            for start, stop in old_offsets[1:]:
                start0, stop0 = offsets[-1]
                if start == stop0:
                    offsets[-1] = (start0, stop)
                    blocks[-1] += data.pop((start, stop))
                else:
                    offsets.append((start, stop))
                    blocks.append(data.pop((start, stop)))

            self.data = dict(zip(offsets, blocks))
        else:
            self.data = {}

    @property
    def size(self):
        return sum(_[1] - _[0] for _ in self.data)

    @size.setter
    def size(self, value):
        pass

    @property
    def nblocks(self):
        return len(self.data)

    @nblocks.setter
    def nblocks(self, value):
        pass

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        self.total_requested_bytes += stop - start

        out = b""
        started = False
        loc_old = 0
        for loc0, loc1 in sorted(self.data):
            if (loc0 <= start < loc1) and (loc0 <= stop <= loc1):
                # entirely within the block
                off = start - loc0
                self.hit_count += 1
                return self.data[(loc0, loc1)][off : off + stop - start]
            if stop <= loc0:
                break
            if started and loc0 > loc_old:
                # a gap where we need data
                self.miss_count += 1
                if self.strict:
                    raise ValueError
                out += b"\x00" * (loc0 - loc_old)
            if loc0 <= start < loc1:
                # found the start
                self.hit_count += 1
                off = start - loc0
                out = self.data[(loc0, loc1)][off : off + stop - start]
                started = True
            elif start < loc0 and stop > loc1:
                # the whole block
                self.hit_count += 1
                out += self.data[(loc0, loc1)]
            elif loc0 <= stop <= loc1:
                # end block
                self.hit_count += 1
                return out + self.data[(loc0, loc1)][: stop - loc0]
            loc_old = loc1
        self.miss_count += 1
        if started and not self.strict:
            return out + b"\x00" * (stop - loc_old)
        raise ValueError


class UpdatableLRU(Generic[P, T]):
    """
    Custom implementation of LRU cache that allows updating keys

    Used by BackgroudBlockCache
    """

    class CacheInfo(NamedTuple):
        hits: int
        misses: int
        maxsize: int
        currsize: int

    def __init__(self, func: Callable[P, T], max_size: int = 128) -> None:
        self._cache: OrderedDict[Any, T] = collections.OrderedDict()
        self._func = func
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if kwargs:
            raise TypeError(f"Got unexpected keyword argument {kwargs.keys()}")
        with self._lock:
            if args in self._cache:
                self._cache.move_to_end(args)
                self._hits += 1
                return self._cache[args]

        result = self._func(*args, **kwargs)

        with self._lock:
            self._cache[args] = result
            self._misses += 1
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        return result

    def is_key_cached(self, *args: Any) -> bool:
        with self._lock:
            return args in self._cache

    def add_key(self, result: T, *args: Any) -> None:
        with self._lock:
            self._cache[args] = result
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def cache_info(self) -> UpdatableLRU.CacheInfo:
        with self._lock:
            return self.CacheInfo(
                maxsize=self._max_size,
                currsize=len(self._cache),
                hits=self._hits,
                misses=self._misses,
            )


class BackgroundBlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks with pre-loading of
    the next block in the background.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored. If the
    next block is not in cache, it is loaded in a separate thread
    in non-blocking way.

    Parameters
    ----------
    blocksize : int
        The number of bytes to store in each block.
        Requests are only ever made for ``blocksize``, so this
        should balance the overhead of making a request against
        the granularity of the blocks.
    fetcher : Callable
    size : int
        The total size of the file being cached.
    maxblocks : int
        The maximum number of blocks to cache for. The maximum memory
        use for this cache is then ``blocksize * maxblocks``.
    """

    name: ClassVar[str] = "background"

    def __init__(
        self, blocksize: int, fetcher: Fetcher, size: int, maxblocks: int = 32
    ) -> None:
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, maxblocks)

        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number: int | None = None
        self._fetch_future: Future[bytes] | None = None
        self._fetch_future_lock = threading.Lock()

    def cache_info(self) -> UpdatableLRU.CacheInfo:
        """
        The statistics on the block cache.

        Returns
        -------
        NamedTuple
            Returned directly from the LRU Cache used internally.
        """
        return self._fetch_block_cached.cache_info()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__
        del state["_fetch_block_cached"]
        del state["_thread_executor"]
        del state["_fetch_future_block_number"]
        del state["_fetch_future"]
        del state["_fetch_future_lock"]
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, state["maxblocks"])
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number = None
        self._fetch_future = None
        self._fetch_future_lock = threading.Lock()

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""

        # byte position -> block numbers
        start_block_number = start // self.blocksize
        end_block_number = end // self.blocksize

        fetch_future_block_number = None
        fetch_future = None
        with self._fetch_future_lock:
            # Background thread is running. Check we we can or must join it.
            if self._fetch_future is not None:
                assert self._fetch_future_block_number is not None
                if self._fetch_future.done():
                    logger.info("BlockCache joined background fetch without waiting.")
                    self._fetch_block_cached.add_key(
                        self._fetch_future.result(), self._fetch_future_block_number
                    )
                    # Cleanup the fetch variables. Done with fetching the block.
                    self._fetch_future_block_number = None
                    self._fetch_future = None
                else:
                    # Must join if we need the block for the current fetch
                    must_join = bool(
                        start_block_number
                        <= self._fetch_future_block_number
                        <= end_block_number
                    )
                    if must_join:
                        # Copy to the local variables to release lock
                        # before waiting for result
                        fetch_future_block_number = self._fetch_future_block_number
                        fetch_future = self._fetch_future

                        # Cleanup the fetch variables. Have a local copy.
                        self._fetch_future_block_number = None
                        self._fetch_future = None

        # Need to wait for the future for the current read
        if fetch_future is not None:
            logger.info("BlockCache waiting for background fetch.")
            # Wait until result and put it in cache
            self._fetch_block_cached.add_key(
                fetch_future.result(), fetch_future_block_number
            )

        # these are cached, so safe to do multiple calls for the same start and end.
        for block_number in range(start_block_number, end_block_number + 1):
            self._fetch_block_cached(block_number)

        # fetch next block in the background if nothing is running in the background,
        # the block is within file and it is not already cached
        end_block_plus_1 = end_block_number + 1
        with self._fetch_future_lock:
            if (
                self._fetch_future is None
                and end_block_plus_1 <= self.nblocks
                and not self._fetch_block_cached.is_key_cached(end_block_plus_1)
            ):
                self._fetch_future_block_number = end_block_plus_1
                self._fetch_future = self._thread_executor.submit(
                    self._fetch_block, end_block_plus_1, "async"
                )

        return self._read_cache(
            start,
            end,
            start_block_number=start_block_number,
            end_block_number=end_block_number,
        )

    def _fetch_block(self, block_number: int, log_info: str = "sync") -> bytes:
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(
                f"'block_number={block_number}' is greater than "
                f"the number of blocks ({self.nblocks})"
            )

        start = block_number * self.blocksize
        end = start + self.blocksize
        logger.info("BlockCache fetching block (%s) %d", log_info, block_number)
        self.total_requested_bytes += end - start
        self.miss_count += 1
        block_contents = super()._fetch(start, end)
        return block_contents

    def _read_cache(
        self, start: int, end: int, start_block_number: int, end_block_number: int
    ) -> bytes:
        """
        Read from our block cache.

        Parameters
        ----------
        start, end : int
            The start and end byte positions.
        start_block_number, end_block_number : int
            The start and end block numbers.
        """
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize

        # kind of pointless to count this as a hit, but it is
        self.hit_count += 1

        if start_block_number == end_block_number:
            block = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]

        else:
            # read from the initial
            out = [self._fetch_block_cached(start_block_number)[start_pos:]]

            # intermediate blocks
            # Note: it'd be nice to combine these into one big request. However
            # that doesn't play nicely with our LRU cache.
            out.extend(
                map(
                    self._fetch_block_cached,
                    range(start_block_number + 1, end_block_number),
                )
            )

            # final block
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])

            return b"".join(out)


caches: dict[str | None, type[BaseCache]] = {
    # one custom case
    None: BaseCache,
}


def register_cache(cls: type[BaseCache], clobber: bool = False) -> None:
    """'Register' cache implementation.

    Parameters
    ----------
    clobber: bool, optional
        If set to True (default is False) - allow to overwrite existing
        entry.

    Raises
    ------
    ValueError
    """
    name = cls.name
    if not clobber and name in caches:
        raise ValueError(f"Cache with name {name!r} is already known: {caches[name]}")
    caches[name] = cls


for c in (
    BaseCache,
    MMapCache,
    BytesCache,
    ReadAheadCache,
    BlockCache,
    FirstChunkCache,
    AllBytes,
    KnownPartsOfAFile,
    BackgroundBlockCache,
):
    register_cache(c)
