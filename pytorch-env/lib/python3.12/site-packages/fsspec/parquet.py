import io
import json
import warnings

from .core import url_to_fs
from .utils import merge_offset_ranges

# Parquet-Specific Utilities for fsspec
#
# Most of the functions defined in this module are NOT
# intended for public consumption. The only exception
# to this is `open_parquet_file`, which should be used
# place of `fs.open()` to open parquet-formatted files
# on remote file systems.


def open_parquet_file(
    path,
    mode="rb",
    fs=None,
    metadata=None,
    columns=None,
    row_groups=None,
    storage_options=None,
    strict=False,
    engine="auto",
    max_gap=64_000,
    max_block=256_000_000,
    footer_sample_size=1_000_000,
    **kwargs,
):
    """
    Return a file-like object for a single Parquet file.

    The specified parquet `engine` will be used to parse the
    footer metadata, and determine the required byte ranges
    from the file. The target path will then be opened with
    the "parts" (`KnownPartsOfAFile`) caching strategy.

    Note that this method is intended for usage with remote
    file systems, and is unlikely to improve parquet-read
    performance on local file systems.

    Parameters
    ----------
    path: str
        Target file path.
    mode: str, optional
        Mode option to be passed through to `fs.open`. Default is "rb".
    metadata: Any, optional
        Parquet metadata object. Object type must be supported
        by the backend parquet engine. For now, only the "fastparquet"
        engine supports an explicit `ParquetFile` metadata object.
        If a metadata object is supplied, the remote footer metadata
        will not need to be transferred into local memory.
    fs: AbstractFileSystem, optional
        Filesystem object to use for opening the file. If nothing is
        specified, an `AbstractFileSystem` object will be inferred.
    engine : str, default "auto"
        Parquet engine to use for metadata parsing. Allowed options
        include "fastparquet", "pyarrow", and "auto". The specified
        engine must be installed in the current environment. If
        "auto" is specified, and both engines are installed,
        "fastparquet" will take precedence over "pyarrow".
    columns: list, optional
        List of all column names that may be read from the file.
    row_groups : list, optional
        List of all row-groups that may be read from the file. This
        may be a list of row-group indices (integers), or it may be
        a list of `RowGroup` metadata objects (if the "fastparquet"
        engine is used).
    storage_options : dict, optional
        Used to generate an `AbstractFileSystem` object if `fs` was
        not specified.
    strict : bool, optional
        Whether the resulting `KnownPartsOfAFile` cache should
        fetch reads that go beyond a known byte-range boundary.
        If `False` (the default), any read that ends outside a
        known part will be zero padded. Note that using
        `strict=True` may be useful for debugging.
    max_gap : int, optional
        Neighboring byte ranges will only be merged when their
        inter-range gap is <= `max_gap`. Default is 64KB.
    max_block : int, optional
        Neighboring byte ranges will only be merged when the size of
        the aggregated range is <= `max_block`. Default is 256MB.
    footer_sample_size : int, optional
        Number of bytes to read from the end of the path to look
        for the footer metadata. If the sampled bytes do not contain
        the footer, a second read request will be required, and
        performance will suffer. Default is 1MB.
    **kwargs :
        Optional key-word arguments to pass to `fs.open`
    """

    # Make sure we have an `AbstractFileSystem` object
    # to work with
    if fs is None:
        fs = url_to_fs(path, **(storage_options or {}))[0]

    # For now, `columns == []` not supported. Just use
    # default `open` command with `path` input
    if columns is not None and len(columns) == 0:
        return fs.open(path, mode=mode)

    # Set the engine
    engine = _set_engine(engine)

    # Fetch the known byte ranges needed to read
    # `columns` and/or `row_groups`
    data = _get_parquet_byte_ranges(
        [path],
        fs,
        metadata=metadata,
        columns=columns,
        row_groups=row_groups,
        engine=engine,
        max_gap=max_gap,
        max_block=max_block,
        footer_sample_size=footer_sample_size,
    )

    # Extract file name from `data`
    fn = next(iter(data)) if data else path

    # Call self.open with "parts" caching
    options = kwargs.pop("cache_options", {}).copy()
    return fs.open(
        fn,
        mode=mode,
        cache_type="parts",
        cache_options={
            **options,
            "data": data.get(fn, {}),
            "strict": strict,
        },
        **kwargs,
    )


def _get_parquet_byte_ranges(
    paths,
    fs,
    metadata=None,
    columns=None,
    row_groups=None,
    max_gap=64_000,
    max_block=256_000_000,
    footer_sample_size=1_000_000,
    engine="auto",
):
    """Get a dictionary of the known byte ranges needed
    to read a specific column/row-group selection from a
    Parquet dataset. Each value in the output dictionary
    is intended for use as the `data` argument for the
    `KnownPartsOfAFile` caching strategy of a single path.
    """

    # Set engine if necessary
    if isinstance(engine, str):
        engine = _set_engine(engine)

    # Pass to specialized function if metadata is defined
    if metadata is not None:
        # Use the provided parquet metadata object
        # to avoid transferring/parsing footer metadata
        return _get_parquet_byte_ranges_from_metadata(
            metadata,
            fs,
            engine,
            columns=columns,
            row_groups=row_groups,
            max_gap=max_gap,
            max_block=max_block,
        )

    # Get file sizes asynchronously
    file_sizes = fs.sizes(paths)

    # Populate global paths, starts, & ends
    result = {}
    data_paths = []
    data_starts = []
    data_ends = []
    add_header_magic = True
    if columns is None and row_groups is None:
        # We are NOT selecting specific columns or row-groups.
        #
        # We can avoid sampling the footers, and just transfer
        # all file data with cat_ranges
        for i, path in enumerate(paths):
            result[path] = {}
            for b in range(0, file_sizes[i], max_block):
                data_paths.append(path)
                data_starts.append(b)
                data_ends.append(min(b + max_block, file_sizes[i]))
        add_header_magic = False  # "Magic" should already be included
    else:
        # We ARE selecting specific columns or row-groups.
        #
        # Gather file footers.
        # We just take the last `footer_sample_size` bytes of each
        # file (or the entire file if it is smaller than that)
        footer_starts = []
        footer_ends = []
        for i, path in enumerate(paths):
            footer_ends.append(file_sizes[i])
            sample_size = max(0, file_sizes[i] - footer_sample_size)
            footer_starts.append(sample_size)
        footer_samples = fs.cat_ranges(paths, footer_starts, footer_ends)

        # Check our footer samples and re-sample if necessary.
        missing_footer_starts = footer_starts.copy()
        large_footer = 0
        for i, path in enumerate(paths):
            footer_size = int.from_bytes(footer_samples[i][-8:-4], "little")
            real_footer_start = file_sizes[i] - (footer_size + 8)
            if real_footer_start < footer_starts[i]:
                missing_footer_starts[i] = real_footer_start
                large_footer = max(large_footer, (footer_size + 8))
        if large_footer:
            warnings.warn(
                f"Not enough data was used to sample the parquet footer. "
                f"Try setting footer_sample_size >= {large_footer}."
            )
            for i, block in enumerate(
                fs.cat_ranges(
                    paths,
                    missing_footer_starts,
                    footer_starts,
                )
            ):
                footer_samples[i] = block + footer_samples[i]
                footer_starts[i] = missing_footer_starts[i]

        # Calculate required byte ranges for each path
        for i, path in enumerate(paths):
            # Deal with small-file case.
            # Just include all remaining bytes of the file
            # in a single range.
            if file_sizes[i] < max_block:
                if footer_starts[i] > 0:
                    # Only need to transfer the data if the
                    # footer sample isn't already the whole file
                    data_paths.append(path)
                    data_starts.append(0)
                    data_ends.append(footer_starts[i])
                continue

            # Use "engine" to collect data byte ranges
            path_data_starts, path_data_ends = engine._parquet_byte_ranges(
                columns,
                row_groups=row_groups,
                footer=footer_samples[i],
                footer_start=footer_starts[i],
            )

            data_paths += [path] * len(path_data_starts)
            data_starts += path_data_starts
            data_ends += path_data_ends

        # Merge adjacent offset ranges
        data_paths, data_starts, data_ends = merge_offset_ranges(
            data_paths,
            data_starts,
            data_ends,
            max_gap=max_gap,
            max_block=max_block,
            sort=False,  # Should already be sorted
        )

        # Start by populating `result` with footer samples
        for i, path in enumerate(paths):
            result[path] = {(footer_starts[i], footer_ends[i]): footer_samples[i]}

    # Transfer the data byte-ranges into local memory
    _transfer_ranges(fs, result, data_paths, data_starts, data_ends)

    # Add b"PAR1" to header if necessary
    if add_header_magic:
        _add_header_magic(result)

    return result


def _get_parquet_byte_ranges_from_metadata(
    metadata,
    fs,
    engine,
    columns=None,
    row_groups=None,
    max_gap=64_000,
    max_block=256_000_000,
):
    """Simplified version of `_get_parquet_byte_ranges` for
    the case that an engine-specific `metadata` object is
    provided, and the remote footer metadata does not need to
    be transferred before calculating the required byte ranges.
    """

    # Use "engine" to collect data byte ranges
    data_paths, data_starts, data_ends = engine._parquet_byte_ranges(
        columns,
        row_groups=row_groups,
        metadata=metadata,
    )

    # Merge adjacent offset ranges
    data_paths, data_starts, data_ends = merge_offset_ranges(
        data_paths,
        data_starts,
        data_ends,
        max_gap=max_gap,
        max_block=max_block,
        sort=False,  # Should be sorted
    )

    # Transfer the data byte-ranges into local memory
    result = {fn: {} for fn in list(set(data_paths))}
    _transfer_ranges(fs, result, data_paths, data_starts, data_ends)

    # Add b"PAR1" to header
    _add_header_magic(result)

    return result


def _transfer_ranges(fs, blocks, paths, starts, ends):
    # Use cat_ranges to gather the data byte_ranges
    ranges = (paths, starts, ends)
    for path, start, stop, data in zip(*ranges, fs.cat_ranges(*ranges)):
        blocks[path][(start, stop)] = data


def _add_header_magic(data):
    # Add b"PAR1" to file headers
    for path in list(data.keys()):
        add_magic = True
        for k in data[path].keys():
            if k[0] == 0 and k[1] >= 4:
                add_magic = False
                break
        if add_magic:
            data[path][(0, 4)] = b"PAR1"


def _set_engine(engine_str):
    # Define a list of parquet engines to try
    if engine_str == "auto":
        try_engines = ("fastparquet", "pyarrow")
    elif not isinstance(engine_str, str):
        raise ValueError(
            "Failed to set parquet engine! "
            "Please pass 'fastparquet', 'pyarrow', or 'auto'"
        )
    elif engine_str not in ("fastparquet", "pyarrow"):
        raise ValueError(f"{engine_str} engine not supported by `fsspec.parquet`")
    else:
        try_engines = [engine_str]

    # Try importing the engines in `try_engines`,
    # and choose the first one that succeeds
    for engine in try_engines:
        try:
            if engine == "fastparquet":
                return FastparquetEngine()
            elif engine == "pyarrow":
                return PyarrowEngine()
        except ImportError:
            pass

    # Raise an error if a supported parquet engine
    # was not found
    raise ImportError(
        f"The following parquet engines are not installed "
        f"in your python environment: {try_engines}."
        f"Please install 'fastparquert' or 'pyarrow' to "
        f"utilize the `fsspec.parquet` module."
    )


class FastparquetEngine:
    # The purpose of the FastparquetEngine class is
    # to check if fastparquet can be imported (on initialization)
    # and to define a `_parquet_byte_ranges` method. In the
    # future, this class may also be used to define other
    # methods/logic that are specific to fastparquet.

    def __init__(self):
        import fastparquet as fp

        self.fp = fp

    def _row_group_filename(self, row_group, pf):
        return pf.row_group_filename(row_group)

    def _parquet_byte_ranges(
        self,
        columns,
        row_groups=None,
        metadata=None,
        footer=None,
        footer_start=None,
    ):
        # Initialize offset ranges and define ParqetFile metadata
        pf = metadata
        data_paths, data_starts, data_ends = [], [], []
        if pf is None:
            pf = self.fp.ParquetFile(io.BytesIO(footer))

        # Convert columns to a set and add any index columns
        # specified in the pandas metadata (just in case)
        column_set = None if columns is None else set(columns)
        if column_set is not None and hasattr(pf, "pandas_metadata"):
            md_index = [
                ind
                for ind in pf.pandas_metadata.get("index_columns", [])
                # Ignore RangeIndex information
                if not isinstance(ind, dict)
            ]
            column_set |= set(md_index)

        # Check if row_groups is a list of integers
        # or a list of row-group metadata
        if row_groups and not isinstance(row_groups[0], int):
            # Input row_groups contains row-group metadata
            row_group_indices = None
        else:
            # Input row_groups contains row-group indices
            row_group_indices = row_groups
            row_groups = pf.row_groups

        # Loop through column chunks to add required byte ranges
        for r, row_group in enumerate(row_groups):
            # Skip this row-group if we are targeting
            # specific row-groups
            if row_group_indices is None or r in row_group_indices:
                # Find the target parquet-file path for `row_group`
                fn = self._row_group_filename(row_group, pf)

                for column in row_group.columns:
                    name = column.meta_data.path_in_schema[0]
                    # Skip this column if we are targeting a
                    # specific columns
                    if column_set is None or name in column_set:
                        file_offset0 = column.meta_data.dictionary_page_offset
                        if file_offset0 is None:
                            file_offset0 = column.meta_data.data_page_offset
                        num_bytes = column.meta_data.total_compressed_size
                        if footer_start is None or file_offset0 < footer_start:
                            data_paths.append(fn)
                            data_starts.append(file_offset0)
                            data_ends.append(
                                min(
                                    file_offset0 + num_bytes,
                                    footer_start or (file_offset0 + num_bytes),
                                )
                            )

        if metadata:
            # The metadata in this call may map to multiple
            # file paths. Need to include `data_paths`
            return data_paths, data_starts, data_ends
        return data_starts, data_ends


class PyarrowEngine:
    # The purpose of the PyarrowEngine class is
    # to check if pyarrow can be imported (on initialization)
    # and to define a `_parquet_byte_ranges` method. In the
    # future, this class may also be used to define other
    # methods/logic that are specific to pyarrow.

    def __init__(self):
        import pyarrow.parquet as pq

        self.pq = pq

    def _row_group_filename(self, row_group, metadata):
        raise NotImplementedError

    def _parquet_byte_ranges(
        self,
        columns,
        row_groups=None,
        metadata=None,
        footer=None,
        footer_start=None,
    ):
        if metadata is not None:
            raise ValueError("metadata input not supported for PyarrowEngine")

        data_starts, data_ends = [], []
        md = self.pq.ParquetFile(io.BytesIO(footer)).metadata

        # Convert columns to a set and add any index columns
        # specified in the pandas metadata (just in case)
        column_set = None if columns is None else set(columns)
        if column_set is not None:
            schema = md.schema.to_arrow_schema()
            has_pandas_metadata = (
                schema.metadata is not None and b"pandas" in schema.metadata
            )
            if has_pandas_metadata:
                md_index = [
                    ind
                    for ind in json.loads(
                        schema.metadata[b"pandas"].decode("utf8")
                    ).get("index_columns", [])
                    # Ignore RangeIndex information
                    if not isinstance(ind, dict)
                ]
                column_set |= set(md_index)

        # Loop through column chunks to add required byte ranges
        for r in range(md.num_row_groups):
            # Skip this row-group if we are targeting
            # specific row-groups
            if row_groups is None or r in row_groups:
                row_group = md.row_group(r)
                for c in range(row_group.num_columns):
                    column = row_group.column(c)
                    name = column.path_in_schema
                    # Skip this column if we are targeting a
                    # specific columns
                    split_name = name.split(".")[0]
                    if (
                        column_set is None
                        or name in column_set
                        or split_name in column_set
                    ):
                        file_offset0 = column.dictionary_page_offset
                        if file_offset0 is None:
                            file_offset0 = column.data_page_offset
                        num_bytes = column.total_compressed_size
                        if file_offset0 < footer_start:
                            data_starts.append(file_offset0)
                            data_ends.append(
                                min(file_offset0 + num_bytes, footer_start)
                            )
        return data_starts, data_ends
