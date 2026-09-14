# Optional SQLite local cache

TorchInductor can store local autotuning records (`.best_config`) and generated
`PyCodeCache` Python source in SQLite to reduce persistent filesystem inode use.
This experimental backend is opt-in; the filesystem backend remains the default.

```sh
export TORCHINDUCTOR_CACHE_BACKEND=sqlite
export TORCHINDUCTOR_CACHE_DIR=/path/to/local/cache/inductor
export TMPDIR=/path/to/local/runtime
python train.py
```

Set configuration before starting Python and compiler workers. The database is
`cache-v1.sqlite3` inside `TORCHINDUCTOR_CACHE_DIR`; when that variable is unset,
the existing default cache directory is used. `TORCHINDUCTOR_CACHE_BACKEND=file`
or an unset backend variable selects the original filesystem behavior. Unknown
backend names raise an error. Old file entries are not migrated or read by SQLite.

Autotuning records use direct byte storage with separate key namespaces. Python
sources still need paths for module loading, source inspection, debugging, worker
processes, and FX-graph reconstruction. They are materialized on use below Python's
temporary directory and removed at normal process exit. Temporary paths are not
portable cache identifiers: reload through `PyCodeCache` using the content key.

This reduces **persistent** inode usage; it does not bound temporary inode use.
Use a temporary filesystem with sufficient inodes outside the constrained cache
directory. Runtime files remain available for the process lifetime. Abnormal
termination may leave `torchinductor-sqlite-*` directories; remove them only after
the owning processes have stopped. No automatic eviction or size limit is provided.

Native objects, compilation locks, FX-graph entries, AOTAutograd entries, and other
cache classes retain their existing file storage. Total savings depend on the
workload's mix of cache objects. Triton has a separate cache and configuration.
Remote caches and their serialization remain unchanged.

Use a local filesystem for the database and temporary files. A database shared
across hosts on NFS, Lustre, or other distributed filesystems is unsupported.
SQLite transactions and a 30-second busy timeout coordinate concurrent local
processes. Each operation closes its connection, so no connection is inherited
by a worker or shared across threads. The default rollback journal is used; a
transient `cache-v1.sqlite3-journal` can exist during writes. Database failures are
reported rather than silently switching to file storage. Corrupt autotune JSON
continues to be treated as a cache miss.

To clear SQLite entries, stop all processes using the cache and remove
`cache-v1.sqlite3` and any associated journal files from the configured cache
directory. Do not remove a journal while a writer is active. Other cache files
can be cleared separately. `PyCodeCache.cache_clear(purge=True)` also removes
SQLite source entries corresponding to its loaded modules.

## Measuring storage overhead

Run `python benchmarks/dynamo/bench_sqlite_cache.py --entries 5000 --directory /path/to/local/storage`
to compare 5,000 Python sources and 5,000 autotune records with separate write
and restart-read processes. One local run measured:

| Backend | Persistent objects | Write seconds | Restart-read seconds | Peak temporary objects |
| --- | ---: | ---: | ---: | ---: |
| File | 11,019 | 0.94 | 0.36 | 0 |
| SQLite | 1 | 6.44 | 2.38 | 6,020 |

Temporary objects returned to zero after normal exit. Counts exclude containing
directories and cache classes outside this backend's scope. Timings exclude
imports and kernel compilation; they are storage-only measurements, not an
end-to-end performance prediction. SQLite durability and materialization add
overhead. Re-run on the intended local filesystem.
