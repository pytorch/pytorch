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
`inductor-cache-v1.sqlite3` inside `TORCHINDUCTOR_CACHE_DIR`; when that variable is unset,
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
transient `inductor-cache-v1.sqlite3-journal` can exist during writes. Database failures are
reported rather than silently switching to file storage. Corrupt autotune JSON
continues to be treated as a cache miss.

## Shared filesystems and read-only snapshots

Multiple processes on one node can share a database on local storage. A shared
filesystem is not made local merely because only one node currently accesses it;
SQLite still relies on the filesystem's locking and synchronization behavior.
The backend does not detect filesystem types or certify NFS/Lustre deployments.
See [SQLite's network-filesystem guidance](https://sqlite.org/useovernet.html).

Lookups open only existing databases, without creating tables or cache files on
a miss. SQLite can still need writable access to recover a hot rollback journal
after an interrupted writer. A complete compilation can also write source records,
autotune results, native files, or reconstructed FX-graph artifacts. Pointing
`TORCHINDUCTOR_CACHE_DIR` at a read-only directory is therefore unsupported.
Read-only permissions for one reader also do not prevent another client writing.

For multi-node reuse, stop all writers and close their connections before copying
the database, or create a consistent snapshot using SQLite's backup API. Publish
that snapshot on shared storage, then stage a separate writable copy into each
node's local cache before starting workers. Use node-local temporary storage too.
Do not copy a live database without its required transactional state, and do not
merge node copies by overwriting one another. A read-only base with automatic local
write-through is not implemented. SQLite's `immutable=1` option is deliberately not
enabled: it disables locking/change detection and requires a guarantee that the
file cannot change, not just that this process cannot write it.

The filename's `v1` identifies this backend's on-disk schema, not the PyTorch or
SQLite release. The `inductor-` prefix prevents collisions with Triton's distinct
schema when both cache roots happen to be the same directory. Earlier experimental
`cache-v1.sqlite3` files are ignored; entries are rebuilt, and the old file can be
removed after stopping its users.

To clear SQLite entries, stop all processes using the cache and remove
`inductor-cache-v1.sqlite3` and any associated journal files from the configured cache
directory. Do not remove a journal while a writer is active. Other cache files
can be cleared separately. `PyCodeCache.cache_clear(purge=True)` also removes
SQLite source entries corresponding to its loaded modules.

## Measuring storage overhead

Run `python benchmarks/dynamo/bench_sqlite_cache.py --entries 5000 --directory /path/to/local/storage`
to compare 5,000 Python sources and 5,000 autotune records with separate write
and restart-read processes. One local run measured:

| Backend | Persistent objects | Write seconds | Restart-read seconds | Live temporary objects |
| --- | ---: | ---: | ---: | ---: |
| File | 11,019 | 0.86 | 0.34 | 0 |
| SQLite | 1 | 6.87 | 2.42 | 6,020 |

Temporary objects returned to zero after normal exit. Counts exclude containing
directories and cache classes outside this backend's scope. Timings exclude
imports and kernel compilation; they are storage-only measurements, not an
end-to-end performance prediction. SQLite durability and materialization add
overhead. Re-run on the intended local filesystem.
