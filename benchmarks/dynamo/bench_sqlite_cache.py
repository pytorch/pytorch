"""Compare Python source/autotune persistence and restart hits without a GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def worker(args):
    from torch._inductor.codecache import get_hash, get_path, PyCodeCache
    from torch._inductor import remote_cache

    factory = getattr(
        remote_cache, "create_local_cache_backend", remote_cache.LocalCacheBackend
    )
    backend = factory()
    start = time.perf_counter()
    for i in range(args.entries):
        source = f"value = {i}\n"
        if args.worker == "write":
            key, path = PyCodeCache.write(source)
            backend.put(
                str(Path(path).with_suffix(".best_config")), b'{"num_warps": 4}'
            )
        else:
            key = get_hash(source.strip())
            path = get_path(key, "py")[2]
            module = PyCodeCache.load_by_key_path(key, path)
            assert module.value == i
            assert (
                backend.get(str(Path(module.__file__).with_suffix(".best_config")))
                == b'{"num_warps": 4}'
            )
    seconds = time.perf_counter() - start
    runtime_objects = len(list(Path(os.environ["TMPDIR"]).rglob("*")))
    print(json.dumps({"seconds": seconds, "runtime_objects": runtime_objects}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, default=1000)
    parser.add_argument(
        "--directory", help="Local parent directory for temporary benchmark storage"
    )
    parser.add_argument("--worker", choices=("write", "read"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    results = {}
    with tempfile.TemporaryDirectory(dir=args.directory) as temporary:
        for backend in ("file", "sqlite"):
            root = Path(temporary) / backend
            cache, runtime = root / "cache", root / "runtime"
            runtime.mkdir(parents=True)
            env = dict(
                os.environ,
                TORCHINDUCTOR_CACHE_BACKEND=backend,
                TORCHINDUCTOR_CACHE_DIR=str(cache),
                TMPDIR=str(runtime),
            )
            phases = {}
            for phase in ("write", "read"):
                output = subprocess.check_output(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        phase,
                        "--entries",
                        str(args.entries),
                    ],
                    env=env,
                    text=True,
                )
                phases[phase] = json.loads(output)
            phases["persistent_objects"] = len(list(cache.rglob("*")))
            phases["runtime_objects_after_exit"] = len(list(runtime.rglob("*")))
            results[backend] = phases
    print(
        json.dumps(
            {
                "sources": args.entries,
                "logical_entries": args.entries * 2,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
