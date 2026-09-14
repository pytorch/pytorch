# Owner(s): ["module: inductor"]
import inspect
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from torch._inductor.codecache import PyCodeCache
from torch._inductor.remote_cache import (
    LocalAutotuneCache,
    LocalCacheBackend,
    SQLiteLocalCacheBackend,
    create_local_cache_backend,
)
from torch._inductor.runtime.sqlite_cache import SQLiteCache, sqlite_cache_enabled


class TestSQLiteCache(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = mock.patch.dict(
            os.environ,
            {
                "TORCHINDUCTOR_CACHE_DIR": str(self.root),
                "TORCHINDUCTOR_CACHE_BACKEND": "sqlite",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        PyCodeCache.cache_clear()
        self.addCleanup(PyCodeCache.cache_clear)
        self.cache = SQLiteCache(str(self.root))

    def test_bytes_namespaces_and_overwrite(self):
        self.assertIsNone(self.cache.get("a", "missing"))
        self.cache.put("a", "same", b"\x00\xff")
        self.cache.put("b", "same", "hello λ".encode())
        self.cache.put("a", "same", b"ignored", overwrite=False)
        self.assertEqual(self.cache.get("a", "same"), b"\x00\xff")
        self.assertEqual(self.cache.get("b", "same"), "hello λ".encode())
        self.cache.put("a", "same", b"updated")
        self.assertEqual(self.cache.get("a", "same"), b"updated")
        self.cache.delete("a", "same")
        self.assertIsNone(self.cache.get("a", "same"))

    def test_default_backend(self):
        os.environ.pop("TORCHINDUCTOR_CACHE_BACKEND")
        self.assertFalse(sqlite_cache_enabled())
        self.assertIsInstance(create_local_cache_backend(), LocalCacheBackend)
        key, path = PyCodeCache.write("value = 42\n")
        self.assertTrue(Path(path).is_relative_to(self.root))
        self.assertEqual(PyCodeCache.load_by_key_path(key, path).value, 42)
        cache = LocalAutotuneCache("test")
        name = str(self.root / "x.best_config")
        cache.put(name, {"num_warps": 4})
        self.assertTrue(Path(name).exists())
        self.assertEqual(cache.get(name), {"num_warps": 4})
        self.assertFalse(Path(self.cache.database).exists())
        os.environ["TORCHINDUCTOR_CACHE_BACKEND"] = "unknown"
        with self.assertRaisesRegex(ValueError, "TORCHINDUCTOR_CACHE_BACKEND"):
            create_local_cache_backend()

    def test_autotune_direct_storage_and_key_isolation(self):
        self.assertIsInstance(create_local_cache_backend(), SQLiteLocalCacheBackend)
        cache = LocalAutotuneCache("test")
        a = str(self.root / "aa" / "x.best_config")
        b = str(self.root / "bb" / "x.best_config")
        cache.put(a, {"num_warps": 4})
        cache.put(b, {"num_warps": 8})
        self.assertEqual(cache.get(a), {"num_warps": 4})
        self.assertEqual(cache.get(b), {"num_warps": 8})
        self.assertFalse(Path(a).exists())
        self.assertEqual(len(list(self.root.rglob("*"))), 1)
        self.cache.put("autotune", "aa/x.best_config", b"broken json")
        self.assertIsNone(cache.get(a))

    def test_python_paths_source_attrs_and_purge(self):
        source = "def answer():\n    return 42\n"
        key, path = PyCodeCache.write(source)
        self.assertFalse(Path(path).is_relative_to(self.root))
        self.assertEqual(self.cache.get("python", key), source.encode())
        module = PyCodeCache.load_by_key_path(key, path, attrs={"constant": 7})
        self.assertEqual(module.answer(), 42)
        self.assertEqual(module.constant, 7)
        self.assertEqual(inspect.getsource(module.answer), source)
        self.assertIs(PyCodeCache.load(source), PyCodeCache.load(source))
        Path(path).unlink()
        self.assertEqual(Path(self.cache.materialize_python(key)).read_text(), source)
        self.assertEqual(
            self.cache.autotune_key(str(Path(path).with_suffix(".best_config"))),
            f"{key[1:3]}/{key}.best_config",
        )
        PyCodeCache.cache_clear(purge=True)
        self.assertIsNone(self.cache.get("python", key))
        self.assertFalse(Path(path).exists())

    def test_restart_cleanup_and_autotune_reuse(self):
        script = """
import json
from pathlib import Path
from torch._inductor.codecache import PyCodeCache
from torch._inductor.remote_cache import LocalAutotuneCache
key, path = PyCodeCache.write('answer = 42')
cache = LocalAutotuneCache('test')
cache.put(str(Path(path).with_suffix('.best_config')), {'num_warps': 4})
print(json.dumps([key, path]))
"""
        key, previous = json.loads(
            subprocess.check_output([sys.executable, "-c", script], text=True)
        )
        self.assertFalse(Path(previous).exists())
        module = PyCodeCache.load_by_key_path(key, previous)
        self.assertEqual(module.answer, 42)
        cache = LocalAutotuneCache("test")
        name = str(Path(module.__file__).with_suffix(".best_config"))
        self.assertEqual(cache.get(name), {"num_warps": 4})

    def test_concurrent_processes_and_interrupted_transaction(self):
        script = """
import sys
from torch._inductor.runtime.sqlite_cache import local_cache
c = local_cache()
for i in range(30):
    c.put('test', str(i), bytes([int(sys.argv[1])]) * 4096)
"""
        workers = [
            subprocess.Popen([sys.executable, "-c", script, str(i)]) for i in range(4)
        ]
        for worker in workers:
            self.assertEqual(worker.wait(timeout=120), 0)
        for i in range(30):
            self.assertIn(
                self.cache.get("test", str(i)), [bytes([n]) * 4096 for n in range(4)]
            )
        script = """
import os, sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute('BEGIN IMMEDIATE')
c.execute('DELETE FROM entries')
os._exit(0)
"""
        subprocess.run(
            [sys.executable, "-c", script, self.cache.database], check=True, timeout=30
        )
        with sqlite3.connect(self.cache.database) as connection:
            self.assertEqual(
                connection.execute("PRAGMA integrity_check").fetchone(), ("ok",)
            )
            self.assertEqual(
                connection.execute("SELECT count(*) FROM entries").fetchone()[0], 30
            )

    def test_corrupt_database_fails_without_file_fallback(self):
        Path(self.cache.database).write_bytes(b"not a database")
        with self.assertRaisesRegex(RuntimeError, "SQLite cache failure"):
            PyCodeCache.write("value = 1")
        self.assertEqual(len(list(self.root.rglob("*"))), 1)

    def test_threads_and_database_recreation(self):
        def put(i):
            self.cache.put("threads", str(i), str(i).encode())
            return self.cache.get("threads", str(i))

        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(
                list(pool.map(put, range(40))), [str(i).encode() for i in range(40)]
            )
        Path(self.cache.database).unlink()
        self.assertIsNone(self.cache.get("threads", "0"))
        self.cache.put("threads", "new", b"recovered")
        self.assertEqual(self.cache.get("threads", "new"), b"recovered")

    def test_inode_reduction(self):
        for i in range(200):
            self.cache.put("autotune", f"{i}.best_config", b"{}")
            PyCodeCache.write(f"value = {i}\n")
        self.assertEqual(len(list(self.root.rglob("*"))), 1)
        with sqlite3.connect(self.cache.database) as connection:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM entries").fetchone()[0], 400
            )


if __name__ == "__main__":
    unittest.main()
