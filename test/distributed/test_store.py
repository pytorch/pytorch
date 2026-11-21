# Owner(s): ["oncall: distributed"]

import ctypes
import datetime
import hashlib
import os
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from sys import platform
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.distributed.rpc as rpc
from torch.distributed import DistError, DistNetworkError, DistStoreError
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests


if not dist.is_available():
    print("torch.distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import (
    create_tcp_store,
    skip_if_win32,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    _restore_fp32_precision,
    _snapshot_fp32_precision,
    ADDRESS_IN_USE,
    CONNECT_TIMEOUT,
    HardwareClassification,
    load_tests,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    set_warn_always_context,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"

DEFAULT_HOSTNAME = "localhost"


_PRIOR_FP32_PRECISION: tuple[str, ...] | None = None


def setUpModule():
    global _PRIOR_FP32_PRECISION
    # allow_tf32 writes both the legacy Float32MatmulPrecision enum and the
    # backend-specific fp32_precision, so snapshot and restore all of it.
    _PRIOR_FP32_PRECISION = _snapshot_fp32_precision()
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        _restore_fp32_precision(_PRIOR_FP32_PRECISION)
        _PRIOR_FP32_PRECISION = None


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class StoreTestBase:
    hw_classification = HardwareClassification.GENERIC

    def _create_store(self, i):
        raise RuntimeError("not implemented")

    def _test_set_get_check(self, fs):
        fs.add("key", 1)
        fs.add("key", 2)
        fs.add("key", 3)
        fs.set("key0", "value0")
        fs.add("key3", 1)
        fs.set("key1", "value1")
        fs.add("key3", 2)
        fs.set("key2", "value2")
        fs.add("key3", 3)
        fs.add("key3", 4)
        fs.add("key3", 5)
        fs.add("key3", 6)
        self.assertEqual(fs.num_keys(), self.num_keys_total)
        self.assertEqual(b"6", fs.get("key"))
        self.assertEqual(b"value0", fs.get("key0"))
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key2"))
        self.assertEqual(b"21", fs.get("key3"))
        self.assertTrue(fs.check(["key3"]))
        self.assertFalse(fs.check(["Randomkey3"]))

        fs.set("-key3", "7")
        self.assertEqual(b"7", fs.get("-key3"))
        fs.delete_key("-key3")
        self.assertEqual(fs.num_keys(), self.num_keys_total)

    def test_set_get_check(self):
        self._test_set_get_check(self._create_store())

    def _test_compare_set(self, store):
        missing_key_result = store.compare_set(
            "cs_key0", "wrong_old_value", "new_value0"
        )
        self.assertEqual(b"wrong_old_value", missing_key_result)

        store.set("cs_key0", "value0")
        self.assertEqual(b"value0", store.get("cs_key0"))
        old_value_result = store.compare_set("cs_key0", "wrong_old_value", "new_value0")
        self.assertEqual(b"value0", old_value_result)
        self.assertEqual(b"value0", store.get("cs_key0"))
        new_value_result = store.compare_set("cs_key0", "value0", "new_value0")
        self.assertEqual(b"new_value0", new_value_result)
        self.assertEqual(b"new_value0", store.get("cs_key0"))
        empty_old_value_result = store.compare_set("cs_key1", "", "new_value1")
        self.assertEqual(b"new_value1", empty_old_value_result)
        self.assertEqual(b"new_value1", store.get("cs_key1"))

    def test_compare_set(self):
        self._test_compare_set(self._create_store())

    def _test_simple_wait(self, fs):
        with self.assertRaisesRegex(RuntimeError, "[t -i]imeout"):
            fs.wait(["bad_key"], timedelta(milliseconds=1))
        fs.add("good_key", 1)
        fs.wait(["good_key"])

    def test_simple_wait(self):
        self._test_simple_wait(self._create_store())

    def _test_append(self, store):
        if not store.has_extended_api():
            # Just return for stores that don't support extended APIs.
            return
        store.set("foo", "po")
        store.append("foo", "tato")
        store.append("bar", "po")
        store.append("bar", "tato")
        self.assertEqual(b"potato", store.get("foo"))
        self.assertEqual(b"potato", store.get("bar"))

    def test_append(self):
        self._test_append(self._create_store())

    def _create_store_or_skip_if_no_queues(self) -> dist.Store:
        store = self._create_store()

        try:
            store.queue_push("test_queue_support", "1")
        except NotImplementedError:
            self.skipTest("Store does not support queues")

        return store

    def test_queues(self) -> None:
        store = self._create_store_or_skip_if_no_queues()

        self.assertFalse(store.check(["foo"]))
        self.assertEqual(store.queue_len("foo"), 0)

        store.queue_push("foo", "1")
        store.queue_push("foo", "2")

        self.assertTrue(store.check(["foo"]))
        self.assertEqual(store.queue_len("foo"), 2)
        store.wait(["foo"])

        self.assertEqual(store.queue_pop("foo"), b"1")
        self.assertEqual(store.queue_pop("foo"), b"2")

        self.assertFalse(store.check(["foo"]))
        self.assertEqual(store.queue_len("foo"), 0)

    def test_queues_nonblocking(self) -> None:
        store = self._create_store_or_skip_if_no_queues()

        with self.assertRaisesRegex(dist.QueueEmptyError, "empty"):
            store.queue_pop("foo", block=False)

        store.queue_push("foo", "a")
        self.assertEqual(store.queue_pop("foo", block=False), b"a")

    def test_queues_bidirectional(self) -> None:
        store = self._create_store_or_skip_if_no_queues()

        def worker_a():
            local_store = store.clone()

            local_store.queue_push("a", "a1")
            self.assertEqual(local_store.queue_pop("b"), b"b1")

        def worker_b():
            local_store = store.clone()

            self.assertEqual(local_store.queue_pop("a"), b"a1")
            local_store.queue_push("b", "b1")

        # test bidirectional communication
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(worker_a),
                pool.submit(worker_b),
            ]
            for fut in futures:
                fut.result()

    def test_queues_timeout(self) -> None:
        store = self._create_store_or_skip_if_no_queues()

        store.set_timeout(timedelta(seconds=0.01))
        with self.assertRaisesRegex(DistStoreError, "timeout"):
            store.queue_pop("non_existant")

    def _test_multi_set(self, store):
        if not store.has_extended_api():
            # Just return for stores that don't support extended APIs.
            return
        store.multi_set(["foo", "bar"], ["po", "tato"])
        self.assertEqual(b"po", store.get("foo"))
        self.assertEqual(b"tato", store.get("bar"))

    def test_multi_set(self):
        self._test_multi_set(self._create_store())

    def _test_multi_get(self, store):
        if not store.has_extended_api():
            # Just return for stores that don't support extended APIs.
            return
        store.set("foo", "po")
        store.set("bar", "tato")
        v0, v1 = store.multi_get(["foo", "bar"])
        self.assertEqual(b"po", v0)
        self.assertEqual(b"tato", v1)

    def test_multi_get(self):
        self._test_multi_get(self._create_store())

    def test_clone(self):
        a = self._create_store()
        b = a.clone()

        self.assertIsInstance(b, dist.Store)

        a.set("foo", "bar")
        self.assertEqual(b.get("foo"), b"bar")

    def test_list_keys(self):
        a = self._create_store()
        a.set("foo", "bar")
        a.set("baz", "qux")
        keys = a.list_keys()
        self.assertIn("foo", keys)
        self.assertIn("baz", keys)

    def _test_barrier(self, store):
        # Barrier with world_size=1 should return immediately
        store.barrier("test_barrier_key", 1)

    def test_barrier(self):
        self._test_barrier(self._create_store())

    # This is the number of keys used in test_set_get. Adding this as a class
    # property instead of hardcoding in the test since some Store
    # implementations will have differing number of keys. In the base case,
    # there will be 5 keys: key, key0, key1, key2, key3.
    @property
    def num_keys_total(self):
        return 5


class FileStoreTest(TestCase, StoreTestBase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f

    def _create_store(self):
        store = dist.FileStore(self.file.name, 1)
        store.set_timeout(timedelta(seconds=300))
        return store

    def test_init_pg_and_rpc_with_same_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as file:
            # Init RPC using file
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
            rpc_backend_options.init_method = f"file://{file.name}"
            rpc_backend_options._transports = tp_transports()
            rpc.init_rpc(
                "worker", rank=0, world_size=1, rpc_backend_options=rpc_backend_options
            )

            # Init PG using file
            dist.init_process_group(
                "gloo", rank=0, world_size=1, init_method=f"file://{file.name}"
            )
            dist.destroy_process_group()
            if not os.path.exists(file.name):
                raise AssertionError(f"Expected file {file.name} to exist")

            rpc.shutdown()
            os.remove(file.name)

    def test_refcount(self):
        with tempfile.NamedTemporaryFile(delete=False) as file:
            store = dist.FileStore(file.name, 1)
            store2 = dist.FileStore(file.name, 1)

            del store
            if not os.path.exists(file.name):
                raise AssertionError(f"Expected file {file.name} to exist")
            del store2
            if os.path.exists(file.name):
                raise AssertionError(f"Expected file {file.name} to not exist")

    @property
    def num_keys_total(self):
        return 6


@skip_if_win32()
class HashStoreTest(TestCase, StoreTestBase):
    hw_classification = HardwareClassification.GENERIC

    def _create_store(self):
        store = dist.HashStore()
        store.set_timeout(timedelta(seconds=300))
        return store


class PrefixStoreTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        # delete is false as FileStore will automatically clean up the file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f

    def test_get_underlying_store(self):
        tcp_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME, port=0, world_size=1, is_master=True
        )
        hash_store = dist.HashStore()
        file_store = dist.FileStore(self.file.name, world_size=1)
        for store in [tcp_store, hash_store, file_store]:
            with self.subTest(f"Testing getting underlying_store for {type(store)}"):
                prefix_store = dist.PrefixStore("prefix", store)
                self.assertEqual(prefix_store.underlying_store, store)

        # We do not allow passing in None as the underlying store, this would cause a segfault if used
        with self.assertRaises(ValueError):
            dist.PrefixStore("prefix", None)


class PrefixFileStoreTest(TestCase, StoreTestBase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f
        self.filestore = dist.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"
        self.filestore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return dist.PrefixStore(self.prefix, self.filestore)

    @property
    def num_keys_total(self):
        return 6


class TCPStoreTest(TestCase, StoreTestBase):
    hw_classification = HardwareClassification.GENERIC

    def _create_store(self):
        store = create_tcp_store()
        store.set_timeout(timedelta(seconds=300))
        return store

    def _create_store_with_ws(self, addr, world_size):
        return create_tcp_store(addr, world_size, wait_for_workers=False)

    def test_address_already_in_use(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        err_msg_reg = "^The server socket has failed to listen on any local .*(?i:address already in use)"
        with self.assertRaisesRegex(dist.DistNetworkError, err_msg_reg):
            # Use noqa to silence flake8.
            # Need to store in an unused variable here to ensure the first
            # object is not destroyed before the second object is created.
            store1 = dist.TCPStore(addr, port, 1, True)
            store2 = dist.TCPStore(addr, port, 1, True)
            self.assertTrue(store1.libuvBackend)
            self.assertTrue(store2.libuvBackend)

    @retry_on_connect_failures
    def test_multitenancy(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        # Use noqa to silence flake8.
        # Need to store in an unused variable here to ensure the first
        # object is not destroyed before the second object is created.
        store1 = dist.TCPStore(addr, port, 1, True, multi_tenant=True)  # type: ignore[call-arg]
        store2 = dist.TCPStore(addr, port, 1, True, multi_tenant=True)  # type: ignore[call-arg]
        self.assertTrue(store1.libuvBackend)
        self.assertTrue(store2.libuvBackend)

    def test_repr(self) -> None:
        # server
        store1 = self._create_store()
        self.assertRegex(
            repr(store1),
            r"TCPStore\("
            r"client=TCPClient\(SocketImpl\(fd=\d+, addr=\[?localhost\]?:\d+, remote=\[?localhost\]?:\d+\)\), "
            r"server=TCPServer\(port=\d+\)\)",
        )

        # client
        store2 = dist.TCPStore(
            store1.host,
            store1.port,
            world_size=2,
            is_master=False,
        )
        self.assertRegex(
            repr(store2),
            r"TCPStore\("
            r"client=TCPClient\(SocketImpl\(fd=\d+, addr=\[?localhost\]?:\d+, remote=\[?localhost\]?:\d+\)\), "
            r"server=<nullptr>\)",
        )

    @skip_if_win32()
    @retry_on_connect_failures
    def test_init_pg_and_rpc_with_same_socket(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        # retry_on_connect_failures re-runs this whole body on a RuntimeError,
        # so every mutation below has to be undone before the next attempt.
        try:
            os.environ["MASTER_ADDR"] = addr
            os.environ["MASTER_PORT"] = str(port)

            # We internally use a multi-tenant TCP store. Both PG and RPC should successfully
            # initialize even when using the same socket address.

            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                rank=0,
                world_size=1,
            )

            backend_opts = rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{addr}:{port}", _transports=tp_transports()
            )
            rpc.init_rpc(
                name="worker0",
                rank=0,
                world_size=1,
                rpc_backend_options=backend_opts,
            )

            rpc.shutdown()
            dist.destroy_process_group()
        finally:
            # A failure can land inside init_rpc, so do not wait on a possibly
            # half-initialized agent, and tear the group down even if this raises.
            try:
                if rpc.api._is_current_rpc_agent_set():
                    rpc.shutdown(graceful=False)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()
                for var in ("MASTER_ADDR", "MASTER_PORT"):
                    os.environ.pop(var, None)

    @skip_if_win32()
    def test_take_over_listen_socket(self):
        listen_sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.bind(("localhost", 0))
        addr, port, *_ = listen_sock.getsockname()
        listen_fd = listen_sock.detach()

        store = dist.TCPStore(addr, port, 1, is_master=True, master_listen_fd=listen_fd)

        self.assertTrue(store.libuvBackend)
        store.set("key", "value")
        self.assertEqual(b"value", store.get("key"))

    # The TCPStore has 6 keys in test_set_get. It contains the 5 keys added by
    # the user and one additional key used for coordinate all the workers.
    @property
    def num_keys_total(self):
        return 6

    def _test_numkeys_delkeys(self, fs):
        # We start off with one init key in the store to coordinate workers
        self.assertEqual(fs.num_keys(), 1)
        fs.add("key", 1)
        fs.add("key", 2)
        fs.add("key", 3)
        fs.set("key0", "value0")
        fs.add("key3", 1)
        fs.set("key1", "value1")
        self.assertEqual(fs.num_keys(), 5)
        fs.delete_key("key")
        self.assertEqual(fs.num_keys(), 4)
        fs.set_timeout(timedelta(milliseconds=1))
        with self.assertRaises(RuntimeError):
            fs.get("key")
        fs.set_timeout(timedelta(seconds=2))
        fs.delete_key("key0")
        fs.delete_key("key3")
        self.assertEqual(fs.num_keys(), 2)
        fs.set("key4", "value2")
        self.assertEqual(fs.num_keys(), 3)
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key4"))

    def test_numkeys_delkeys(self):
        self._test_numkeys_delkeys(self._create_store())

    def _create_client(self, index, addr, port, world_size):
        client_store = dist.TCPStore(
            addr, port, world_size=world_size, timeout=timedelta(seconds=10)
        )
        self.assertEqual(b"value", client_store.get("key"))
        client_store.set(f"new_key{index}", f"new_value{index}")
        self.assertEqual(
            f"next_value{index}".encode(),
            client_store.compare_set(
                f"new_key{index}", f"new_value{index}", f"next_value{index}"
            ),
        )

    def _multi_worker_helper(self, world_size):
        addr = DEFAULT_HOSTNAME
        server_store = self._create_store_with_ws(addr, world_size)
        self.assertTrue(server_store.libuvBackend)
        server_store.set("key", "value")
        port = server_store.port

        num_indices = world_size if world_size else 1
        for i in range(num_indices):
            self._create_client(i, addr, port, world_size)

    def test_multi_worker_with_fixed_world_size(self):
        self._multi_worker_helper(5)

    def test_multi_worker_with_nonfixed_world_size(self):
        self._multi_worker_helper(None)

    def test_append(self):
        store = self._create_store()
        self.assertTrue(store.libuvBackend)
        store.set("foo", "po")
        store.append("foo", "tato")
        store.append("bar", "po")
        store.append("bar", "tato")
        self.assertEqual(b"potato", store.get("foo"))
        self.assertEqual(b"potato", store.get("bar"))

    def test_multi_set(self):
        store = self._create_store()
        self.assertTrue(store.libuvBackend)
        store.multi_set(["foo", "bar"], ["po", "tato"])
        self.assertEqual(b"po", store.get("foo"))
        self.assertEqual(b"tato", store.get("bar"))

    def test_multi_get(self):
        store = self._create_store()
        self.assertTrue(store.libuvBackend)
        store.set("foo", "po")
        store.set("bar", "tato")
        v0, v1 = store.multi_get(["foo", "bar"])
        self.assertEqual(b"po", v0)
        self.assertEqual(b"tato", v1)

    def test_store_timeout_on_missing_clients(self):
        with self.assertRaisesRegex(
            DistStoreError,
            r"Timed out after \d+ seconds waiting for clients. \d+/\d+ clients joined.",
        ):
            # world_size is 2 so it should timeout
            dist.TCPStore("localhost", 0, 2, True, timeout=timedelta(seconds=2))

        # when wait_for_workers is not set, then there should be no exception raised
        dist.TCPStore(
            "localhost",
            0,
            2,
            True,
            timeout=timedelta(seconds=2),
            wait_for_workers=False,
        )

    @skip_if_win32()
    def test_world_size_0_raises(self):
        with self.assertRaisesRegex(ValueError, "TCPStore world size cannot be 0"):
            dist.TCPStore("localhost", 0, world_size=0, is_master=False)

    def test_agent_store(self) -> None:
        store = self._create_store()

        with self.assertRaisesRegex(
            dist.DistNetworkError,
            "The server socket has failed to listen on any local network address",
        ):
            dist.TCPStore(
                host_name="localhost", port=store.port, world_size=1, is_master=True
            )

        USE_AGENT_STORE = "TORCHELASTIC_USE_AGENT_STORE"
        MASTER_PORT = "MASTER_PORT"

        os.environ[USE_AGENT_STORE] = "1"
        os.environ[MASTER_PORT] = str(store.port)
        try:
            second_server = dist.TCPStore(
                host_name="localhost", port=store.port, world_size=1, is_master=True
            )
        finally:
            os.environ.pop(USE_AGENT_STORE, None)
            os.environ.pop(MASTER_PORT, None)

        self.assertEqual(second_server.port, store.port)

    def test_barrier(self):
        store = self._create_store()
        # Single worker barrier should return immediately
        store.barrier("test_barrier_key", 1)

    def test_barrier_with_timeout(self):
        store = self._create_store()
        store.barrier("test_barrier_timeout_key", 1, timedelta(seconds=10))

    def test_barrier_timeout_expires(self):
        store = self._create_store()
        with self.assertRaisesRegex(DistStoreError, "barrier timeout"):
            store.barrier("test_barrier_fail", 2, timedelta(milliseconds=1))

    def test_barrier_multi_worker(self):
        server_store = self._create_store()
        world_size = 4

        def worker(rank):
            if rank == 0:
                worker_store = server_store
            else:
                worker_store = dist.TCPStore(
                    host_name=server_store.host,
                    port=server_store.port,
                    is_master=False,
                    wait_for_workers=False,
                )
            worker_store.barrier("multi_barrier", world_size, timedelta(seconds=10))

        with ThreadPoolExecutor(max_workers=world_size) as pool:
            futures = [pool.submit(worker, i) for i in range(world_size)]
            for f in futures:
                f.result()


def _load_libssl():
    for name in ("libssl.so.3", "libssl.so.1.1"):
        try:
            # SSL_CTX_set_num_tickets excludes OpenSSL 1.1.0, like the C++ loader.
            lib = ctypes.CDLL(name)
            lib.OpenSSL_version_num.restype = ctypes.c_ulong
            lib.OpenSSL_version.restype = ctypes.c_char_p
            if (
                lib.OpenSSL_version_num() >= 0x10101000
                and lib.OpenSSL_version(0).startswith(b"OpenSSL ")
                and hasattr(lib, "SSL_CTX_set_num_tickets")
            ):
                return True
        except (OSError, AttributeError):
            pass
    return False


def _openssl(*args):
    res = subprocess.run(["openssl", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"openssl {args[0]} failed: {res.stderr}")


def _make_ca(d, name):
    key = os.path.join(d, f"{name}.key")
    cert = os.path.join(d, f"{name}.pem")
    _openssl(
        "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
        "-nodes", "-keyout", key, "-out", cert, "-days", "1", "-subj", f"/CN={name}",
    )  # fmt: skip
    return cert, key


def _make_cert(d, name, ca, key_args=None):
    key = os.path.join(d, f"{name}.key")
    csr = os.path.join(d, f"{name}.csr")
    cert = os.path.join(d, f"{name}.pem")
    _openssl(
        "req", "-newkey", *(key_args or ["ec", "-pkeyopt", "ec_paramgen_curve:prime256v1"]),
        "-nodes", "-keyout", key, "-out", csr, "-subj", f"/CN={name}",
    )  # fmt: skip
    _openssl(
        "x509", "-req", "-in", csr, "-CA", ca[0], "-CAkey", ca[1],
        "-set_serial", "1", "-out", cert, "-days", "1",
    )  # fmt: skip
    return cert, key


@unittest.skipUnless(
    sys.platform == "linux" and sys.maxsize > 2**32,
    "TCPStore TLS requires 64-bit Linux",
)
class TCPStoreTLSTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if shutil.which("openssl") is None or not _load_libssl():
            raise unittest.SkipTest("openssl CLI or libssl not available")
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmpdir.cleanup)
        d = cls._tmpdir.name
        ca = _make_ca(d, "ca")
        cls.ca = ca[0]
        cls.server = _make_cert(d, "server", ca)
        cls.client = _make_cert(d, "client", ca)
        other_ca = _make_ca(d, "other_ca")
        cls.other_ca = other_ca[0]
        cls.untrusted = _make_cert(d, "untrusted", other_ca)
        # Certificate and key in one file.
        cls.combined = os.path.join(d, "combined.pem")
        with open(cls.combined, "w") as out:
            for path in cls.client:
                with open(path) as f:
                    out.write(f.read())
        cls.encrypted_key = os.path.join(d, "encrypted.key")
        _openssl(
            "pkey", "-in", cls.client[1], "-aes256", "-passout", "pass:secret",
            "-out", cls.encrypted_key,
        )  # fmt: skip
        cls.psk = os.urandom(16).hex()
        cls.other_psk = os.urandom(32).hex()

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("CERT", "KEY", "CA", "PSK", "PSK_FILE", "REQUIRE"):
            os.environ.pop(f"TORCH_TCPSTORE_TLS_{name}", None)

    def _env(self, cert, ca=None):
        return {
            "TORCH_TCPSTORE_TLS_CERT": cert[0],
            "TORCH_TCPSTORE_TLS_KEY": cert[1],
            "TORCH_TCPSTORE_TLS_CA": ca or self.ca,
        }

    def _tls(self, mode, cert):
        if mode == "psk":
            return {"tls_psk": self.psk}
        return {"tls_cert": cert[0], "tls_key": cert[1], "tls_ca": self.ca}

    def _store(
        self,
        port=0,
        is_master=True,
        cert=None,
        ca=None,
        timeout=30,
        use_libuv=True,
        tls=None,
        multi_tenant=False,
    ):
        if cert is not None:
            tls = {"tls_cert": cert[0], "tls_key": cert[1], "tls_ca": ca or self.ca}
        return dist.TCPStore(
            "localhost",
            port,
            2,
            is_master,
            timeout=timedelta(seconds=timeout),
            wait_for_workers=False,
            use_libuv=use_libuv,
            multi_tenant=multi_tenant,
            **(tls or {}),
        )

    def _assert_fails_fast(self, regex, **kwargs):
        start = time.monotonic()
        with self.assertRaisesRegex(DistStoreError, regex):
            self._store(**kwargs)
        self.assertLess(time.monotonic() - start, 10)

    def test_tls_ops(self):
        server = self._store(cert=self.server)
        client = self._store(server.port, False, cert=self.client)
        client.set("key", "value")
        self.assertEqual(server.get("key"), b"value")
        self.assertEqual(client.add("counter", 2), 2)
        self.assertEqual(server.add("counter", 3), 5)
        client.multi_set(["a", "b"], ["1", "2"])
        self.assertEqual(server.multi_get(["a", "b"]), [b"1", b"2"])
        self.assertEqual(client.compare_set("a", "1", "3"), b"3")
        self.assertTrue(client.check(["a", "b"]))
        self.assertFalse(client.check(["missing"]))
        self.assertTrue(client.delete_key("b"))

        with self.assertRaisesRegex(DistStoreError, "wait timeout"):
            client.wait(["waited"], timedelta(milliseconds=10))
        server.set("waited", "yes")
        client.wait(["waited"])
        self.assertEqual(client.get("waited"), b"yes")

        # Spans multiple TLS write chunks.
        large = os.urandom(3 * 1024 * 1024 + 7)
        client.set("large", large)
        # assertEqual compares bytes elementwise, which is slow.
        digest = hashlib.sha256(large).hexdigest()
        for store in (server, client):
            value = store.get("large")
            self.assertEqual(len(value), len(large))
            self.assertEqual(hashlib.sha256(value).hexdigest(), digest)

        clone = client.clone()
        self.assertEqual(clone.get("key"), b"value")

        client.queue_push("q", "x")
        self.assertEqual(server.queue_len("q"), 1)
        self.assertEqual(clone.queue_pop("q"), b"x")

    @parametrize("mode", ["cert", "psk"])
    def test_tls_env(self, mode):
        env = (
            {"TORCH_TCPSTORE_TLS_PSK": self.psk}
            if mode == "psk"
            else self._env((self.combined, self.combined))
        )
        with mock.patch.dict(os.environ, env):
            server = self._store()
            client = self._store(server.port, False)
            client.set("key", "value")
            self.assertEqual(server.get("key"), b"value")
        # Clones keep the resolved settings.
        self.assertEqual(client.clone().get("key"), b"value")
        # Without the env vars the client is plaintext and is rejected.
        self._assert_fails_fast(
            "Server requires TLS", port=server.port, is_master=False
        )

    @parametrize("mode", ["cert", "psk"])
    def test_tls_options_override_env(self, mode):
        env = (
            self._env(self.server)
            if mode == "cert"
            else {"TORCH_TCPSTORE_TLS_PSK": self.psk}
        )
        empty = (
            {"tls_cert": "", "tls_key": "", "tls_ca": ""}
            if mode == "cert"
            else {"tls_psk": ""}
        )
        with mock.patch.dict(os.environ, env):
            server = self._store(tls=empty)
            client = self._store(server.port, False, tls=empty)
            self.assertEqual(client.clone().add("plain", 1), 1)
            self.assertEqual(server.get("plain"), b"1")
            # Options are not merged with the env vars.
            with self.assertRaisesRegex(DistStoreError, "missing tls_key, tls_ca"):
                self._store(tls={"tls_cert": self.server[0]})

    @parametrize("option", ["cert", "key", "ca", "psk", "all"])
    @parametrize("from_env", [False, True])
    @parametrize("is_master", [False, True])
    def test_tls_empty_config(self, option, from_env, is_master):
        fields = ["cert", "key", "ca", "psk"] if option == "all" else [option]
        values = {
            (f"TORCH_TCPSTORE_TLS_{field.upper()}" if from_env else f"tls_{field}"): ""
            for field in fields
        }
        server = self._store()
        with mock.patch.dict(os.environ, values if from_env else {}):
            store = self._store(
                0 if is_master else server.port,
                is_master,
                tls={} if from_env else values,
            )
            self.assertEqual(store.add("plain", 1), 1)
            self.assertEqual(store.clone().get("plain"), b"1")

    @parametrize("option", ["cert", "key", "ca", "psk"])
    @parametrize("from_env", [False, True])
    @parametrize("is_master", [False, True])
    def test_tls_partial_empty_config(self, option, from_env, is_master):
        values = {
            "tls_cert": self.server[0],
            "tls_key": self.server[1],
            "tls_ca": self.ca,
        }
        values[f"tls_{option}"] = ""
        if from_env:
            values = {
                f"TORCH_TCPSTORE_{name.upper()}": value
                for name, value in values.items()
            }
        with mock.patch.dict(os.environ, values if from_env else {}):
            with self.assertRaisesRegex(
                DistStoreError,
                f"{option.upper() if from_env else option} must not be empty",
            ):
                self._store(is_master=is_master, tls={} if from_env else values)

    @parametrize("option", ["cert", "key", "ca", "psk"])
    def test_tls_nul_path(self, option):
        tls = {"tls_cert": self.server[0], "tls_key": self.server[1], "tls_ca": self.ca}
        if option == "psk":
            tls = {"tls_psk": self.psk}
        tls[f"tls_{option}"] += "\0suffix"
        with self.assertRaisesRegex(
            DistStoreError, f"tls_{option} must not contain NUL"
        ):
            self._store(tls=tls)

    def test_plaintext_clone_tls_env(self):
        server = self._store()
        self.assertEqual(server.clone().add("plain", 1), 1)
        with mock.patch.dict(os.environ, {"TORCH_TCPSTORE_TLS_CERT": ""}):
            self.assertEqual(server.clone().add("plain", 1), 2)
        with mock.patch.dict(os.environ, self._env(self.client)):
            self.assertEqual(server.clone().add("plain", 1), 3)

    def test_tls_missing_ca(self):
        env = self._env(self.server)
        env["TORCH_TCPSTORE_TLS_CA"] = ""
        with mock.patch.dict(os.environ, env):
            with self.assertRaisesRegex(DistStoreError, "TORCH_TCPSTORE_TLS_CA"):
                self._store()

    def test_tls_encrypted_key(self):
        with self.assertRaisesRegex(DistStoreError, "encrypted private key"):
            self._store(cert=(self.client[0], self.encrypted_key))

    def test_tls_ignores_use_libuv(self):
        with set_warn_always_context(True):
            with self.assertWarnsRegex(FutureWarning, "use_libuv=False"):
                server = self._store(cert=self.server, use_libuv=False)
            with self.assertWarnsRegex(FutureWarning, "use_libuv=False"):
                client = self._store(
                    server.port, False, cert=self.client, use_libuv=False
                )
        self.assertTrue(server.libuvBackend)
        client.set("key", "value")
        self.assertEqual(server.get("key"), b"value")
        self.assertEqual(client.clone().get("key"), b"value")

    def test_tls_multi_tenant_mismatch(self):
        server = self._store(cert=self.server, multi_tenant=True)
        with self.assertRaisesRegex(DistStoreError, "different TLS configuration"):
            self._store(server.port, multi_tenant=True)
        same = self._store(server.port, cert=self.server, multi_tenant=True)
        self.assertEqual(same.add("shared", 1), 1)

    def test_tls_wait_timeout_partial_record(self):
        # A server that sends only part of a TLS record must not block the
        # client past the wait timeout.
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(*self.server)
        ctx.load_verify_locations(self.ca)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.num_tickets = 0
        listener = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(listener.close)
        listener.settimeout(60)
        received = []
        errors = []
        partial_record_sent = threading.Event()

        def recv_exact(conn, n):
            data = b""
            while len(data) < n:
                chunk = conn.recv(n - len(data))
                if not chunk:
                    raise ConnectionError("closed")
                data += chunk
            return data

        def serve():
            try:
                conn, _ = listener.accept()
                conn.settimeout(60)
                with ctx.wrap_socket(conn, server_side=True) as tls:
                    recv_exact(tls, 5)  # VALIDATE
                    ping = recv_exact(tls, 5)
                    tls.sendall(ping[1:])
                    # Header of an application data record without its body.
                    os.write(tls.fileno(), b"\x17\x03\x03\x00\x20")
                    partial_record_sent.set()
                    recv_exact(tls, 1 + 8 + 8 + len("/k"))  # WAIT
                    received.append(recv_exact(tls, 1))
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        def join_server():
            thread.join(timeout=60)
            self.assertFalse(thread.is_alive(), "server thread hung")
            self.assertEqual(errors, [])

        self.addCleanup(join_server)
        client = dist.TCPStore(
            "127.0.0.1",
            listener.getsockname()[1],
            timeout=timedelta(seconds=60),
            wait_for_workers=False,
            tls_cert=self.client[0],
            tls_key=self.client[1],
            tls_ca=self.ca,
        )
        self.assertTrue(partial_record_sent.wait(60), "partial record not sent")
        start = time.monotonic()
        # The wait times out, then the server closes after CANCEL_WAIT.
        with self.assertRaisesRegex(DistNetworkError, "Failed to recv"):
            client.wait(["k"], timedelta(milliseconds=500))
        self.assertLess(time.monotonic() - start, 30)
        join_server()
        # CANCEL_WAIT
        self.assertEqual(received, [bytes([12])])

    @parametrize("version", [ssl.TLSVersion.TLSv1_2, ssl.TLSVersion.TLSv1_3])
    @parametrize("pending_wait", [False, True])
    def test_tls_close_notify_drains_commands(self, version, pending_wait):
        server = self._store(cert=self.server)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.load_verify_locations(self.ca)
        ctx.load_cert_chain(*self.client)
        ctx.minimum_version = ctx.maximum_version = version
        incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        tls = ctx.wrap_bio(incoming, outgoing, server_hostname="localhost")

        def blob(value):
            return struct.pack("=Q", len(value)) + value

        def command(op, *values):
            return bytes([op]) + b"".join(map(blob, values))

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as wire:
            wire.settimeout(30)
            if pending_wait:
                # Keep the large GET queued while the other client wakes WAIT.
                wire.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            wire.connect(("127.0.0.1", server.port))

            def flush():
                wire.sendall(outgoing.read())

            while True:
                try:
                    tls.do_handshake()
                    flush()
                    break
                except ssl.SSLWantReadError:
                    flush()
                    data = wire.recv(65536)
                    self.assertTrue(data, "TLS peer closed during handshake")
                    incoming.write(data)

            # Send VALIDATE, commands and close_notify in one TCP write.
            key, value = b"/last", b"value"
            if pending_wait:
                value = b"x" * (8 * 1024 * 1024)
                server.set("last", value)
                server.add("preloaded", 1)  # Wait for SET on this connection.
                # WAIT, GET, then a marker to order the other client's wake.
                request = struct.pack("=BQ", 6, 1) + blob(b"/trigger")
                request += command(3, key) + command(1, b"/parsed", b"yes")
            else:
                request = command(1, key, value) + command(3, key)
            tls.write(struct.pack("=BI", 0, 0x3C85F7CE) + request)
            with self.assertRaisesRegex(ssl.SSLWantReadError, "read"):
                tls.unwrap()
            flush()
            if pending_wait:
                self.assertEqual(server.get("parsed"), b"yes")
                server.set("trigger", "yes")
                server.add("barrier", 1)  # Ensures the wake was processed.
            while data := wire.recv(65536):
                incoming.write(data)
            incoming.write_eof()
            expected = blob(value)
            response = bytearray()
            while len(response) < len(expected):
                data = tls.read(len(expected) - len(response))
                self.assertTrue(data, "TLS peer discarded a completed command")
                response.extend(data)
            self.assertTrue(response == expected, "GET reply was truncated or corrupt")
            tls.unwrap()  # Verify the peer sent close_notify.

    def _psk_file(self, key=None, mode=0o600):
        fd, path = tempfile.mkstemp(dir=self._tmpdir.name)
        self.addCleanup(os.unlink, path)
        with os.fdopen(fd, "wb") as out:
            out.write((self.psk if key is None else key).encode())
        os.chmod(path, mode)
        return path

    @parametrize("from_env", [False, True])
    def test_tls_psk_file(self, from_env):
        path = self._psk_file()
        if from_env:
            os.environ["TORCH_TCPSTORE_TLS_PSK_FILE"] = path
        server = self._store(tls={} if from_env else {"tls_psk_file": path})
        client = self._store(server.port, False, tls={"tls_psk": self.psk})
        self.assertEqual(client.add("file", 1), 1)
        # The resolved key, not the path, survives file rotation and env changes.
        with open(path, "w") as out:
            out.write(self.other_psk)
        self.assertEqual(server.clone().add("file", 1), 2)
        os.environ["TORCH_TCPSTORE_TLS_PSK_FILE"] = path
        os.environ["TORCH_TCPSTORE_TLS_PSK"] = self.psk
        with self.assertRaisesRegex(DistStoreError, "cannot be combined"):
            self._store()

    @parametrize(
        "invalid",
        [
            "missing",
            "mode",
            "directory",
            "symlink",
            "fifo",
            "short",
            "long",
            "nul",
            "mixed",
        ],
    )
    def test_tls_psk_file_invalid(self, invalid):
        path = self._psk_file()
        if invalid == "missing":
            path += ".missing"
        elif invalid == "mode":
            os.chmod(path, 0o640)
        elif invalid == "directory":
            path = self._tmpdir.name
        elif invalid in ("symlink", "fifo"):
            target = path + "." + invalid
            if invalid == "symlink":
                os.symlink(path, target)
            else:
                os.mkfifo(target, 0o600)
            self.addCleanup(os.unlink, target)
            path = target
        elif invalid in ("short", "long", "nul"):
            value = {"short": "x" * 31, "long": "x" * 257, "nul": "x" * 31 + "\0"}[
                invalid
            ]
            with open(path, "w") as out:
                out.write(value)
        options = {"tls_psk_file": path}
        if invalid == "mixed":
            options["tls_psk"] = self.psk
        with self.assertRaisesRegex(DistStoreError, "PSK"):
            self._store(tls=options)

    def test_tls_require(self):
        plain = self._store()
        os.environ["TORCH_TCPSTORE_TLS_REQUIRE"] = "1"
        for options in ({}, {"tls_psk": ""}, {"tls_psk_file": ""}):
            with self.assertRaisesRegex(DistStoreError, "required but not configured"):
                self._store(tls=options)
        with self.assertRaisesRegex(DistStoreError, "required but not configured"):
            plain.clone()
        secure = self._store(tls={"tls_psk": self.psk})
        self.assertEqual(secure.add("secure", 1), 1)
        os.environ["TORCH_TCPSTORE_TLS_REQUIRE"] = "invalid"
        with self.assertRaisesRegex(DistStoreError, "must be 0 or 1"):
            self._store(tls={"tls_psk": self.psk})

    def test_tls_cipher_policy(self):
        ca = (self.ca, os.path.join(self._tmpdir.name, "ca.key"))
        cert = _make_cert(self._tmpdir.name, "rsa", ca, ["rsa:2048"])
        server = self._store(cert=cert)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.load_verify_locations(self.ca)
        ctx.load_cert_chain(*self.client)
        ctx.minimum_version = ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        for cipher, accepted in (
            ("AES128-SHA", False),
            ("ECDHE-RSA-AES128-SHA256", False),
            ("ECDHE-RSA-AES128-GCM-SHA256", True),
        ):
            ctx.set_ciphers(cipher)
            with socket.create_connection(
                ("localhost", server.port), timeout=10
            ) as sock:
                if not accepted:
                    with self.assertRaisesRegex(
                        ssl.SSLError, "handshake failure|no shared cipher"
                    ):
                        ctx.wrap_socket(sock, server_hostname="localhost")
                    continue
                with ctx.wrap_socket(sock, server_hostname="localhost") as tls:
                    self.assertEqual(tls.cipher()[0], cipher)

    def test_tls_server_handshake_timeout(self):
        server = self._store(cert=self.server)
        with socket.create_connection(("localhost", server.port), timeout=15) as idle:
            with socket.create_connection(
                ("localhost", server.port), timeout=15
            ) as partial:
                partial.sendall(b"\x16")
                self.assertEqual(idle.recv(1), b"")
                self.assertEqual(partial.recv(1), b"")
        self.assertEqual(server.add("alive", 1), 1)

    def test_tls_untrusted_client(self):
        server = self._store(cert=self.server)
        self._assert_fails_fast(
            "unknown ca",
            port=server.port,
            is_master=False,
            cert=self.untrusted,
            ca=self.ca,
        )
        self.assertEqual(server.add("alive", 1), 1)

    def test_tls_client_without_cert(self):
        server = self._store(cert=self.server)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(self.ca)
        ctx.check_hostname = False
        with socket.create_connection(("localhost", server.port), timeout=10) as sock:
            # TLS 1.3 reports the rejection on the first read.
            with self.assertRaisesRegex(
                ssl.SSLError, "certificate required|handshake failure"
            ):
                with ctx.wrap_socket(sock) as tls:
                    tls.recv(1)
        self.assertEqual(server.add("alive", 1), 1)

    def test_tls_untrusted_server(self):
        server = self._store(cert=self.server)
        self._assert_fails_fast(
            "certificate verify failed",
            port=server.port,
            is_master=False,
            cert=self.client,
            ca=self.other_ca,
        )
        self.assertEqual(server.add("alive", 1), 1)

    @parametrize("mode", ["cert", "psk"])
    def test_tls_server_plaintext_client(self, mode):
        server = self._store(tls=self._tls(mode, self.server))
        self._assert_fails_fast(
            "Server requires TLS", port=server.port, is_master=False
        )
        self.assertEqual(server.add("alive", 1), 1)

    @parametrize("mode", ["cert", "psk"])
    def test_plaintext_server_tls_client(self, mode):
        server = self._store()
        self._assert_fails_fast(
            "protocol version",
            port=server.port,
            is_master=False,
            tls=self._tls(mode, self.client),
        )
        self.assertEqual(server.add("alive", 1), 1)

    def test_tls_psk_ops(self):
        psk = {"tls_psk": self.psk}
        server = self._store(tls=psk, multi_tenant=True)
        client = self._store(server.port, False, tls=psk)
        client.set("key", "value")
        self.assertEqual(server.get("key"), b"value")
        self.assertEqual(client.add("counter", 2), 2)
        with self.assertRaisesRegex(DistStoreError, "wait timeout"):
            client.wait(["waited"], timedelta(milliseconds=10))
        self.assertEqual(client.clone().get("key"), b"value")
        same = self._store(server.port, tls=psk, multi_tenant=True)
        self.assertEqual(same.add("counter", 1), 3)
        with self.assertRaisesRegex(DistStoreError, "different TLS configuration"):
            self._store(server.port, tls={"tls_psk": self.other_psk}, multi_tenant=True)

    def test_tls_psk_wrong_key(self):
        server = self._store(tls={"tls_psk": self.psk})
        # OpenSSL 3.x sends decrypt error, 1.1.1 sends illegal parameter.
        self._assert_fails_fast(
            "alert (decrypt error|illegal parameter)",
            port=server.port,
            is_master=False,
            tls={"tls_psk": self.other_psk},
        )
        self.assertEqual(server.add("alive", 1), 1)

    @unittest.skipIf(
        not getattr(ssl, "HAS_PSK", False)
        or not hasattr(ssl.SSLContext, "set_psk_client_callback"),
        "needs Python 3.13 with PSK support",
    )
    def test_tls_psk_unknown_identity(self):
        server = self._store(tls={"tls_psk": self.psk})
        key = self.psk.encode("utf-8")

        def handshake(identity):
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            ctx.minimum_version = ssl.TLSVersion.TLSv1_3
            ctx.set_psk_client_callback(lambda hint: (identity, key))
            with socket.create_connection(("localhost", server.port), timeout=10) as s:
                with ctx.wrap_socket(s) as conn:
                    return conn.version()

        self.assertEqual(handshake("torch.distributed.TCPStore.psk.v1"), "TLSv1.3")
        with self.assertRaisesRegex(ssl.SSLError, "alert handshake failure"):
            handshake("torch.distributed.TCPStore.psk.v0")
        self.assertEqual(server.add("alive", 1), 1)

    def test_tls_psk_cert_mismatch(self):
        psk = {"tls_psk": self.psk}
        server = self._store(tls=psk)
        self._assert_fails_fast(
            "alert handshake failure.*same TLS mode",
            port=server.port,
            is_master=False,
            cert=self.client,
        )
        self.assertEqual(server.add("alive", 1), 1)
        server = self._store(cert=self.server)
        self._assert_fails_fast(
            "certificate verify failed.*same TLS mode",
            port=server.port,
            is_master=False,
            tls=psk,
        )
        self.assertEqual(server.add("alive", 1), 1)

    @parametrize("size", [1, 31, 257])
    @parametrize("from_env", [False, True])
    def test_tls_psk_bad_key(self, size, from_env):
        key = "~" * size
        env = {"TORCH_TCPSTORE_TLS_PSK": key} if from_env else {}
        tls = {} if from_env else {"tls_psk": key}
        with mock.patch.dict(os.environ, env):
            with self.assertRaisesRegex(
                DistStoreError, "PSK must contain 32 to 256 bytes"
            ) as caught:
                self._store(tls=tls)
        # Do not include the key in assertion diagnostics, even on failure.
        self.assertFalse(key in str(caught.exception), "PSK leaked in error")

    @parametrize("size", [32, 256])
    def test_tls_psk_key_sizes(self, size):
        psk = {"tls_psk": os.urandom(size // 2).hex()}
        server = self._store(tls=psk)
        client = self._store(server.port, False, tls=psk)
        self.assertEqual(client.add("counter", 1), 1)
        self.assertEqual(client.clone().add("counter", 1), 2)

    @parametrize(
        "invalid", ["port", "is_master", "unknown", "missing", "duplicate", "psk_type"]
    )
    def test_tls_psk_argument_errors_redacted(self, invalid):
        args = ["localhost", 0]
        kwargs = {"tls_psk": self.psk}
        if invalid == "port":
            args[1] = "bad port"
        elif invalid == "is_master":
            kwargs["is_master"] = "bad bool"
        elif invalid == "unknown":
            kwargs["unknown"] = True
        elif invalid == "missing":
            args = []
        elif invalid == "duplicate":
            kwargs["port"] = 0
        else:
            kwargs["tls_psk"] = [self.psk]
        try:
            dist.TCPStore(*args, **kwargs)
        except TypeError as error:
            self.assertFalse(
                self.psk in "".join(traceback.format_exception(error)),
                "PSK leaked in traceback",
            )
            self.assertTrue(error.__cause__ is None, "Unexpected exception cause")
            self.assertTrue(error.__context__ is None, "Unexpected exception context")
        else:
            self.fail("Invalid arguments accepted")

    def test_tls_psk_constructor_wrapper(self):
        class IndexValue:
            def __index__(self):
                return 0

        class DerivedStore(dist.TCPStore):
            pass

        # Keep conversion retries and Python subclass initialization working.
        server = DerivedStore(
            "localhost",
            IndexValue(),
            is_master=True,
            wait_for_workers=False,
            tls_psk=self.psk,
        )
        self.assertEqual(server.clone().add("counter", 1), 1)
        self.assertIn("tls_psk", dist.TCPStore.__init__.__doc__)

    def test_tls_psk_diagnostics_redacted(self):
        # Capture native stdout/stderr too; never include either in failures.
        code = """
import os
import torch.distributed as dist
key = os.environ.pop("TCPSTORE_TEST_SECRET")
def store(**kwargs):
    return dist.TCPStore("localhost", 0, is_master=True, wait_for_workers=False, **kwargs)
for kwargs in ({"tls_psk": key[:31]}, {"tls_psk": key + "\\0suffix"},
               {"tls_psk": key, "tls_ca": "unused"}):
    try:
        store(**kwargs)
    except dist.DistStoreError as error:
        print(error)
    else:
        raise AssertionError("Invalid TLS configuration accepted")
server = store(tls_psk=key)
print(server)
try:
    dist.TCPStore("localhost", server.port, tls_psk=key[::-1])
except dist.DistStoreError as error:
    print(error)
else:
    raise AssertionError("Wrong PSK accepted")
os.environ["TORCH_TCPSTORE_TLS_PSK"] = key[:31]
try:
    store()
except dist.DistStoreError as error:
    print(error)
else:
    raise AssertionError("Invalid environment PSK accepted")
"""
        env = {**os.environ, "TCPSTORE_TEST_SECRET": self.psk}
        result = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, timeout=60
        )
        output = result.stdout + result.stderr
        for secret in (self.psk, self.psk[:31], self.psk[::-1]):
            self.assertFalse(
                secret.encode() in output, "PSK leaked in native diagnostics"
            )
        self.assertEqual(result.returncode, 0, "PSK diagnostic subprocess failed")

    def test_tls_psk_config(self):
        with self.assertRaisesRegex(DistStoreError, "tls_psk cannot be combined"):
            self._store(tls={"tls_psk": self.psk, "tls_ca": self.ca})
        env = self._env(self.server)
        env["TORCH_TCPSTORE_TLS_PSK"] = self.psk
        with mock.patch.dict(os.environ, env):
            with self.assertRaisesRegex(
                DistStoreError, "TORCH_TCPSTORE_TLS_PSK cannot be combined"
            ):
                self._store()
            # Options are not merged with the env vars.
            server = self._store(tls={"tls_psk": self.psk})
            client = self._store(server.port, False, tls={"tls_psk": self.psk})
            self.assertEqual(client.add("counter", 1), 1)


instantiate_parametrized_tests(TCPStoreTLSTest)


class PrefixTCPStoreTest(TestCase, StoreTestBase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self.tcpstore = create_tcp_store()
        self.prefix = "test_prefix"
        self.tcpstore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return dist.PrefixStore(self.prefix, self.tcpstore)

    # The PrefixTCPStore has 6 keys in test_set_get. It contains the 5 keys
    # added by the user and one additional key used for coordinate all the
    # workers.
    @property
    def num_keys_total(self):
        return 6

    def test_underlying_non_prefix_store(self):
        store = self._create_store()
        wrapped_store = dist.PrefixStore(
            self.prefix, dist.PrefixStore(self.prefix, store)
        )
        self.assertEqual(self.tcpstore, store._underlying_non_prefix_store)
        self.assertEqual(self.tcpstore, wrapped_store._underlying_non_prefix_store)


class MyPythonStore(dist.Store):
    def __init__(self) -> None:
        super().__init__()
        self.store = {}

    def set(self, key, value):
        if not isinstance(key, (str, bytes)):
            raise AssertionError("Expected set to be called with string key")
        if type(value) is not bytes:
            raise AssertionError("Expected set to be called with bytes value")
        self.store[key] = value

    def get(self, key):
        value = self.store.get(key, b"")
        if type(value) is not bytes:
            raise AssertionError("Expected get to return bytes value")
        return value

    def add(self, key, value):
        new = int(self.store.get(key, 0)) + value
        self.set(key, bytes(str(new).encode("utf-8")))
        return new

    def compare_set(self, key, expected, newValue):
        if type(expected) is not bytes:
            raise AssertionError("compare_set::expected not bytes")
        if type(newValue) is not bytes:
            raise AssertionError("compare_set::newValue not bytes")

        val = self.store.get(key, None)
        if expected == val or val is None:
            val = self.store[key] = newValue
        return val

    def clone(self) -> "MyPythonStore":
        return self


class PythonStoreTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_set_get(self):
        # If we were to inherit from StoreTestBase and try to use
        # its test_set_get function, we would exercise the Python
        # API directly, instead of going through the C++ trampoline.
        # We care about testing the C++ trampoline, so run the
        # equivalent of StoreTestBase.test_set_get from C++.
        # See `torch/csrc/distributed/c10d/init.cpp` for the definition
        # of this test function.
        dist._test_python_store(MyPythonStore())


class RendezvousTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_unknown_handler(self):
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            dist.rendezvous("invalid://")

    def test_url_with_node_params(self):
        with self.assertRaisesRegex(AssertionError, "has node-specific arguments"):
            dist.rendezvous("file://foo?rank=12&world_size=16", 12, 16)


class RendezvousEnvTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @retry_on_connect_failures
    def test_nominal(self):
        # retry_on_connect_failures re-runs this whole body on a RuntimeError,
        # so every mutation below has to be undone before the next attempt.
        try:
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(common.find_free_port())

            # Single rank
            os.environ["RANK"] = "0"
            gen0 = dist.rendezvous("env://")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(1, size0)

            store0.set("key0", "value0")

            # check with get
            self.assertEqual(b"value0", store0.get("key0"))
        finally:
            for var in ("WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "RANK"):
                os.environ.pop(var, None)


class RendezvousFileTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, "path missing"):
            gen = dist.rendezvous("file://?rank=0&world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = dist.rendezvous("file:///tmp/foo?world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = dist.rendezvous("file:///tmp/foo?rank=0")
            next(gen)

    def test_nominal(self):
        with tempfile.NamedTemporaryFile(delete=False) as file:
            url = f"file:///{file.name.replace(os.path.sep, '/')}?world_size=2"
            gen0 = dist.rendezvous(url + "&rank=0")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(2, size0)
            gen1 = dist.rendezvous(url + "&rank=1")
            store1, rank1, size1 = next(gen1)
            self.assertEqual(1, rank1)
            self.assertEqual(2, size1)

            # Set value on both stores
            store0.set("key0", "value0")
            store1.set("key1", "value1")

            # Cross check with get
            self.assertEqual(b"value0", store1.get("key0"))
            self.assertEqual(b"value1", store0.get("key1"))


@skip_if_win32()
class RendezvousTCPTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def create_tcp_url(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()
        url = f"tcp://{addr}:{port:d}?world_size=1"
        return url

    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, "port number missing"):
            gen = dist.rendezvous("tcp://127.0.0.1?rank=0&world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = dist.rendezvous("tcp://127.0.0.1:23456?world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = dist.rendezvous("tcp://127.0.0.1:23456?rank=0")
            next(gen)

    def test_dns_timeout(self):
        with self.assertRaisesRegex(
            DistNetworkError, "client socket has timed out after.*dnsnotexist"
        ) as manager:
            gen = dist.rendezvous(
                "tcp://dnsnotexist:23456?world_size=2&rank=0",
                timeout=timedelta(seconds=1),
            )
            next(gen)
        self.assertTrue(isinstance(manager.exception, DistError))

    @retry_on_connect_failures
    def test_nominal(self):
        url = self.create_tcp_url()
        gen0 = dist.rendezvous(url + "&rank=0")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        # Set value on the single store
        store0.set("key0", "value0")

        # check with get
        self.assertEqual(b"value0", store0.get("key0"))

    @retry_on_connect_failures(connect_errors=(CONNECT_TIMEOUT, ADDRESS_IN_USE))
    def test_tcp_store_timeout_set(self):
        url = self.create_tcp_url()
        test_store_timeout = timedelta(seconds=0.1)
        gen0 = dist.rendezvous(url + "&rank=0", timeout=timedelta(seconds=10))
        store0, _, _ = next(gen0)
        store0.set_timeout(test_store_timeout)
        # this should time out in 0.1s. If the timeout passed into rendezvous was
        # not respected, it will take much longer to timeout.
        start = time.time()
        with self.assertRaisesRegex(
            DistStoreError, "wait timeout after 100ms, keys: /nonexistent key"
        ):
            store0.get("nonexistent key")

        end = time.time()
        time_diff = end - start
        self.assertGreater(10, time_diff)

    def test_tcp_store_timeout_doest_break_client(self):
        url = self.create_tcp_url()
        test_store_timeout = timedelta(seconds=0.1)
        gen0 = dist.rendezvous(url + "&rank=0", timeout=timedelta(seconds=10))
        store0, _, _ = next(gen0)
        store0.set_timeout(test_store_timeout)
        # this should time out in 10s. If the timeout passed into rendezvous was
        # not respected, it will take much longer to timeout.
        start = time.time()
        with self.assertRaisesRegex(
            DistStoreError, "wait timeout after 100ms, keys: /the_key"
        ):
            store0.get("the_key")

        store0.set("the_key", "x")

        self.assertEqual(b"x", store0.get("the_key"))

        end = time.time()
        time_diff = end - start
        self.assertGreater(10, time_diff)

    def test_tcp_store_url_with_libuv(self):
        url = self.create_tcp_url()
        gen0 = dist.rendezvous(url + "&rank=0&use_libuv=1")
        store0, _, _ = next(gen0)
        self.assertTrue(store0.libuvBackend)

    def test_agent_store_ignored_for_other_address(self):
        # torchrun's elastic agent exports TORCHELASTIC_USE_AGENT_STORE into
        # every worker, meaning "the agent serves a store at
        # MASTER_ADDR:MASTER_PORT". _create_c10d_store used to honor it for any
        # address it was handed, so a private store on a caller-chosen port
        # became client-only on every rank, nobody ever started a server and
        # every rank blocked in connect-retry until the store timeout.
        url = self.create_tcp_url()
        env = {
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "MASTER_ADDR": DEFAULT_HOSTNAME,
            "MASTER_PORT": str(common.find_free_port()),
        }
        with mock.patch.dict(os.environ, env):
            gen0 = dist.rendezvous(url + "&rank=0", timeout=timedelta(seconds=10))
            store0, _, _ = next(gen0)
        store0.set("key0", "value0")
        self.assertEqual(b"value0", store0.get("key0"))

    def test_agent_store_honored_for_master_address(self):
        # The flip side: at MASTER_ADDR:MASTER_PORT the agent is the server, so
        # every rank must stay a client. Point at a port nothing listens on and
        # check we fail to connect rather than silently starting a server.
        port = common.find_free_port()
        url = f"tcp://{DEFAULT_HOSTNAME}:{port:d}?world_size=1&rank=0"
        env = {
            "TORCHELASTIC_USE_AGENT_STORE": "True",
            "MASTER_ADDR": DEFAULT_HOSTNAME,
            "MASTER_PORT": str(port),
        }
        with mock.patch.dict(os.environ, env):
            with self.assertRaises(DistNetworkError):
                next(dist.rendezvous(url, timeout=timedelta(seconds=1)))


class DummyStore(dist.Store):
    def __init__(self) -> None:
        self.appends = []
        self.multi_sets = []
        self.multi_gets = []
        self.multi_get_res = []
        super().__init__()

    def append(self, key, value):
        self.appends.append((key, value))

    def multi_get(self, keys):
        self.multi_gets.append(keys)
        return self.multi_get_res.pop(0)

    def multi_set(self, keys, values):
        self.multi_sets.append((keys, values))

    def has_extended_api(self):
        return True


class TestPythonStore(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_optional_methods_fail(self):
        class TestStore(dist.Store):
            pass

        store = TestStore()
        self.assertFalse(store.has_extended_api())
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.append("foo", "bar")
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_get(["foo", "bar"])
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_set(["foo", "bar"], [b"v", b"v"])

    def test_has_extended_api_passthrough(self):
        class TestStore(dist.Store):
            pass

        test_store = TestStore()
        store = dist.PrefixStore("p", test_store)
        self.assertFalse(store.has_extended_api())
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.append("foo", "bar")
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_get(["foo", "bar"])
        with self.assertRaisesRegex(RuntimeError, "Not implemented."):
            store.multi_set(["foo", "bar"], [b"v", b"v"])

    def test_has_extended_api_roundtrip(self):
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        self.assertTrue(prefix.has_extended_api())

    def test_append_roundtrip(self):
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        prefix.append("foo", "bar")
        self.assertEqual(1, len(store.appends))
        self.assertEqual(("p/foo", b"bar"), store.appends[0])

    def test_multi_get_roundtrip(self):
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        store.multi_get_res.append([b"x", b"y"])
        res = prefix.multi_get(["foo", "bar"])
        self.assertEqual(1, len(store.multi_gets))
        self.assertEqual(["p/foo", "p/bar"], store.multi_gets[0])
        self.assertEqual([b"x", b"y"], res)

    def test_multi_set_roundtrip(self):
        store = DummyStore()
        prefix = dist.PrefixStore("p", store)
        prefix.multi_set(["foo", "bar"], [b"x", b"y"])
        self.assertEqual(1, len(store.multi_sets))
        self.assertEqual(["p/foo", "p/bar"], store.multi_sets[0][0])
        self.assertEqual([b"x", b"y"], store.multi_sets[0][1])

    def test_extended_methods_fallbacks(self):
        test_store = MyPythonStore()
        store = dist.PrefixStore("p", test_store)
        self.assertFalse(store.has_extended_api())
        store.append("foo", b"po")
        store.append("foo", b"tato")
        self.assertEqual(store.get("foo"), b"potato")

        store.multi_set(["a", "b"], [b"c", b"d"])
        self.assertEqual(store.multi_get(["a", "b", "foo"]), [b"c", b"d", b"potato"])


class TestMultiThreadedWait(MultiThreadedTestCase):
    hw_classification = HardwareClassification.GENERIC

    file_store = dist.FileStore(tempfile.NamedTemporaryFile(delete=False).name, 1)  # noqa: SIM115
    hash_store = dist.HashStore()

    tcp_store = create_tcp_store()

    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    def _test_wait(self, store):
        store.set_timeout(timedelta(seconds=2))
        if dist.get_rank() == 0:
            store.wait(["key1"])
            self.assertEqual(b"value1", store.get("key1"))
        if dist.get_rank() == 1:
            store.set("key1", "value1")

    def test_wait_hash_store(self):
        self._test_wait(self.hash_store)

    def test_wait_file_store(self):
        self._test_wait(self.file_store)

    def test_wait_prefix_file_store(self):
        store = dist.PrefixStore("pre", self.file_store)
        self._test_wait(store)

    def _test_wait_tcp_store(self, master_store):
        store = (
            master_store
            if dist.get_rank() == 0
            else dist.TCPStore(
                host_name=master_store.host,
                port=master_store.port,
                is_master=False,
                wait_for_workers=False,
            )
        )
        self._test_wait(store)

        prefix_store = dist.PrefixStore("pre", store)
        self._test_wait(prefix_store)

    def test_wait_tcp_store(self):
        self._test_wait_tcp_store(self.tcp_store)


instantiate_parametrized_tests(TestMultiThreadedWait)


@skip_if_win32()
class TimeoutTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def tearDown(self):
        import signal

        super().tearDown()
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)

    def test_interrupt_doesnt_break_wait(self):
        import signal

        rank_res = [None, None]

        def run(rank, my_store):
            nonlocal rank_res
            try:
                if rank == 0:
                    time.sleep(4)
                    my_store.set("foo", "bar")
                else:
                    my_store.wait(["foo"], datetime.timedelta(seconds=10))
                rank_res[rank] = True
            except BaseException as e:  # noqa: E261
                rank_res[rank] = e
            time.sleep(1)

        rank0_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME,
            port=0,
            world_size=2,
            is_master=True,
            wait_for_workers=False,
        )
        rank1_store = dist.TCPStore(
            host_name=DEFAULT_HOSTNAME,
            port=rank0_store.port,
            world_size=2,
            is_master=False,
            wait_for_workers=False,
        )

        threads = []
        for i in range(2):
            t = threading.Thread(
                target=run,
                args=(
                    i,
                    [rank0_store, rank1_store][i],
                ),
            )
            t.start()
            threads.append(t)

        def handler(a, b):
            pass

        signal.signal(signal.SIGUSR1, handler)
        time.sleep(1)
        signal.pthread_kill(threads[1].ident, signal.SIGUSR1)

        for t in threads:
            t.join()
        self.assertTrue(rank_res[0], "rank0")
        self.assertTrue(rank_res[1], "rank1")


class UseLibUvDeprecationTest(TestCase):
    """
    ``use_libuv=False`` is a no-op that warns: libuv is the only TCPStore backend.
    """

    hw_classification = HardwareClassification.GENERIC

    def tearDown(self):
        super().tearDown()
        if dist.is_initialized():
            dist.destroy_process_group()
        os.environ.pop("USE_LIBUV", None)
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)

    def test_tcp_store_arg(self):
        with set_warn_always_context(True):
            with self.assertWarnsRegex(FutureWarning, "use_libuv=False"):
                store = create_tcp_store(use_libuv=False)
        self.assertTrue(store.libuvBackend)
        store.set("key", "value")
        self.assertEqual(b"value", store.get("key"))

    def test_with_url_param(self):
        port = common.find_free_port()
        with set_warn_always_context(True):
            with self.assertWarnsRegex(FutureWarning, "use_libuv=False"):
                dist.init_process_group(
                    "gloo",
                    rank=0,
                    world_size=1,
                    init_method=f"tcp://{DEFAULT_HOSTNAME}:{port}?use_libuv=0",
                )
        self._run_test()

    def test_with_env_var(self):
        port = common.find_free_port()
        os.environ["USE_LIBUV"] = "0"
        os.environ["MASTER_ADDR"] = DEFAULT_HOSTNAME
        os.environ["MASTER_PORT"] = str(port)
        with set_warn_always_context(True):
            with self.assertWarnsRegex(FutureWarning, "use_libuv=False"):
                dist.init_process_group(
                    "gloo", rank=0, world_size=1, init_method="env://"
                )
        self._run_test()

    def _run_test(self):
        pg = dist.group.WORLD
        store = c10d._get_process_group_store(pg)
        self.assertTrue(isinstance(store, dist.PrefixStore))
        # c10d does multiple levels of wrapping
        while isinstance(store, dist.PrefixStore):
            store = store.underlying_store
        self.assertTrue(isinstance(store, dist.TCPStore))
        self.assertTrue(store.libuvBackend)
        dist.destroy_process_group()


class TestClientProtocol(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_client_connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

        def listen() -> None:
            sock.listen()
            conn, _ = sock.accept()

            # VALIDATE
            # 0x3C85F7CE
            self.assertEqual(conn.recv(5), b"\x00\xce\xf7\x85\x3c")

            # PING
            data = conn.recv(5)
            self.assertEqual(data[0], 13)
            nonce = struct.unpack("i", data[1:])[0]
            self.assertEqual(nonce, os.getpid())

            # send PING nonce response
            conn.sendall(data[1:])

            conn.close()

        thread = threading.Thread(target=listen)
        thread.start()

        dist.TCPStore(
            host_name="localhost",
            port=port,
            world_size=2,
            is_master=False,
            timeout=timedelta(seconds=2),
            wait_for_workers=False,
        )

        thread.join()


class PrefixStoreBarrierTimeoutTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_barrier_respects_store_timeout(self):
        # Store::barrier(key, world_size) must use the virtual getTimeout(),
        # so PrefixStore forwards the wrapped store's timeout, not the 300s default.
        import time

        store = dist.PrefixStore("prefix", dist.HashStore())
        store.set_timeout(timedelta(milliseconds=1))
        start = time.monotonic()
        with self.assertRaises(DistStoreError):
            store.barrier("barrier_key", 2)
        self.assertLess(time.monotonic() - start, 10)

    def test_barrier_explicit_timeout_overrides_store_timeout(self):
        import time

        store = dist.PrefixStore("prefix", dist.HashStore())
        store.set_timeout(timedelta(seconds=300))
        start = time.monotonic()
        with self.assertRaises(DistStoreError):
            store.barrier("barrier_key2", 2, timedelta(milliseconds=1))
        self.assertLess(time.monotonic() - start, 10)


if __name__ == "__main__":
    if device_type != "cpu":
        if getattr(torch.get_device_module(device_type), "_initialized", False):
            raise AssertionError(
                f"test_distributed must not have initialized {device_type} context on main process"
            )
    run_tests()
