# Owner(s): ["oncall: distributed"]

import datetime
import os
import socket
import struct
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from sys import platform

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
    ADDRESS_IN_USE,
    CONNECT_TIMEOUT,
    load_tests,
    retry_on_connect_failures,
    run_tests,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"

DEFAULT_HOSTNAME = "localhost"

torch.backends.cuda.matmul.allow_tf32 = False

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    visible_devices = list(range(torch.accelerator.device_count()))
    gpus_per_process = torch.accelerator.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class StoreTestBase:
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
            fs.wait(["bad_key"], timedelta(seconds=0.25))
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

    # This is the number of keys used in test_set_get. Adding this as a class
    # property instead of hardcoding in the test since some Store
    # implementations will have differing number of keys. In the base case,
    # there will be 5 keys: key, key0, key1, key2, key3.
    @property
    def num_keys_total(self):
        return 5


class FileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super().setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def _create_store(self):
        store = dist.FileStore(self.file.name, 1)
        store.set_timeout(timedelta(seconds=300))
        return store

    def test_init_pg_and_rpc_with_same_file(self):
        file = tempfile.NamedTemporaryFile(delete=False)
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
        assert os.path.exists(file.name)

        rpc.shutdown()
        os.remove(file.name)

    def test_refcount(self):
        file = tempfile.NamedTemporaryFile(delete=False)
        store = dist.FileStore(file.name, 1)
        store2 = dist.FileStore(file.name, 1)

        del store
        assert os.path.exists(file.name)
        del store2
        assert not os.path.exists(file.name)

    @property
    def num_keys_total(self):
        return 6


@skip_if_win32()
class HashStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        store = dist.HashStore()
        store.set_timeout(timedelta(seconds=300))
        return store


class PrefixStoreTest(TestCase):
    def setUp(self):
        # delete is false as FileStore will automatically clean up the file
        self.file = tempfile.NamedTemporaryFile(delete=False)

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
    def setUp(self):
        super().setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.filestore = dist.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"
        self.filestore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return dist.PrefixStore(self.prefix, self.filestore)

    @property
    def num_keys_total(self):
        return 6


class TCPStoreTest(TestCase, StoreTestBase):
    _use_libuv = False

    def _create_store(self):
        store = create_tcp_store(use_libuv=self._use_libuv)
        store.set_timeout(timedelta(seconds=300))
        return store

    def _create_store_with_ws(self, addr, world_size):
        return create_tcp_store(
            addr, world_size, wait_for_workers=False, use_libuv=self._use_libuv
        )

    def test_address_already_in_use(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        err_msg_reg = f"^The server socket has failed to listen on any local .*{port}"
        with self.assertRaisesRegex(dist.DistNetworkError, err_msg_reg):
            # Use noqa to silence flake8.
            # Need to store in an unused variable here to ensure the first
            # object is not destroyed before the second object is created.
            store1 = dist.TCPStore(addr, port, 1, True, use_libuv=self._use_libuv)  # noqa: F841
            store2 = dist.TCPStore(addr, port, 1, True, use_libuv=self._use_libuv)  # noqa: F841
            self.assertEqual(store1.libuvBackend, self._use_libuv)
            self.assertEqual(store2.libuvBackend, self._use_libuv)

    @retry_on_connect_failures
    def test_multitenancy(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        # Use noqa to silence flake8.
        # Need to store in an unused variable here to ensure the first
        # object is not destroyed before the second object is created.
        store1 = dist.TCPStore(
            addr, port, 1, True, multi_tenant=True, use_libuv=self._use_libuv
        )  # type: ignore[call-arg] # noqa: F841
        store2 = dist.TCPStore(
            addr, port, 1, True, multi_tenant=True, use_libuv=self._use_libuv
        )  # type: ignore[call-arg] # noqa: F841
        self.assertEqual(store1.libuvBackend, self._use_libuv)
        self.assertEqual(store2.libuvBackend, self._use_libuv)

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

        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        # We internally use a multi-tenant TCP store. Both PG and RPC should successfully
        # initialize even when using the same socket address.

        os.environ["USE_LIBUV"] = "1" if self._use_libuv else "0"
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

        del os.environ["USE_LIBUV"]
        assert "USE_LIBUV" not in os.environ
        rpc.shutdown()
        dist.destroy_process_group()

    @skip_if_win32()
    def test_take_over_listen_socket(self):
        listen_sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_sock.bind(("localhost", 0))
        addr, port, *_ = listen_sock.getsockname()
        listen_fd = listen_sock.detach()

        store = dist.TCPStore(
            addr,
            port,
            1,
            is_master=True,
            master_listen_fd=listen_fd,
            use_libuv=self._use_libuv,
        )

        self.assertEqual(store.libuvBackend, self._use_libuv)
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
        fs.set_timeout(timedelta(seconds=2))
        with self.assertRaises(RuntimeError):
            fs.get("key")
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
            addr,
            port,
            world_size=world_size,
            timeout=timedelta(seconds=10),
            use_libuv=self._use_libuv,
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
        self.assertEqual(server_store.libuvBackend, self._use_libuv)
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
        self.assertEqual(store.libuvBackend, self._use_libuv)
        store.set("foo", "po")
        store.append("foo", "tato")
        store.append("bar", "po")
        store.append("bar", "tato")
        self.assertEqual(b"potato", store.get("foo"))
        self.assertEqual(b"potato", store.get("bar"))

    def test_multi_set(self):
        store = self._create_store()
        self.assertEqual(store.libuvBackend, self._use_libuv)
        store.multi_set(["foo", "bar"], ["po", "tato"])
        self.assertEqual(b"po", store.get("foo"))
        self.assertEqual(b"tato", store.get("bar"))

    def test_multi_get(self):
        store = self._create_store()
        self.assertEqual(store.libuvBackend, self._use_libuv)
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
            dist.TCPStore(
                "localhost",
                0,
                2,
                True,
                timeout=timedelta(seconds=2),
                use_libuv=self._use_libuv,
            )

        # when wait_for_workers is not set, then there should be no exception raised
        dist.TCPStore(
            "localhost",
            0,
            2,
            True,
            timeout=timedelta(seconds=2),
            wait_for_workers=False,
            use_libuv=self._use_libuv,
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
                host_name="localhost",
                port=store.port,
                world_size=1,
                is_master=True,
                use_libuv=self._use_libuv,
            )

        USE_AGENT_STORE = "TORCHELASTIC_USE_AGENT_STORE"
        MASTER_PORT = "MASTER_PORT"

        os.environ[USE_AGENT_STORE] = "1"
        os.environ[MASTER_PORT] = str(store.port)
        second_server = dist.TCPStore(
            host_name="localhost",
            port=store.port,
            world_size=1,
            is_master=True,
            use_libuv=self._use_libuv,
        )
        del os.environ[USE_AGENT_STORE]
        del os.environ[MASTER_PORT]

        self.assertEqual(second_server.port, store.port)


class LibUvTCPStoreTest(TCPStoreTest):
    _use_libuv = True

    def _create_store(self):
        store = create_tcp_store(use_libuv=True)
        store.set_timeout(timedelta(seconds=300))
        return store

    def _create_store_with_ws(self, addr, world_size):
        return create_tcp_store(
            addr, world_size, wait_for_workers=False, use_libuv=True
        )


class PrefixTCPStoreTest(TestCase, StoreTestBase):
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
    def test_unknown_handler(self):
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            dist.rendezvous("invalid://")

    def test_url_with_node_params(self):
        with self.assertRaisesRegex(AssertionError, "has node-specific arguments"):
            dist.rendezvous("file://foo?rank=12&world_size=16", 12, 16)


class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    def test_nominal(self):
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


class RendezvousFileTest(TestCase):
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
    file_store = dist.FileStore(tempfile.NamedTemporaryFile(delete=False).name, 1)
    hash_store = dist.HashStore()

    tcp_store = create_tcp_store(use_libuv=False)
    tcp_store_uv = create_tcp_store(use_libuv=True)

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
                use_libuv=False,
            )
        )
        self._test_wait(store)

        prefix_store = dist.PrefixStore("pre", store)
        self._test_wait(prefix_store)

    def test_wait_tcp_store(self):
        self._test_wait_tcp_store(self.tcp_store)

    def test_wait_tcp_store_uv(self):
        self._test_wait_tcp_store(self.tcp_store_uv)


instantiate_parametrized_tests(TestMultiThreadedWait)


@skip_if_win32()
class TimeoutTest(TestCase):
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
            except Error as e:  # noqa: F821
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


class InitPgWithNonUvStore(TestCase):
    """
    This test shows how to use the legacy TCPStore (non-libuv) backend since libuv is now
    the default backend.
    """

    def tearDown(self):
        super().tearDown()
        os.environ.pop("USE_LIBUV", None)
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)

    def test_with_url_param(self):
        port = common.find_free_port()
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
        dist.init_process_group("gloo", rank=0, world_size=1, init_method="env://")
        self._run_test()

    def _run_test(self):
        pg = dist.group.WORLD
        store = c10d._get_process_group_store(pg)
        self.assertTrue(isinstance(store, dist.PrefixStore))
        # c10d does multiple levels of wrapping
        while isinstance(store, dist.PrefixStore):
            store = store.underlying_store
        self.assertTrue(isinstance(store, dist.TCPStore))
        self.assertFalse(store.libuvBackend)
        dist.destroy_process_group()


class TestClientProtocol(TestCase):
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


if __name__ == "__main__":
    if device_type != "cpu":
        assert not torch.get_device_module()._initialized, (
            f"test_distributed must not have initialized {device_type} context on main process"
        )
    run_tests()
