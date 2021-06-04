import os
import random
import sys
import tempfile
import time
import traceback
import unittest
from datetime import timedelta
from sys import platform

import torch
import torch.distributed as c10d
import torch.distributed.rpc as rpc

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.testing._internal.common_utils as common
from torch._six import string_classes
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_win32,
    create_tcp_store
)
from torch.testing._internal.common_utils import (
    TestCase,
    load_tests,
    run_tests,
    retry_on_connect_failures,
    ADDRESS_IN_USE,
    CONNECT_TIMEOUT,
    IS_WINDOWS,
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


def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    visible_devices = list(range(torch.cuda.device_count()))
    gpus_per_process = torch.cuda.device_count() // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process: (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class StoreTestBase(object):
    def _create_store(self, i):
        raise RuntimeError("not implemented")

    def _test_set_get(self, fs):
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

    def test_set_get(self):
        self._test_set_get(self._create_store())

    def _test_compare_set(self, store):
        missing_key_result = store.compare_set("cs_key0", "wrong_old_value", "new_value0")
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

    # This is the number of keys used in test_set_get. Adding this as a class
    # property instead of hardcoding in the test since some Store
    # implementations will have differing number of keys. In the base case,
    # there will be 5 keys: key, key0, key1, key2, key3.
    @property
    def num_keys_total(self):
        return 5


class FileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(FileStoreTest, self).setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def _create_store(self):
        store = c10d.FileStore(self.file.name, 1)
        store.set_timeout(timedelta(seconds=300))
        return store


@skip_if_win32()
class HashStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(HashStoreTest, self).setUp()

    def _create_store(self):
        store = c10d.HashStore()
        store.set_timeout(timedelta(seconds=300))
        return store


class PrefixFileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(PrefixFileStoreTest, self).setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.filestore = c10d.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"
        self.filestore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return c10d.PrefixStore(self.prefix, self.filestore)


class TCPStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        store = create_tcp_store()
        store.set_timeout(timedelta(seconds=300))
        return store

    def test_address_already_in_use(self):
        if sys.platform == "win32":
            err_msg_reg = "Only one usage of each socket address*"
        else:
            err_msg_reg = "^Address already in use$"
        with self.assertRaisesRegex(RuntimeError, err_msg_reg):
            addr = DEFAULT_HOSTNAME
            port = common.find_free_port()

            # Use noqa to silence flake8.
            # Need to store in an unused variable here to ensure the first
            # object is not destroyed before the second object is created.
            store1 = c10d.TCPStore(addr, port, 1, True)  # noqa: F841
            store2 = c10d.TCPStore(addr, port, 1, True)  # noqa: F841

    def test_multitenancy(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

        # Use noqa to silence flake8.
        # Need to store in an unused variable here to ensure the first
        # object is not destroyed before the second object is created.
        store1 = c10d.TCPStore(addr, port, 1, True, multi_tenant=True)  # type: ignore[call-arg] # noqa: F841
        store2 = c10d.TCPStore(addr, port, 1, True, multi_tenant=True)  # type: ignore[call-arg] # noqa: F841

    @skip_if_win32()
    def test_init_pg_and_rpc_with_same_socket(self):
        addr = DEFAULT_HOSTNAME
        port = common.find_free_port()

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

        backend_opts = rpc.ProcessGroupRpcBackendOptions(
            init_method=f"tcp://{addr}:{port}"
        )
        rpc.init_rpc(
            name="worker0",
            rank=0,
            world_size=1,
            rpc_backend_options=backend_opts,
        )

        rpc.shutdown()

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

    def _create_client(self, index, addr, port, world_size, messages):
        try:
            client_store = dist.TCPStore(addr, port, world_size, timeout=timedelta(seconds=10))
            self.assertEqual("value".encode(), client_store.get("key"))
            client_store.set(f"new_key{index}", f"new_value{index}")
            self.assertEqual(f"next_value{index}".encode(),
                             client_store.compare_set(f"new_key{index}", f"new_value{index}", f"next_value{index}"))
        except Exception:
            messages.put('Caught exception: \n{}exiting process with exit code: {}'
                         .format(traceback.format_exc(), MultiProcessTestCase.TEST_ERROR_EXIT_CODE))
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)

    def _multi_worker_helper(self, world_size):
        addr = DEFAULT_HOSTNAME
        server_store = create_tcp_store(addr, world_size, wait_for_workers=False)
        server_store.set("key", "value")
        port = server_store.port
        messages = mp.Queue()
        processes = []
        num_proccesses = random.randint(3, 5) if world_size == -1 else world_size
        for i in range(num_proccesses):
            p = mp.Process(target=self._create_client, args=(i, addr, port, world_size, messages))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        error_message = ""
        while not messages.empty():
            error_message += messages.get() + "\n"
        if any([p.exitcode != 0 for p in processes]):
            raise RuntimeError(error_message)

    @unittest.skipIf(
        IS_WINDOWS, "Skip test for windows due to multiprocessing library error when using windows spawn"
    )
    def test_multi_worker_with_fixed_world_size(self):
        self._multi_worker_helper(5)

    @unittest.skipIf(
        IS_WINDOWS, "Skip test for windows due to multiprocessing library error when using windows spawn"
    )
    def test_multi_worker_with_nonfixed_world_size(self):
        self._multi_worker_helper(-1)


class PrefixTCPStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(PrefixTCPStoreTest, self).setUp()
        self.tcpstore = create_tcp_store()
        self.prefix = "test_prefix"
        self.tcpstore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return c10d.PrefixStore(self.prefix, self.tcpstore)

    # The PrefixTCPStore has 6 keys in test_set_get. It contains the 5 keys
    # added by the user and one additional key used for coordinate all the
    # workers.
    @property
    def num_keys_total(self):
        return 6


class MyPythonStore(c10d.Store):
    def __init__(self):
        super(MyPythonStore, self).__init__()
        self.store = dict()

    def set(self, key, value):
        if not isinstance(key, string_classes):
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


class PythonStoreTest(TestCase):
    def setUp(self):
        super(PythonStoreTest, self).setUp()

    def test_set_get(self):
        # If we were to inherit from StoreTestBase and try to use
        # its test_set_get function, we would exercise the Python
        # API directly, instead of going through the C++ trampoline.
        # We care about testing the C++ trampoline, so run the
        # equivalent of StoreTestBase.test_set_get from C++.
        # See `torch/csrc/distributed/c10d/init.cpp` for the definition
        # of this test function.
        c10d._test_python_store(MyPythonStore())


class RendezvousTest(TestCase):
    def test_unknown_handler(self):
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            c10d.rendezvous("invalid://")


class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    def test_nominal(self):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())

        # Single rank
        os.environ["RANK"] = "0"
        gen0 = c10d.rendezvous("env://")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(1, size0)

        store0.set("key0", "value0")

        # check with get
        self.assertEqual(b"value0", store0.get("key0"))


class RendezvousFileTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, "path missing"):
            gen = c10d.rendezvous("file://?rank=0&world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = c10d.rendezvous("file:///tmp/foo?world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = c10d.rendezvous("file:///tmp/foo?rank=0")
            next(gen)

    def test_nominal(self):
        with tempfile.NamedTemporaryFile(delete=False) as file:
            url = f'file:///{file.name.replace(os.path.sep, "/")}?world_size=2'
            gen0 = c10d.rendezvous(url + "&rank=0")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(2, size0)
            gen1 = c10d.rendezvous(url + "&rank=1")
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
        url = "tcp://%s:%d?world_size=%d" % (addr, port, 1)
        return url

    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, "port number missing"):
            gen = c10d.rendezvous("tcp://127.0.0.1?rank=0&world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "rank parameter missing"):
            gen = c10d.rendezvous("tcp://127.0.0.1:23456?world_size=1")
            next(gen)
        with self.assertRaisesRegex(ValueError, "size parameter missing"):
            gen = c10d.rendezvous("tcp://127.0.0.1:23456?rank=0")
            next(gen)

    @retry_on_connect_failures
    def test_nominal(self):
        url = self.create_tcp_url()
        gen0 = c10d.rendezvous(url + "&rank=0")
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
        test_store_timeout = timedelta(seconds=10)
        gen0 = c10d.rendezvous(url + "&rank=0", timeout=test_store_timeout)
        store0, rank0, size0 = next(gen0)
        # this should time out in 10s. If the timeout passed into rendezvous was
        # not respected, it will take much longer to timeout.
        start = time.time()
        with self.assertRaisesRegex(RuntimeError, "Timeout"):
            store0.get("nonexistant key")

        end = time.time()
        time_diff = end - start
        self.assertGreater(test_store_timeout.seconds * 10, time_diff)


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
