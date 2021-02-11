import copy
import math
import operator
import os
import random
import signal
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from datetime import timedelta
from functools import reduce
from itertools import groupby, product
from sys import platform

import torch
import torch.distributed as c10d
import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch._six import string_classes

from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    requires_nccl_version,
    skip_if_not_multigpu,
    skip_if_lt_x_gpu,
    get_timeout,
    skip_if_rocm,
    simple_sparse_reduce_tests,
    skip_if_win32,
    create_device,
)
from torch.testing._internal.common_utils import (
    TestCase,
    load_tests,
    run_tests,
    retry_on_connect_failures,
    ADDRESS_IN_USE,
    CONNECT_TIMEOUT,
    TEST_WITH_TSAN,
    slowTest,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)


if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"


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
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([world_size]),
        ),
    ]

    # Generate tests for BAND.
    # The bit that is set changes in every iteration to check
    # that the output changes accordingly.
    for i in range(4):
        vin = rank | (1 << i)
        vout = 1 << i
        tests.append(
            (
                c10d.ReduceOp.BAND,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for BOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-OR'ed.
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for XOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-XOR'ed.
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    return tests


def simple_coalesced_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([rank + 1]), torch.tensor([(rank + 1) ** 2])],
            [
                torch.tensor([float(world_size * (world_size + 1) / 2)]),
                torch.tensor(
                    [float(world_size * (world_size + 1) * (2 * world_size + 1) / 6)]
                ),
            ],
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([rank + 1.0]), torch.tensor([rank + 2.0])],
            [
                torch.tensor([float(math.factorial(world_size))]),
                torch.tensor([float(math.factorial(world_size + 1))]),
            ],
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([rank + x]) for x in [0.0, 1.0]],
            [torch.tensor([0.0]), torch.tensor([1.0])],
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([rank + x]) for x in [1.0, 2.0]],
            [torch.tensor([world_size]), torch.tensor([world_size + 1.0])],
        ),
    ]


def simple_multi_input_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])],
            torch.tensor([float(world_size * (2 * world_size - 1))]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([float(math.factorial(2 * world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([2 * world_size]),
        ),
    ]


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


class PrefixFileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(PrefixFileStoreTest, self).setUp()
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.filestore = c10d.FileStore(self.file.name, 1)
        self.prefix = "test_prefix"
        self.filestore.set_timeout(timedelta(seconds=300))

    def _create_store(self):
        return c10d.PrefixStore(self.prefix, self.filestore)


def create_tcp_store(addr):
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    ports = []
    for _ in range(10):
        try:
            port = common.find_free_port()
            ports.append(port)
            return c10d.TCPStore(addr, port, 1, True)
        except RuntimeError as error:
            if str(error) == "Address already in use":
                continue
            raise
    raise RuntimeError("Unable to find free port (tried %s)" % ", ".join(ports))


class TCPStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        store = create_tcp_store("localhost")
        store.set_timeout(timedelta(seconds=300))
        return store

    def test_address_already_in_use(self):
        if sys.platform == "win32":
            err_msg_reg = "Only one usage of each socket address*"
        else:
            err_msg_reg = "^Address already in use$"
        with self.assertRaisesRegex(RuntimeError, err_msg_reg):
            addr = "localhost"
            port = common.find_free_port()

            # Use noqa to silence flake8.
            # Need to store in an unused variable here to ensure the first
            # object is not destroyed before the second object is created.
            store1 = c10d.TCPStore(addr, port, 1, True)  # noqa: F841
            store2 = c10d.TCPStore(addr, port, 1, True)  # noqa: F841

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

    # https://github.com/pytorch/pytorch/issues/46064 <- takes 5+ min to finish
    @slowTest
    def test_numkeys_delkeys(self):
        self._test_numkeys_delkeys(self._create_store())

    def test_compare_set(self):
        store = self._create_store()
        missing_key_result = store.compare_set("key0", "wrong_old_value", "new_value0")
        self.assertEqual(b"wrong_old_value", missing_key_result)

        store.set("key0", "value0")
        self.assertEqual(b"value0", store.get("key0"))
        old_value_result = store.compare_set("key0", "wrong_old_value", "new_value0")
        self.assertEqual(b"wrong_old_value", old_value_result)
        self.assertEqual(b"value0", store.get("key0"))
        new_value_result = store.compare_set("key0", "value0", "new_value0")
        self.assertEqual(b"new_value0", new_value_result)
        self.assertEqual(b"new_value0", store.get("key0"))

class PrefixTCPStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        super(PrefixTCPStoreTest, self).setUp()
        self.tcpstore = create_tcp_store("localhost")
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
    @requires_nccl()
    def test_common_errors(self):
        if torch.cuda.device_count() == 0:
            raise unittest.SkipTest("No GPUs available, skipping test")

        vars = {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": common.find_free_port(),
        }

        class Env(object):
            def __init__(self, vars):
                self.vars = vars

            def __enter__(self):
                for key, value in self.vars.items():
                    os.environ[key] = str(value)

            def __exit__(self, type, value, traceback):
                for key in self.vars.keys():
                    del os.environ[key]

        def without(d, key):
            d = d.copy()
            d.pop(key)
            return d

        def withouts(d, keys):
            d = d.copy()
            for key in keys:
                d.pop(key)
            return d

        with Env(without(vars, "WORLD_SIZE")):
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="nccl", world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "RANK")):
            with self.assertRaisesRegex(ValueError, "RANK expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="nccl", rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            c10d.init_process_group(backend="nccl", rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend="nccl")
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "MASTER_ADDR")):
            with self.assertRaisesRegex(ValueError, "MASTER_ADDR expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "MASTER_PORT")):
            with self.assertRaisesRegex(ValueError, "MASTER_PORT expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "WORLD_SIZE")):
            gen = c10d.rendezvous("env://?world_size={}".format(1))
            _, _, size = next(gen)
            self.assertEqual(size, 1)

        with Env(without(vars, "RANK")):
            gen = c10d.rendezvous("env://?rank={}".format(0))
            _, rank, _ = next(gen)
            self.assertEqual(rank, 0)

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            gen = c10d.rendezvous("env://?rank={}&world_size={}".format(0, 1))
            _, rank, size = next(gen)
            self.assertEqual(rank, 0)
            self.assertEqual(size, 1)

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
        addr = "localhost"
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


class TimeoutTest(TestCase):
    def _test_store_timeout(self, backend, init_method, c2p):
        try:
            c10d.distributed_c10d.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=1,
                rank=0,
                timeout=timedelta(seconds=1),
            )
            default_store = c10d.distributed_c10d._get_default_store()
            tik = time.time()
            with self.assertRaisesRegex(RuntimeError, "Timeout"):
                default_store.get("nonexistent key")
            tok = time.time()
            c10d.destroy_process_group()
            c2p.append(float(tok - tik))
        except RuntimeError as e:
            # catch "Address already in use" error and report it to the main
            # thread
            c2p.append(e)

    def _init_methods(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        if sys.platform == "win32":
            yield "file:///%s" % f.name.replace("\\", "/")
            f.close()
        else:
            yield "file://%s" % f.name
            f.close()
            yield "tcp://127.0.0.1:%d" % common.find_free_port()

    def _test_default_store_timeout(self, backend):
        for init_method in self._init_methods():
            c2p = []
            t = threading.Thread(
                target=self._test_store_timeout, args=(backend, init_method, c2p)
            )
            t.daemon = True
            t.start()
            t.join(5)

            self.assertEqual(1, len(c2p))
            if isinstance(c2p[0], float):
                # waiting time should be 1s, use 3s to rule out false alarm
                self.assertGreater(3, c2p[0])
            elif isinstance(c2p[0], RuntimeError):
                # let @retry_on_connect_failures handle the error
                raise c2p[0]
            else:
                raise RuntimeError("Unexpected type {}".format(type(c2p[0])))

    @requires_nccl()
    @retry_on_connect_failures
    def test_default_store_timeout_nccl(self):
        if torch.cuda.device_count() == 0:
            raise unittest.SkipTest("No GPUs available, skipping test")
        self._test_default_store_timeout("nccl")

    @requires_gloo()
    @retry_on_connect_failures
    def test_default_store_timeout_gloo(self):
        self._test_default_store_timeout("gloo")


@requires_gloo()
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class ProcessGroupGlooTest(MultiProcessTestCase):
    def setUp(self):
        super(ProcessGroupGlooTest, self).setUp()

        # For Windows platform, Python does not support fork, change it to spawn here.
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def opts(self, threads=2):
        opts = c10d.ProcessGroupGloo.Options()
        opts.devices = [create_device(interface=LOOPBACK)]
        opts.timeout = 5.0
        opts.threads = threads
        return opts

    def test_multi_device_constructor(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = c10d.ProcessGroupGloo.Options()
        opts.timeout = 5.0
        opts.devices = [
            create_device(interface=LOOPBACK),
            create_device(interface=LOOPBACK),
        ]
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, opts)

        # Execute 2x the number of operations to ensure we use every device.
        for work in [pg.allreduce(torch.ones(i + 1)) for i in range(4)]:
            work.wait()

    def test_empty_tensors(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        xs = [torch.FloatTensor([])]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())

    def test_broadcast_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)

    def _test_broadcast_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            broadcast([x], i, 0)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(torch.tensor([i]), x)

            # Run with 2 input tensors
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.tensor([self.rank * num + 0.0])),
                    fn(torch.tensor([self.rank * num + 1.0])),
                ]

                broadcast(xs, i, j)
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.tensor([i * num + j]), xs[0])
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.tensor([i * num + j]), xs[1])

        # Test overloaded convenience function
        x = torch.tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.tensor([1.0]), x)

    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_broadcast_basics_cuda(self):
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_broadcast_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.tensor([(i * self.world_size) + (i % self.world_size)]),
                inputs[i],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_broadcast_stress(self):
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @skip_if_not_multigpu
    def test_broadcast_stress_cuda(self):
        inputs = [
            torch.tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)
        ]
        self._test_broadcast_stress(inputs)

    def test_allreduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    def _test_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            work = pg.allreduce([tensor], opts)
            work.wait()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(output, tensor)

        # Multi input tests
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for (op, inputs, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            work = pg.allreduce(tensors, opts)
            work.wait()
            for tensor in tensors:
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(output, tensor)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        work = pg.allreduce(x)
        work.wait()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]), x
        )

    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_allreduce_basics_cuda(self):
        self._test_allreduce_basics(lambda t: t.clone().cuda())

    def _test_allreduce_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = [pg.allreduce(inputs[i]) for i in range(len(inputs))]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(
                torch.tensor(
                    [
                        (i * self.world_size)
                        + (self.world_size * (self.world_size - 1) / 2)
                    ]
                ),
                inputs[i],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_allreduce_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @skip_if_not_multigpu
    def test_allreduce_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allreduce_stress(inputs)

    def test_allreduce_coalesced_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros(1, dtype=torch.float32)
        t2 = torch.zeros(1, dtype=torch.float64)
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(1,))

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([], opts)

        with self.assertRaisesRegex(ValueError, "tensors must all have the same type"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor layout at index"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t3], opts)

        with self.assertRaisesRegex(ValueError, "unsupported layout"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t3, t3.clone()], opts)

    @skip_if_lt_x_gpu(1)
    def test_allreduce_coalesced_checks_cuda(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros(1, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "unsupported device type"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1.cuda(), t1.cuda()], opts)

    def _test_allreduce_coalesced_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        test_cases = simple_coalesced_reduce_tests(self.rank, self.world_size)
        for op, inputs, outputs in test_cases:
            opts = c10d.AllreduceCoalescedOptions()
            opts.reduceOp = op
            tensors = [fn(x) for x in inputs]
            work = pg.allreduce_coalesced(tensors, opts)
            work.wait()
            for result_tensor, expected in zip(tensors, outputs):
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(result_tensor, expected)

    def test_allreduce_coalesced_basics(self):
        self._test_allreduce_coalesced_basics(lambda t: t.clone())

    def _test_allreduce_coalesced_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = [pg.allreduce_coalesced(input) for input in inputs]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(
                2
                * [
                    torch.tensor(
                        [
                            (i * self.world_size)
                            + (self.world_size * (self.world_size - 1) / 2)
                        ]
                    )
                ],
                inputs[i],
                msg="Mismatch in interation {}".format(i),
            )

    def test_allreduce_coalesced_stress(self):
        inputs = [2 * [torch.tensor([i + self.rank])] for i in range(1000)]
        self._test_allreduce_coalesced_stress(inputs)

    def test_sparse_allreduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1])
        t2 = torch.sparse_coo_tensor([[0]], [1], size=(2,))
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(4,))

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor layout"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t2, t3], opts)

        # Sparse allreduce only works with c10d.ReduceOp.SUM.
        for op in [c10d.ReduceOp.PRODUCT, c10d.ReduceOp.MIN, c10d.ReduceOp.MAX]:
            with self.assertRaisesRegex(ValueError, "unsupported reduction operation"):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                pg.allreduce([t3], opts)

    def _test_sparse_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        for num_inputs_per_rank in [1, 2]:
            tests = simple_sparse_reduce_tests(
                self.rank, self.world_size, num_inputs=num_inputs_per_rank
            )
            for (inputs, outputs) in tests:
                tensors = [fn(input) for input in inputs]
                work = pg.allreduce(tensors)
                work.wait()
                self.assertEqual(tensors, outputs)
                self.assertEqual(work.result(), outputs)

    def test_sparse_allreduce_basics(self):
        self._test_sparse_allreduce_basics(lambda t: t)

    @skip_if_not_multigpu
    def test_sparse_allreduce_basics_cuda(self):
        self._test_sparse_allreduce_basics(lambda t: t.clone().cuda())

    def test_scatter_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], [], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([t1, t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * self.world_size, [t1] * self.world_size], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect input list size {}. Input list size should be {}"
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t2] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t3] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "requires empty input on non-root"):
            opts = c10d.ScatterOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.scatter([t1], [[t1] * self.world_size], opts)

    def _test_scatter_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank])) for _ in range(self.world_size)]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the scatter root and accumulate work items
        work = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                work.append(pg.scatter([outputs[i]], [input], opts))
            else:
                work.append(pg.scatter([outputs[i]], [], opts))

        # Wait for work to complete
        for i in range(self.world_size):
            work[i].wait()
            self.assertEqual(torch.tensor([i]), outputs[i])

    def test_scatter_basics(self):
        self._test_scatter_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_scatter_basics_cuda(self):
        self._test_scatter_basics(lambda t: t.clone().cuda())

    def _test_scatter_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        outputs = [
            [fn(torch.tensor([-1])) for _ in range(self.world_size)]
            for _ in range(len(inputs))
        ]
        work_handles = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ScatterOptions()
                opts.rootRank = root
                if root == self.rank:
                    work = pg.scatter(
                        [outputs[i][root]], [[fn(e) for e in inputs[i]]], opts
                    )
                else:
                    work = pg.scatter([outputs[i][root]], [], opts)
                work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size

            self.assertEqual(
                torch.tensor([iter + root]),
                outputs[iter][root],
                msg=("Mismatch in iteration %d for rank %d" % (iter, root)),
            )

    def test_scatter_stress(self):
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone())

    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/15963")
    @skip_if_not_multigpu
    def test_scatter_stress_cuda(self):
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone().cuda())

    def test_gather_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [t1, t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * self.world_size, [t1] * self.world_size], [t1], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect output list size {}. Output list size should be {}"
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t2] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t3] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires empty output on non-root"):
            opts = c10d.GatherOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.gather([[t1] * self.world_size], [t1], opts)

    def _test_gather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank]))]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the gather root and accumulate work items
        work = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                work.append(pg.gather([outputs], input, opts))
            else:
                work.append(pg.gather([], input, opts))

        # Wait for work to complete
        expected = [torch.tensor([rank]) for rank in range(self.world_size)]
        for i in range(self.world_size):
            work[i].wait()
            if i == self.rank:
                self.assertEqual(expected, outputs)

    def test_gather_basics(self):
        self._test_gather_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_gather_basics_cuda(self):
        self._test_gather_basics(lambda t: t.clone().cuda())

    def _test_gather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.GatherOptions()
                opts.rootRank = root
                if root == self.rank:
                    work = pg.gather(outputs[i], [fn(inputs[i])], opts)
                else:
                    work = pg.gather([], [fn(inputs[i])], opts)
                work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                self.assertEqual(
                    expected_outputs[iter],
                    outputs[iter],
                    msg=("Mismatch in iteration %d for root %d" % (iter, root)),
                )

    def test_gather_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone())

    @skip_if_not_multigpu
    def test_gather_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone().cuda())

    def test_allgather_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty input tensor list"):
            pg.allgather([], [])

        with self.assertRaisesRegex(
            ValueError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([], [t1])

        with self.assertRaisesRegex(
            ValueError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t2]
            )

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t3]
            )

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather([([t1, t2] * (self.world_size))[: self.world_size]], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather([([t1, t3] * (self.world_size))[: self.world_size]], [t1])

    def _test_allgather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Run with N input tensor per rank
        for n in [1, 2, 3]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [
                [fn(torch.tensor([-1])) for _ in range(n * self.world_size)]
                for _ in range(n)
            ]
            expected_output = [
                [torch.tensor([i]) for i in range(n * self.world_size)]
                for _ in range(n)
            ]
            work = pg.allgather(output, input)
            work.wait()
            self.assertEqual(expected_output, output)

    def test_allgather_basics(self):
        self._test_allgather_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_allgather_basics_cuda(self):
        self._test_allgather_basics(lambda t: t.clone().cuda())

    def _test_allgather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            work = pg.allgather(outputs[i], [fn(inputs[i])])
            work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                expected_outputs[i],
                outputs[i],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_allgather_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone())

    @skip_if_not_multigpu
    def test_allgather_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone().cuda())

    def test_allgather_coalesced_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())
        dummy_input = [torch.zeros([1], dtype=torch.float32)]
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size)
        ]

        # One of output tensors does not match input list.
        dummy_output_lists[0] = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(
            ValueError, "invalid size of output tensor at index 0"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # One of output tensors does not match input list.
        dummy_output_lists[0] = [torch.zeros([1], dtype=torch.float64)]
        with self.assertRaisesRegex(ValueError, "invalid tensor type at index 0"):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # Output lists have too many elements
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size + 1)
        ]
        with self.assertRaisesRegex(
            ValueError, "output lists should be equal to world size"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # Output is not a list of lists.
        dummy_output_lists = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(
            RuntimeError, "Invalid function argument.*output_tensor_lists"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

    def test_reduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element tensor list"
        ):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.reduce([t1, t1], opts)

    def _test_reduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())
        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                work = pg.reduce([tmp], opts)
                work.wait()
                if root == self.rank:
                    # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                    self.assertEqualIgnoreType(output, tmp)

    def test_reduce_basics(self):
        self._test_reduce_basics(lambda t: t.clone())

    @skip_if_not_multigpu
    def test_reduce_basics_cuda(self):
        self._test_reduce_basics(lambda t: t.clone().cuda())

    def _test_reduce_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = []
        outputs = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.rootRank = root
                tmp = inputs[i].clone()
                outputs.append(tmp)
                work = pg.reduce([tmp], opts)
                work_handles.append(work)

        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(
                    torch.tensor(
                        [
                            (iter * self.world_size)
                            + (self.world_size * (self.world_size - 1) / 2)
                        ]
                    ),
                    outputs[i],
                    msg=("Mismatch in iteration %d with root rank %d" % (iter, root)),
                )

    def test_reduce_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_reduce_stress(inputs)

    @skip_if_not_multigpu
    def test_reduce_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_reduce_stress(inputs)

    def test_send_recv_all_to_all(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]

        # Issue sends
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # Issue recvs
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # Wait for sends to complete
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Wait for recvs to complete
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Test that every output other than our own contains the respective rank
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.tensor([i]), outputs[i])

    def test_barrier_implies_wait(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Kick off allreduce operations
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            # Note: leak the returned work handle
            pg.allreduce(tensor)

        # Barrier should ensure all previous work has completed
        pg.barrier().wait()

        for i, tensor in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)

    @skip_if_win32()
    def test_round_robin(self):
        num_process_groups = 2
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d._round_robin_process_groups(
            [
                c10d.ProcessGroupGloo(
                    c10d.PrefixStore(str(i), store), self.rank, self.world_size
                )
                for i in range(num_process_groups)
            ]
        )

        # Run a few collectives so that we have called each process group
        for _ in range(num_process_groups + 1):
            tensor = torch.full([100, 100], float(self.rank))
            pg.broadcast(tensor, root=0).wait()
            self.assertEqual(torch.full([100, 100], 0.0), tensor)

    @skip_if_win32()
    def test_round_robin_create_destroy(self):
        store = c10d.FileStore(self.file_name, self.world_size)

        def create(num, prefix):
            return c10d._round_robin_process_groups(
                [
                    c10d.ProcessGroupGloo(
                        c10d.PrefixStore("%s/%d" % (prefix, i), store),
                        self.rank,
                        self.world_size,
                    )
                    for i in range(num)
                ]
            )

        # Run create/use/destroy twice
        for i in range(2):
            num_process_groups = 2
            pg = create(num=num_process_groups, prefix=i)
            for _ in range(3):
                tensor = torch.ones([10, 10])
                pg.allreduce(tensor).wait()
                self.assertEqual(torch.full([10, 10], float(self.world_size)), tensor)
            del pg


class ProcessGroupNCCLNoGPUTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 0:
            raise unittest.SkipTest("GPUs are available, skipping test")

    def tearDown(self):
        pass

    @requires_nccl()
    def test_init_no_gpus(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        with self.assertRaisesRegex(
            RuntimeError, "ProcessGroupNCCL is only supported with GPUs, no GPUs found!"
        ):
            c10d.ProcessGroupNCCL(store, self.rank, self.world_size)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class ProcessGroupNCCLTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")

    def tearDown(self):
        pass

    @requires_nccl()
    def test_empty_tensors(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        xs = [torch.cuda.FloatTensor([])]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.allreduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.reduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        ys = [[torch.cuda.FloatTensor([]) for _ in range(self.world_size)]]
        pg.allgather(ys, xs).wait()
        for y in ys[0]:
            self.assertEqual(0, y.numel())

        ys = [torch.cuda.FloatTensor([])]
        xs = [[torch.cuda.FloatTensor([]) for _ in range(self.world_size)]]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    @requires_nccl()
    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.num_gpus):
            tensors = []
            for i in range(self.num_gpus):
                tensors.append(torch.tensor([i]).cuda(i))

            broadcast(tensors, self.rank, rt)

            for i in range(self.num_gpus):
                self.assertEqual(tensors[i], tensors[rt])

    @requires_nccl()
    def test_allreduce_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce(tensors, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        # Sum
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.SUM)

        for i in range(self.num_gpus):
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(
                torch.tensor([float(self.num_gpus * (self.num_gpus + 1) / 2)]),
                tensors[i],
            )

        # Product
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.PRODUCT)

        for i in range(self.num_gpus):
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(
                torch.tensor([float(math.factorial(self.num_gpus))]), tensors[i]
            )

        # Min
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MIN)

        for i in range(self.num_gpus):
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(torch.tensor([1.0]), tensors[i])

        # Max
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MAX)

        for i in range(self.num_gpus):
            self.assertEqual(torch.tensor([self.num_gpus]), tensors[i])

        for op in (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR):
            with self.assertRaisesRegex(
                RuntimeError, "Cannot use " + str(op) + " with NCCL"
            ):
                allreduce(tensors, op)

    @requires_nccl()
    def test_reduce_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def reduce(xs, rootRank, rootTensor, op=None):
            opts = c10d.ReduceOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            if op:
                opts.reduceOp = op
            work = pg.reduce(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.num_gpus):
            tensors = []
            for i in range(self.num_gpus):
                tensors.append(torch.tensor([i + 1]).cuda(i))

            reduce(tensors, self.rank, rt)

            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(
                torch.tensor([float(self.num_gpus * (self.num_gpus + 1) / 2)]),
                tensors[rt],
            )

            for op in (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR):
                with self.assertRaisesRegex(
                    RuntimeError, "Cannot use " + str(op) + " with NCCL"
                ):
                    reduce(tensors, self.rank, rt, op)

    @requires_nccl()
    def test_allgather_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allgather(output_ts, input_ts):
            work = pg.allgather(output_ts, input_ts)
            work.wait()

        tensors = []
        output_ts = [[] for _ in range(self.num_gpus)]

        for idx, ls in enumerate(output_ts):
            for _ in range(self.world_size * self.num_gpus):
                ls.append(torch.tensor([0]).cuda(idx))

        for i in range(self.num_gpus):
            tensors.append(torch.tensor([i]).cuda(i))

        allgather(output_ts, tensors)

        # Verification
        for device_ts in output_ts:
            for s_idx, t in enumerate(device_ts):
                self.assertEqual(torch.tensor([s_idx]), t)

    @requires_nccl()
    def test_reduce_scatter_ops(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def reduce_scatter(outputs, input_lists, op):
            opts = c10d.ReduceScatterOptions()
            opts.reduceOp = op
            work = pg.reduce_scatter(outputs, input_lists, opts)
            work.wait()

        virtual_rank = self.rank * self.world_size
        virtual_world_size = self.num_gpus * self.world_size

        output = [torch.tensor([0]).cuda(i) for i in range(self.num_gpus)]

        #           0                   1                   2
        #   0   [0..11]             [1..12]
        #   1   [3..14]
        #   2
        #   3

        # Sum
        tensor_lists = [
            [
                torch.tensor([self.rank * self.num_gpus + i + j]).cuda(i)
                for j in range(virtual_world_size)
            ]
            for i in range(self.num_gpus)
        ]

        reduce_scatter(output, tensor_lists, c10d.ReduceOp.SUM)

        for i in range(self.num_gpus):
            expected = torch.tensor(
                [
                    float(self.num_gpus * (self.num_gpus - 1) / 2)
                    + (virtual_rank + i) * virtual_world_size
                ]
            )
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(expected, output[i])

        # Min
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MIN)

        for i in range(self.num_gpus):
            expected = torch.tensor([self.rank * self.world_size + i])
            self.assertEqual(expected, output[i])

        # Max
        reduce_scatter(output, tensor_lists, c10d.ReduceOp.MAX)

        for i in range(self.num_gpus):
            expected = torch.tensor(
                [self.rank * self.world_size + i + virtual_world_size - 1]
            )
            self.assertEqual(expected, output[i])

        # Product
        tensor_lists = [
            [
                torch.tensor(
                    [(self.rank * self.num_gpus + i + j) % virtual_world_size + 1]
                ).cuda(i)
                for j in range(virtual_world_size)
            ]
            for i in range(self.num_gpus)
        ]

        reduce_scatter(output, tensor_lists, c10d.ReduceOp.PRODUCT)

        for i in range(self.num_gpus):
            expected = torch.tensor([float(math.factorial(virtual_world_size))])
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(expected, output[i])

    @requires_nccl()
    def test_barrier(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce(tensors):
            opts = c10d.AllreduceOptions()
            work = pg.allreduce(tensors, opts)
            return work

        # Making the collective to operate on
        # 1, 2, 3, 4, .... self.num_gpus GPUs
        tensors_list = [[] for _ in range(2, self.num_gpus + 1)]
        for i in range(2, self.num_gpus + 1):
            for j in range(i):
                tensors_list[i - 2].append(torch.tensor([j + 1]).cuda(j))

        works = []
        for tensors in tensors_list:
            work = allreduce(tensors)
            works.append(work)

        # Barrier will ensure that all previous work is completed
        pg.barrier().wait()

        for i in range(2, self.num_gpus + 1):
            for j in range(i):
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(
                    torch.tensor([float(i * (i + 1) / 2)]), tensors_list[i - 2][j]
                )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DoubleGpuNet(nn.Module):
    def __init__(self, gpus):
        super(DoubleGpuNet, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[1])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.fc3(x)
        return F.softmax(x, dim=1).to(dev0)


class QuadraGpuNet(nn.Module):
    def __init__(self, gpus):
        super(QuadraGpuNet, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[2])
        self.fc4 = nn.Linear(4, 4, bias=False).to(gpus[3])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])

    def forward(self, x):
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        dev2 = self.fc3.weight.device
        dev3 = self.fc4.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.relu(self.fc3(x.to(dev2)))
        x = self.fc4(x.to(dev3))
        return F.softmax(x, dim=1).to(dev0)


class ConvNet(nn.Module):
    def __init__(self, gpus, layouts, dtypes):
        super(ConvNet, self).__init__()
        self.dtypes = dtypes
        if isinstance(gpus, list):
            self.layer_gpus = gpus
        else:
            gpus = [gpus] * 4
        self.conv0 = torch.nn.Conv2d(8, 16, (2, 2)).to(
            device=gpus[0], memory_format=layouts[0], dtype=dtypes[0]
        )
        self.conv1 = torch.nn.Conv2d(16, 32, (2, 2)).to(
            device=gpus[1], memory_format=layouts[1], dtype=dtypes[1]
        )
        self.conv2 = torch.nn.Conv2d(32, 16, (2, 2)).to(
            device=gpus[2], memory_format=layouts[2], dtype=dtypes[2]
        )
        self.conv3 = torch.nn.Conv2d(16, 8, (2, 2)).to(
            device=gpus[3], memory_format=layouts[3], dtype=dtypes[3]
        )

    def forward(self, x):
        x = x.to(self.dtypes[0])
        # Could say
        # x = self.conv0(x).to(device=self.conv1.weight.device, dtype=self.dtypes[1])
        # etc.  But I don't want to appeal to the weights' devices directly, because part of this test's purpose
        # is to verify weights are where expected if the model gets replicated.
        gpus = self.layer_gpus if hasattr(self, "layer_gpus") else [x.device] * 4
        x = self.conv0(x).to(device=gpus[1], dtype=self.dtypes[1])
        x = self.conv1(x).to(device=gpus[2], dtype=self.dtypes[2])
        x = self.conv2(x).to(device=gpus[3], dtype=self.dtypes[3])
        return self.conv3(x)


class Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        return self.p + x


class ModuleForDdpCommHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        return self.t0(x + rank)


class SparseGradientModule(nn.Module):
    def __init__(self):
        super(SparseGradientModule, self).__init__()
        self.embedding = nn.EmbeddingBag(10, 10, sparse=True)

    def forward(self, x):
        return F.softmax(self.embedding(x), dim=1)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class DistributedDataParallelTest(MultiProcessTestCase):
    def setUp(self):
        super(DistributedDataParallelTest, self).setUp()
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def tearDown(self):
        # DistributedDataParallel test doesn't seem to call FileStore destructor
        # TODO: investigate this test and the test is known to have issues
        # Use this hack to remove files for that test
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def _prepare_single_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        model = Net()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).to(devices[0]),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        model.to(devices[0])

        input = torch.randn(global_batch_size, 2).to(devices[0])
        target = torch.randn(global_batch_size, 4).to(devices[0])

        return model, ddp_model, input, target

    def _prepare_multi_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        self.assertTrue(
            len(devices) == 2 or len(devices) == 4,
            "unexpected devices for ddp tests {}".format(devices),
        )
        if len(devices) == 2:
            model = DoubleGpuNet(devices)
        elif len(devices) == 4:
            model = QuadraGpuNet(devices)

        ddp_model = DistributedDataParallel(
            copy.deepcopy(model),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        input = torch.randn(global_batch_size, 2).cuda(devices[0])
        target = torch.randn(global_batch_size, 4)

        return model, ddp_model, input, target

    def _test_ddp_with_process_group(
        self,
        process_group,
        devices,
        device_ids,
        multi_device=False,
        gradient_as_bucket_view=False,
    ):
        """
        Note: we pass down `device_ids` all the way to DistributedDataParallel
        as part of the test. Below you find tests that either use a list of
        integers, a list of `torch.Device` instances, or an empty list.
        The `devices` argument is used to control placement of the model and
        must always be specified as list of `torch.Device` instances.
        """
        local_batch_size = len(devices)
        global_batch_size = self.world_size * local_batch_size

        if multi_device:
            model, ddp_model, input, target = self._prepare_multi_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
        else:
            model, ddp_model, input, target = self._prepare_single_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        def update_parameters(model):
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        # check two model parameters over 2 iterations
        for iteration in range(2):
            # single cpu/gpu training
            step_model(model, input, target)

            # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
            step_model(
                ddp_model,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            # Update weights and run a second iteration to shake out errors
            update_parameters(model)
            update_parameters(ddp_model)
            self.assertEqual(
                len(list(model.parameters())), len(list(ddp_model.parameters()))
            )
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                self.assertEqual(i, j)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    def _test_gloo_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        store = c10d.FileStore(self.file_name, self.world_size)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [create_device(interface=LOOPBACK)]
        process_group = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, options
        )
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_gloo()
    def test_gloo_backend_cpu_module(self):
        self._test_gloo_backend([torch.device("cpu")], [])

    @requires_gloo()
    def test_gloo_backend_cpu_module_grad_is_view(self):
        self._test_gloo_backend([torch.device("cpu")], [], gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_not_multigpu
    def test_gloo_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, int_devices)

    @requires_gloo()
    @skip_if_not_multigpu
    def test_gloo_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, devices)

    @requires_gloo()
    @skip_if_lt_x_gpu(4)
    def test_gloo_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, [], multi_device=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(8)
    def test_gloo_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, [], multi_device=True)

    def _test_nccl_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_nccl_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_not_multigpu
    def test_nccl_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, [], multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    def test_nccl_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, [], multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_ddp_multi_device_module_config(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]

        self.assertTrue(len(gpus) >= 2, "expecting at least 2 gpus per process")

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        gpus = gpus[:2]
        model = DoubleGpuNet(gpus)

        with self.assertRaisesRegex(
            AssertionError, "output_device .* single-device GPU"
        ):
            ddp_model = DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group
            )

        with self.assertRaisesRegex(AssertionError, "device_ids .* single-device GPU"):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

        with self.assertRaisesRegex(
            AssertionError, "input module must be on the same type of devices"
        ):
            model.fc1 = model.fc1.cpu()
            ddp_model = DistributedDataParallel(model, process_group=process_group)

        model = model.cpu()
        with self.assertRaisesRegex(AssertionError, "device_ids .* single-device GPU"):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

    def _test_fp16(self, gradient_as_bucket_view=False):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.tensor([[2 ** 15]]).cuda(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))

    @requires_nccl()
    @skip_if_not_multigpu
    def test_fp16(self):
        self._test_fp16()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_fp16_grad_is_view(self):
        self._test_fp16(gradient_as_bucket_view=True)

    def _test_arbitrary_forward_return_value(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class ForwardReturnValueModule(nn.Module):
            def __init__(self):
                super(ForwardReturnValueModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x, fn):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # The first softmax does NOT include fc3 in its autograd graph
                # whereas the second softmax DOES. If we pass only the first
                # tensor we see in the output to the reducer, it marks the
                # gradient for fc3 as ready (because it doesn't show up). If
                # downstream uses of this return value choose to differentiate
                # against the second output tensor, it would still receive a
                # gradient and a callback for this tensor, resulting in a crash.
                return fn(
                    F.softmax(x, dim=1),
                    F.softmax(self.fc3(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            ForwardReturnValueModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Always run "backward" to ensure the reducer is called by autograd.
        # If we don't correctly capture the output tensors from the return value,
        # the reducer won't see a hook for the unused parameter, and throw an error.
        # The correct capture is what we're testing in this function.
        def test(box, unbox):
            output = model(input, fn=box)
            loss = criterion(unbox(output), target)
            loss.backward()

        # Test with identity return value
        test(
            box=lambda x, y: (x, y),
            unbox=lambda obj: obj[1],
        )

        # Test with list return value
        test(
            box=lambda x, y: ["foo", x, "bar", y],
            unbox=lambda obj: obj[3],
        )

        # Test with tuple return value
        test(
            box=lambda x, y: ("foo", x, "bar", y),
            unbox=lambda obj: obj[3],
        )

        # Test with dict return value
        test(
            box=lambda x, y: {"foo": "bar", "a": x, "b": y},
            unbox=lambda obj: obj["b"],
        )

        # Test with list with dict return value
        test(
            box=lambda x, y: ["foo", "bar", {"a": x, "b": y}],
            unbox=lambda obj: obj[2]["b"],
        )

        # Test with dict with list return value
        test(
            box=lambda x, y: {"foo": "bar", "list": [0, x, 1, y]},
            unbox=lambda obj: obj["list"][3],
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_arbitrary_forward_return_value(self):
        self._test_arbitrary_forward_return_value()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_not_multigpu
    def test_ddp_with_lazy_parameters(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        with self.assertRaisesRegex(
            RuntimeError, "Modules with uninitialized parameters"
        ):
            DistributedDataParallel(
                torch.nn.LazyLinear(10), process_group=process_group
            )

    def _test_find_unused_parameters_kwarg(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class FindUnusedParametersModule(nn.Module):
            def __init__(self):
                super(FindUnusedParametersModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # Return the fc3 module so that the caller can invoke it
                # outside of the forward function. While this is bad practice,
                # we can use it to trigger a reducer error.
                return (F.softmax(x, dim=1), self.fc3)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        def test_find_unused_parameters(
            find_unused_parameters, test_default=False, gradient_as_bucket_view=False
        ):
            if test_default:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            else:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    find_unused_parameters=find_unused_parameters,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )

            output, fc3 = model(input)
            output = fc3(output)
            loss = criterion(output, target)
            loss.backward()

        # First test that finding unused params under these conditions is to
        # trigger an error when `backward` is called (because fc3 is an unused
        # parameter and will therefore be marked ready twice).
        try:
            test_find_unused_parameters(
                True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.assertTrue(
                str(ex).startswith("Expected to mark a variable ready only once.")
            )
        else:
            self.fail("Expected exception")

        # Then test that the default behavior can be overridden by setting
        # `find_unused_parameters=False`.
        try:
            test_find_unused_parameters(
                False, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

        # Test find_unused_parameters defaults to False
        try:
            test_find_unused_parameters(
                True, test_default=True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail("Unexpected exception: %s" % ex)

    @requires_nccl()
    @skip_if_not_multigpu
    def test_find_unused_parameters_kwarg(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_find_unused_parameters_kwarg_grad_is_view(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    def _test_global_local_unused_params_grad(self, gradient_as_bucket_view=False):
        """
        By simulating a multi-task training, this test is to make sure:
        1) DDP does not touch the grad of globally unused parameters.
        2) DDP does update the grad of locally unused parameters.
        """

        class GlobalLocalUnusedParamModule(nn.Module):
            def __init__(self):
                super(GlobalLocalUnusedParamModule, self).__init__()
                self.t0 = Task()
                self.t1 = Task()
                self.task_unused = Task()

            def task_parameters(self):
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                return self.t0(x) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            t0_p, t1_p, task_unused_p = model.module.task_parameters()
            self.assertIsNone(t0_p.grad)
            self.assertIsNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            # However the globally unused parameter should still have None grad.
            self.assertIsNotNone(t0_p.grad)
            self.assertIsNotNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Test on CPU
        cpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        run_and_verify_grad(gpu_model)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        self._test_global_local_unused_params_grad()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        """
        An empty unused_parameters array does not imply find_unused_parameters =
        false. This test makes sure that DDP allreduces unused parameters
        accordingly where the forward pass in some process uses all parameters.
        This unit test creates a module that uses all parameters in rank = 0, and
        has unused parameters in other ranks.
        """

        class FindUnusedParamModule(nn.Module):
            def __init__(self):
                super(FindUnusedParamModule, self).__init__()
                self.t0 = Task()
                self.t1 = Task()

            def task_parameters(self):
                return (self.t0.p, self.t1.p)

            def forward(self, x, rank):
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            [self.assertIsNone(t_p.grad) for t_p in model.module.task_parameters()]

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            [self.assertIsNotNone(t_p.grad) for t_p in model.module.task_parameters()]

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Test on CPU
        cpu_model = DistributedDataParallel(
            FindUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            FindUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(gpu_model)

    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class MultipleOutputModule(nn.Module):
            def __init__(self):
                super(MultipleOutputModule, self).__init__()

                def define_module():
                    return nn.Sequential(
                        nn.Linear(2, 10, bias=False),
                        nn.ReLU(),
                        nn.Linear(10, 4, bias=False),
                        nn.ReLU(),
                    )

                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                return (
                    F.softmax(self.module0(x), dim=1),
                    F.softmax(self.module1(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            MultipleOutputModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Compute loss and gradients for both outputs
        output1, output2 = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        loss2 = criterion(output2, target)
        loss2.backward()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_multiple_outputs_multiple_backward(self):
        self._test_multiple_outputs_multiple_backward()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_not_multigpu
    def test_no_grad(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        class NoGradModule(nn.Module):
            def __init__(self):
                super(NoGradModule, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            NoGradModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        def check_no_grads():
            for p in model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNone(p.grad)

        # After initialization, no parameter has their gradient set.
        check_no_grads()

        # Run `forward` function with torch.no_grad()
        with torch.no_grad():
            output = model(input)
            self.assertTrue(isinstance(output, torch.Tensor))

        # No parameter should have their gradient set.
        check_no_grads()

    def _test_accumulate_gradients_no_sync(
        self, num_iters=2, ddp_comm_hook=None, gradient_as_bucket_view=False
    ):
        """
        This is the recommended way to implement accumulate grads.
        If ``ddp_comm_hook`` input was specified, it will also register that hook
        to the ``ddp_model``. The hook fed into this function should not change
        the resulting gradients.
        """
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        global_batch_size = self.world_size
        local_batch_size = len(devices)

        model, ddp_model, input, target = self._prepare_single_device_module(
            process_group, devices, devices, global_batch_size, gradient_as_bucket_view
        )

        if ddp_comm_hook is not None:
            ddp_model.register_comm_hook(process_group, ddp_comm_hook)

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        # ensure accumulate grads works with no_grad
        with torch.no_grad():
            with ddp_model.no_sync():
                ddp_model.train()
                ddp_model(input)

        # check two model parameters over num_iters iterations
        for iteration in range(num_iters):
            # single cpu/gpu training
            step_model(model, input, target)

            ddp_input = input[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ]
            ddp_target = target[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ]

            if iteration % num_iters == 0:
                # accumulate grads locally
                with ddp_model.no_sync():
                    step_model(ddp_model, ddp_input, ddp_target)
            else:
                # sync grads
                step_model(ddp_model, ddp_input, ddp_target)

            for i, j in zip(model.parameters(), ddp_model.parameters()):
                if iteration % num_iters == 0:
                    self.assertNotEqual(i.grad, j.grad)
                else:
                    self.assertEqual(i.grad, j.grad)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    @requires_nccl()
    @skip_if_not_multigpu
    def test_accumulate_gradients_no_sync(self):
        """
        Runs _test_accumulate_gradients_no_sync using default inputs
        """
        self._test_accumulate_gradients_no_sync()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_accumulate_gradients_no_sync_grad_is_view(self):
        """
        Runs _test_accumulate_gradients_no_sync using default inputs
        """
        self._test_accumulate_gradients_no_sync(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_not_multigpu
    def test_accumulate_gradients_no_sync_allreduce_hook(self):
        """
        Runs multiple iterations on _test_accumulate_gradients_no_sync
        using allreduce hook and validates whether future result was properly
        passed as gradients in reducer.
        """

        def allreduce_hook(
            process_group: object, bucket: dist._GradBucket
        ) -> torch._C.Future:
            tensors = [t / self.world_size for t in bucket.get_tensors()]
            return process_group.allreduce(tensors).get_future()

        self._test_accumulate_gradients_no_sync(
            num_iters=4, ddp_comm_hook=allreduce_hook
        )

    @requires_nccl()
    @skip_if_not_multigpu
    def test_accumulate_gradients_no_sync_allreduce_with_then_hook(self):
        """
        Runs multiple iterations on _test_accumulate_gradients_no_sync using allreduce
        hook that also uses then callbacks. In first then callback result is multiplied
        by 2, and the second callback divides the result by 2 * world_size. It validates
        whether final result was properly passed as gradients in reducer.
        """

        def allreduce_with_then_hook(
            process_group: object, bucket: dist._GradBucket
        ) -> torch.futures.Future:
            fut = process_group.allreduce(bucket.get_tensors()).get_future()

            def mult(fut):
                # Multiply the result by 2.
                return [2 * t for t in fut.wait()]

            def div(fut):
                # Divide the result by 2 * world_size.
                return [t / (2 * self.world_size) for t in fut.wait()]

            return fut.then(mult).then(div)

        self._test_accumulate_gradients_no_sync(
            num_iters=4, ddp_comm_hook=allreduce_with_then_hook
        )

    def _test_accumulate_gradients_module(self, gradient_as_bucket_view=False):
        # This is NOT the recommended way to implement accumulating grads, but
        # we would like to make sure DDP does not mess up with the underlying
        # module.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        global_batch_size = self.world_size

        model, ddp_model, input, target = self._prepare_single_device_module(
            process_group, devices, devices, global_batch_size, gradient_as_bucket_view
        )

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        # ensure accumulate grads works with no_grad
        with torch.no_grad():
            ddp_model.train()
            ddp_model.module(input)

        # Check two model parameters over 4 iterations.
        # Use 4 iterations because we alternate between reducing and
        # not reducing and want to make sure we switch both ways.
        for iteration in range(4):
            step_model(model, input, target)

            if iteration % 2 == 0:
                # Skip gradients sync without calling prepare_for_backward
                step_model(
                    ddp_model.module,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertNotEqual(i.grad, j.grad)
            else:
                step_model(
                    ddp_model,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                    self.assertEqualIgnoreType(i.grad, j.grad)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    @requires_nccl()
    @skip_if_not_multigpu
    def test_accumulate_gradients_module(self):
        self._test_accumulate_gradients_module()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_accumulate_gradients_module_with_grad_is_view(self):
        self._test_accumulate_gradients_module(gradient_as_bucket_view=True)

    @requires_gloo()
    def test_ignored_output(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        class IgnoredOutput(nn.Module):
            def __init__(self):
                super(IgnoredOutput, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutput().float(),
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    def test_ignored_output_with_unused_parameters(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called, if not all model
        parameters participated in computing the model output.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        class IgnoredOutputWithUnusedParameters(nn.Module):
            def __init__(self):
                super(IgnoredOutputWithUnusedParameters, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutputWithUnusedParameters().float(),
            process_group=process_group,
            find_unused_parameters=True,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_failure_recovery(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # need to create a separate file for the recovered FileStore, because
        # the original one will be deleted when destructing the first FileStore.
        recovery_filename = self.file_name + "_recovery"

        if self.rank == 0:
            # the file will be deleted by the recovered FileStore
            open(recovery_filename, "w").close()

        # not necessary to run barrier here, as DDP will synchronize

        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = TestModel().float().to(device_id)
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

        del ddp
        del process_group
        del store  # this will delete self.file_name

        store = c10d.FileStore(recovery_filename, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_nccl()
    @skip_if_not_multigpu
    def test_pass_default_pg(self):
        dist.init_process_group(
            "nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        default_pg = c10d.distributed_c10d._get_default_group()
        dist.destroy_process_group(default_pg)
        self.assertFalse(dist.is_initialized())

    @requires_nccl()
    @skip_if_not_multigpu
    def test_save_load_checkpoint(self):
        dist.init_process_group(
            "gloo",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        def train_loop(model, optimizer, iterations):
            for _ in range(iterations):
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        device_id = gpus_for_rank(self.world_size)[self.rank][0]

        model_withload = TestModel().float().to(device_id)
        model_withoutload = TestModel().float().to(device_id)

        ddp_withload = DistributedDataParallel(
            model_withload,
            device_ids=[device_id],
        )
        ddp_withoutload = DistributedDataParallel(
            model_withoutload,
            device_ids=[device_id],
        )

        # ensure that both models start with the same set of parameters. By default they are randomized on construction
        for p in ddp_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in ddp_withoutload.parameters():
            with torch.no_grad():
                p.zero_()

        batch_size = 4
        criterion = nn.CrossEntropyLoss()

        optimizer_withload = torch.optim.SGD(ddp_withload.parameters(), lr=0.001)
        optimizer_withoutload = torch.optim.SGD(ddp_withoutload.parameters(), lr=0.001)

        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # run the model for 6 iterations, with a checkpoint in the middle
        train_loop(ddp_withload, optimizer_withload, 3)

        # zero out parameters and reload them from the state dict
        checkpoint_path = tempfile.gettempdir() + "/model.checkpoint"
        if self.rank == 0:
            torch.save(ddp_withload.state_dict(), checkpoint_path)

        dist.barrier()
        for p in ddp_withload.parameters():
            with torch.no_grad():
                p.zero_()
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
        ddp_withload.load_state_dict(
            torch.load(checkpoint_path, map_location=map_location)
        )

        train_loop(ddp_withload, optimizer_withload, 3)

        # re-run the model with the same inputs for 6 iterations with no checkpoint
        train_loop(ddp_withoutload, optimizer_withoutload, 6)

        for p_withload, p_withoutload in zip(
            ddp_withload.parameters(), ddp_withoutload.parameters()
        ):
            self.assertEqual(p_withload, p_withoutload)

    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        mult = 2
        batch_size = mult * self.world_size
        criterion = nn.CrossEntropyLoss()
        input = torch.randint(0, 10, [batch_size, 2])
        target = torch.randint(0, 10, [batch_size])

        # Run with entire batch against single process version
        criterion(vanilla_model(input), target).backward()

        # Run with partial batch against multi process version
        partial_input = input.split(mult)[self.rank]
        partial_target = target.split(mult)[self.rank]
        criterion(ddp_model(partial_input), partial_target).backward()

        # Check that the gradients are sparse and identical
        vanilla_parameter = next(vanilla_model.parameters())
        ddp_parameter = next(ddp_model.parameters())
        self.assertEqual(vanilla_parameter.grad, ddp_parameter.grad)

    def _test_sparse_gradients(self, gradient_as_bucket_view=False):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    @requires_gloo()
    def test_sparse_gradients(self):
        self._test_sparse_gradients()

    @requires_gloo()
    def test_sparse_gradients_grad_is_view(self):
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    def _test_grad_layout(self, replica_devices, layer_devs, local_batch_size):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        global_batch_size = local_batch_size * self.world_size

        # Carry out some trials with small buckets and some with big buckets.
        bucketsizes = (0.000001, 25)
        # Tuples of lists.  Each list describes per-layer characteristics for one trial.
        layer_formats = (
            [torch.contiguous_format] * 4,
            [torch.channels_last] * 2 + [torch.contiguous_format] * 2,
            [torch.channels_last] * 4,
        )
        layer_dtypes = (
            [torch.float] * 4,
            [torch.float] * 2 + [torch.half] * 2,
            [torch.half] * 4,
        )

        input_dev = layer_devs[0] if isinstance(layer_devs, list) else layer_devs
        target_dev = layer_devs[-1] if isinstance(layer_devs, list) else layer_devs
        input = torch.randn(
            (global_batch_size, 8, 8, 8), device=input_dev, dtype=torch.float
        )
        target = torch.randn(
            (global_batch_size, 8, 4, 4), device=target_dev, dtype=torch.float
        )
        local_batch_start = self.rank * local_batch_size
        local_batch_end = (self.rank + 1) * local_batch_size

        # Reducer.cpp sneakily creates one "initial bucket" that ignores the "bucket_cap_mb"
        # argument.  The following makes sure the initial bucket also complies.
        @contextmanager
        def first_bucket_size(ddp_bucket_mb):
            old_DEFAULT_FIRST_BUCKET_BYTES = dist._DEFAULT_FIRST_BUCKET_BYTES
            dist._DEFAULT_FIRST_BUCKET_BYTES = int(ddp_bucket_mb * 1.0e6)
            try:
                yield
            finally:
                dist._DEFAULT_FIRST_BUCKET_BYTES = old_DEFAULT_FIRST_BUCKET_BYTES

        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=False
        ):
            for formats, dtypes, bucketsize in product(
                layer_formats, layer_dtypes, bucketsizes
            ):
                with first_bucket_size(bucketsize):
                    model_msg = (
                        "rank = {} formats = {} dtypes = {} bucketsize = {} ".format(
                            self.rank, formats, dtypes, bucketsize
                        )
                    )
                    try:
                        m = ConvNet(layer_devs, formats, dtypes)
                        m_ddp = DistributedDataParallel(
                            copy.deepcopy(m),
                            device_ids=replica_devices,
                            process_group=process_group,
                            bucket_cap_mb=bucketsize,
                        )
                        opt = torch.optim.SGD(m.parameters(), lr=0.1)
                        opt_ddp = torch.optim.SGD(m_ddp.parameters(), lr=0.1)
                        has_half = any(p.dtype is torch.half for p in m.parameters())
                        tol = 1.0e-3 if has_half else 1.0e-5
                    except BaseException:
                        # Prints case-specific debugging info to narrow down failing case.
                        print(
                            "Caught exception during model creation for " + model_msg,
                            flush=True,
                        )
                        raise
                    # 3 iters:  First iter creates grads, second iter retests after rebucketing,
                    # third iter tries zeroed grads.
                    for it in range(3):
                        iter_msg = "iter = {} ".format(it) + model_msg
                        named_msg = iter_msg
                        try:
                            F.mse_loss(m(input).float(), target).backward()
                            F.mse_loss(
                                m_ddp(input[local_batch_start:local_batch_end]).float(),
                                target[local_batch_start:local_batch_end],
                            ).backward()
                            for i, ((layer_name, m_child), m_ddp_child) in enumerate(
                                zip(m.named_children(), m_ddp.module.children())
                            ):
                                named_msg = layer_name + ".weight" + " " + iter_msg
                                self.assertTrue(
                                    m_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                self.assertTrue(
                                    m_ddp_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                for j, ((param_name, p), p_ddp) in enumerate(
                                    zip(
                                        m_child.named_parameters(),
                                        m_ddp_child.parameters(),
                                    )
                                ):
                                    named_msg = (
                                        layer_name + "." + param_name + " " + iter_msg
                                    )
                                    self.assertEqual(
                                        p.grad, p_ddp.grad, rtol=tol, atol=tol
                                    )
                            opt.step()
                            opt_ddp.step()
                            if it == 0:
                                for p, p_ddp in zip(m.parameters(), m_ddp.parameters()):
                                    p.grad = None
                                    p_ddp.grad = None
                            else:
                                m.zero_grad()
                                m_ddp.zero_grad()
                        except BaseException:
                            # Makes sure we still get info if an error occurred somewhere other than the asserts.
                            print(
                                "Caught exception during iterations at " + named_msg,
                                flush=True,
                            )
                            raise

    @requires_nccl()
    @skip_if_not_multigpu
    def test_grad_layout_1devicemodule_1replicaperprocess(self):
        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        # Tells DDP to use just one device.
        replica_devices = [dev0]
        # Tells _test_grad_layout to construct ConvNet with all layers on this process's first assigned device.
        layer_devs = dev0
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @unittest.skipIf(
        True, "Re-enable when DDP with multiple GPUs per process is confirmed to work"
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_grad_layout_1devicemodule_2replicaperprocess(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device("cuda:" + str(int_devices[0]))
        dev1 = torch.device("cuda:" + str(int_devices[1]))
        # Tells DDP to replicate the model to both of this process's devices.
        replica_devices = [dev0, dev1]
        # Tells _test_grad_layout to construct ConvNet with all layers on this process's first assigned device.
        layer_devs = dev0
        local_batch_size = 16
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_if_rocm
    def test_grad_layout_2devicemodule(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device("cuda:" + str(int_devices[0]))
        dev1 = torch.device("cuda:" + str(int_devices[1]))
        # DDP's default behavior for a multi-device module is "don't replicate."
        replica_devices = None
        # Tells _test_grad_layout to constructs this process's ConvNet on 2 devices, with 2 layers on each device.
        layer_devs = [dev0] * 2 + [dev1] * 2
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_not_multigpu
    def test_param_layout_mismatch_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        layer_devs = dev0
        layer_formats = (
            [torch.contiguous_format] * 4
            if self.rank == 0
            else [torch.channels_last] * 4
        )
        layer_dtypes = [torch.float] * 4

        m = ConvNet(layer_devs, layer_formats, layer_dtypes)
        if self.rank == 0:
            m_ddp = DistributedDataParallel(
                m, device_ids=[dev0], process_group=process_group
            )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                ".* appears not to match strides of the same param in process 0",
            ):
                m_ddp = DistributedDataParallel(
                    m, device_ids=[dev0], process_group=process_group
                )

    @requires_gloo()
    def test_ddp_comm_hook_future_passing_cpu(self):
        """
        This unit test verifies whether the Future object is passed properly.
        The callback function creates a Future object and sets a value to it.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Test on CPU
        cpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().cpu(), process_group=process_group
        )

        # Register DDP Communication Hook
        cpu_model.register_comm_hook(None, self._simple_hook)

        # check whether the grads are equal to what then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    def _gpu_model_with_builtin_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a built-in DDP communication hook if defined
        if hook is not None:
            gpu_model._register_builtin_comm_hook(hook)

        return gpu_model

    def _run_and_verify_hook(self, model, input, expected_grad):
        # Run forward
        output = model(input, self.rank)

        # Run backward
        output.mean().backward()

        [self.assertEqual(p.grad, expected_grad) for p in model.parameters()]

    def _simple_hook(
        self, state: object, bucket: dist._GradBucket
    ) -> torch.futures.Future:
        fut = torch.futures.Future()
        fut.set_result([torch.ones_like(t) for t in bucket.get_tensors()])

        def fut_then(fut):
            # Add ones to fut's result.
            return [t + torch.ones_like(t) for t in fut.value()]

        return fut.then(fut_then)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_gloo(self):
        """
        This unit test verifies whether the Future object is passed properly using gloo backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_nccl(self):
        """
        This unit test verifies whether the Future object is passed properly using nccl backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    def _test_ddp_comm_hook_allreduce_hook_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether a DDP communication hook that just calls
        allreduce gives the same result with the case of no hook registered.
        Without the then callback, the future_value in reducer is no longer
        a PyObject, and this unit test verifies future_value is properly checked.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce_hook(state: object, bucket: dist._GradBucket) -> torch._C.Future:
            tensors = [t / self.world_size for t in bucket.get_tensors()]
            return process_group.allreduce(tensors).get_future()

        # Get GPU model with allreduce_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_hook, gradient_as_bucket_view
        )

        # check whether the grads are equal to what DDP without hook would return.
        self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_default_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether default Python DDP communication hooks ALLREDUCE and FP16_COMPRESS
        can give the same result with the case of no hook registered.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # For these default DDP comm hooks, the only state is process group.
        state = process_group
        for hook in [default.allreduce_hook, default.fp16_compress_hook]:
            # Get GPU model with the hook registered.
            # The first arg 'process_group' is used for initializing the test environment,
            # so it cannot be replaced by 'state', although they have the same value.
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group, hook, gradient_as_bucket_view, state
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_powerSGD_ddp_comm_hook_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether Python DDP communication hook POWER_SGD
        can give the same result with the case of no hook registered.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        # Get GPU model with the hook registered.
        # Test the hook with different algorithmic configs.
        for use_error_feedback, warm_start in product([True, False], [True, False]):
            state = powerSGD.PowerSGDState(
                process_group=process_group,
                matrix_approximation_rank=1,
                use_error_feedback=use_error_feedback,
                warm_start=warm_start,
            )
            for hook in [powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook]:
                gpu_model = self._gpu_model_with_ddp_comm_hook(
                    process_group, hook, gradient_as_bucket_view, state
                )

                # check whether the grads are equal to what DDP without hook would return.
                self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_builtin_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether built-in C++ DDP communication hooks ALLREDUCE and FP16_COMPRESS
        can give the same result with the case of no hook registered.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        for comm_hook_type in [
            dist.BuiltinCommHookType.ALLREDUCE,
            dist.BuiltinCommHookType.FP16_COMPRESS,
        ]:
            # Get GPU model with the built-in communication hook.
            gpu_model = self._gpu_model_with_builtin_ddp_comm_hook(
                process_group, comm_hook_type, gradient_as_bucket_view
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl(self):
        self._test_default_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl(self):
        self._test_builtin_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl(self):
        self._test_powerSGD_ddp_comm_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_grad_is_view(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl(gradient_as_bucket_view=True)

    def test_invalid_powerSGD_state(self):
        for start_powerSGD_iter, use_error_feedback, warm_start in product([0, 1], [True, False], [True, False]):
            if not use_error_feedback and not warm_start:
                continue
            with self.assertRaisesRegex(
                    ValueError,
                    "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                    "because PowerSGD can only be applied after the first two iterations in DDP."):
                state = powerSGD.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=1,
                    start_powerSGD_iter=start_powerSGD_iter,
                    use_error_feedback=use_error_feedback,
                    warm_start=warm_start,
                )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl_is_view(self):
        self._test_default_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl_grad_is_view(self):
        self._test_builtin_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl_grad_is_view(self):
        self._test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_then_hook_nccl(self):
        """
        This unit test verifies whether a DDP communication hook that calls allreduce and then
        multiplies the result by ten and divides by two gives the expected result.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

        def allreduce_with_then_hook(
            state: object, bucket: dist._GradBucket
        ) -> torch.futures.Future:
            tensors = [t / self.world_size for t in bucket.get_tensors()]
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                # Multiply the result by 10.
                return [10 * t for t in fut.value()]

            def div(fut):
                # Divide the result by 2.
                return [0.5 * t for t in fut.value()]

            return fut.then(mult).then(div)

        # Get GPU model with allreduce_with_then_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_with_then_hook
        )

        # check whether the grads are equal to what allreduce returns multuplied by 5.
        # without the comm_hook, result would be still 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))

    @requires_gloo()
    def test_ddp_invalid_comm_hook_init(self):
        """
        This unit test makes sure that register_comm_hook properly checks the format
        of hook defined by user. The Python hook must be callable. This test also
        checks whether bucket annotation checked properly if defined.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        with self.assertRaisesRegex(TypeError, "Communication hook must be callable."):
            model.register_comm_hook(state=None, hook=1)

        with self.assertRaisesRegex(
            ValueError, "bucket annotation should be dist._GradBucket."
        ):

            def comm_hook(state: object, bucket: int) -> torch.futures.Future:
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_gloo()
    def test_ddp_invalid_comm_hook_return_type(self):
        """
        This test checks whether return annotation checked properly if defined. It also
        checks whether an internal error is thrown if return type is incorrect and user
        hasn't specified any return type annotation.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        with self.assertRaisesRegex(
            ValueError,
            "Communication hook: return annotation should be torch.futures.Future or torch._C.Future.",
        ):

            def comm_hook(state: object, bucket: dist._GradBucket) -> int:
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "callback must return a torch.futures.Future or torch._C.Future object, but got",
        ):

            def comm_hook(state: object, bucket: dist._GradBucket):
                return 1

            model.register_comm_hook(state=None, hook=comm_hook)

            # Run forward
            output = model(8, self.rank)

            # Run backward
            output.mean().backward()

    @requires_gloo()
    def test_ddp_comm_hook_register_just_once(self):
        """
        DDP communication hook can only be registered once. This test validates whether
        the error is thrown properly when register_comm_hook is called more than once.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result(bucket.get_tensors())
            return fut

        model.register_comm_hook(None, dummy_hook)

        with self.assertRaisesRegex(
            RuntimeError,
            "register_comm_hook or register_builtin_comm_hook can only be called once.",
        ):
            model.register_comm_hook(None, dummy_hook)

    @requires_gloo()
    def test_ddp_comm_hook_sparse_gradients(self):
        """
        Runs "test_sparse_gradients" unit test with DDP communication hook. We define a
        simple hook that does allreduce and works with gloo backend for this test.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
        )

        # "get_future" API does not support gloo backend, see GH Issue #42048.
        # Instead, we wait for an allreduce work, and write its result to a Future.
        def allreduce_hook_gloo(
            state: object, bucket: dist._GradBucket
        ) -> torch.futures.Future:
            # Prepare allreduced grad bucket tensors by running an async work.
            work = process_group.allreduce(bucket.get_tensors())
            work.wait()

            fut = torch.futures.Future()
            fut.set_result([t / self.world_size for t in bucket.get_tensors()])
            return fut

        ddp_model.register_comm_hook(None, allreduce_hook_gloo)

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)


class ReducerModule(nn.Module):
    def __init__(self):
        super(ReducerModule, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, use_fc3=True):
        x = self.relu(self.fc1(x)).float()
        x = self.relu(self.fc2(x)).float()
        if use_fc3:
            x = self.fc3(x).float()
        return F.softmax(x, dim=1)


@requires_gloo()
class ReducerTest(TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.store = c10d.FileStore(self.file.name, 1)
        self.process_group = c10d.ProcessGroupGloo(self.store, 0, 1)

    def test_single_dtype_single_bucket(self):
        model = ReducerModule()
        parameters = list(model.parameters())
        buckets = [list(range(len(parameters)))]
        dist.Reducer([parameters], buckets, self.process_group)

    def _create_mixed_precision_model(self):
        model = ReducerModule()
        model.float()
        model.fc1.double()
        return model

    def test_multi_dtype_single_bucket(self):
        model = self._create_mixed_precision_model()

        # Raise if there are multiple types per bucket.
        # In this case we create one bucket for all parameters.
        with self.assertRaises(RuntimeError):
            parameters = [list(model.parameters())]
            buckets = [list(range(len(parameters[0])))]
            dist.Reducer(parameters, buckets, self.process_group)

    def test_multi_dtype_multi_bucket(self):
        model = self._create_mixed_precision_model()
        parameters = [list(model.parameters())]
        group_by_dtype = groupby(
            range(len(parameters[0])), key=lambda i: parameters[0][i].dtype
        )
        buckets = [list(indices) for _, indices in group_by_dtype]
        dist.Reducer(parameters, buckets, self.process_group)

    def _create_reducer_for_models(self, models, find_unused_parameters=False):
        parameters = [list(model.parameters()) for model in models]
        group_by_dtype = groupby(
            range(len(parameters[0])), key=lambda i: parameters[0][i].dtype
        )
        buckets = [list(indices) for _, indices in group_by_dtype]
        return dist.Reducer(
            parameters,
            buckets,
            self.process_group,
            find_unused_parameters=find_unused_parameters,
        )

    def test_forward_backward_single_replica(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model])
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input), target)
        reducer.prepare_for_backward(output)
        output.backward()

    def test_forward_backward_multi_replica(self):
        batch_size = 10
        num_replicas = 2
        models = [self._create_mixed_precision_model() for _ in range(num_replicas)]
        reducer = self._create_reducer_for_models(models)
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double).chunk(num_replicas)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        outputs = [models[i](input[i]) for i in range(num_replicas)]
        output = loss(torch.cat(outputs), target)
        reducer.prepare_for_backward(output)
        output.backward()

        # The reducer will have reduced the gradients for all model replicas.
        # Verify that they are equal across model replicas.
        for parameters in zip(*[model.parameters() for model in models]):
            for parameter in parameters:
                self.assertEqual(parameters[0].grad, parameter.grad)

    def test_forward_backward_unused_parameters(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input, use_fc3=False), target)

        # Check that the grad of fc3 is not set.
        self.assertEqual(None, model.fc3.weight.grad)

        # Compute and accumulate gradients.
        reducer.prepare_for_backward(output)
        output.backward()

        # The reducer will have marked the grad of fc3 as ready, because
        # it doesn't show up in the autograd graph of `output`. Since fc3.weight
        # is considered being globally unused, it will be kept untouched as None.
        self.assertEqual(None, model.fc3.weight.grad)

    def test_forward_backward_optimizer(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for i in range(3):
            input = torch.rand([batch_size, 2], dtype=torch.double)
            target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

            # The `zero_grad` function calls `detach_` and `zero_` on the grad
            # tensors of model parameters. If we tried to set the grad tensors
            # to a view of the reducer's bucket tensors, this would blow up.
            optimizer.zero_grad()

            # Unused parameter only in the first iteration.
            output = loss(model(input, use_fc3=(i > 0)), target)
            reducer.prepare_for_backward(output)
            output.backward()
            optimizer.step()

    def test_ddp_comm_hook_multiple_replica_check(self):
        """
        DDP communication hook does not support single process multiple device mode.
        This unit test validates this condition is properly checked by reducer.
        Related to GH Issue #42542.
        """
        num_replicas = 2
        models = [self._create_mixed_precision_model() for _ in range(num_replicas)]
        reducer = self._create_reducer_for_models(models)

        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result(bucket.get_tensors())
            return fut

        with self.assertRaisesRegex(
            RuntimeError,
            "Communication hook does not support single-process multiple-device mode.",
        ):
            dist._register_comm_hook(reducer, None, dummy_hook)


class ComputeBucketAssignmentTest(TestCase):
    def test_single_limit_single_dtype(self):
        tensors = [
            torch.empty([100], dtype=torch.float),
            torch.empty([200], dtype=torch.float),
            torch.empty([100], dtype=torch.float),
            torch.empty([50], dtype=torch.float),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        self.assertEqual([[0], [1], [2], [3]], result)

    def test_single_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], result)

    def test_multi_limit_single_dtype(self):
        tensors = [
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [40, 80])
        self.assertEqual([[0], [1, 2], [3]], result)

    def test_multi_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5]], result)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class NcclErrorHandlingTest(MultiProcessTestCase):
    def setUp(self):
        super(NcclErrorHandlingTest, self).setUp()
        # Need to skip return code checking for these tests since the child
        # processes don't exit cleanly.
        self.skip_return_code_checks = [
            self.test_nccl_errors_blocking_abort.__wrapped__,
            self.test_nccl_errors_blocking_sigkill.__wrapped__,
            self.test_nccl_errors_blocking_sigterm.__wrapped__,
            self.test_nccl_errors_blocking_nonzero_exit.__wrapped__,
        ]
        self._fork_processes()

    def tearDown(self):
        super(NcclErrorHandlingTest, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return 3

    @property
    def blocking_wait_error_msg(self):
        return "Caught collective operation timeout"

    def _run_all_reduce(self, pg):
        pg.allreduce(torch.rand(10).cuda(self.rank))

    @requires_nccl()
    @requires_nccl_version(2400, "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_errors_nonblocking(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            # This allreduce does not block Python thread as allreduce enqueues
            # the cuda operation, and then wait only blocks the current cuda
            # stream.
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()

            # Now the work scheduled next should hang forever since the previous
            # allreduce will never complete.
            t = threading.Thread(target=self._run_all_reduce, args=(process_group,))
            t.daemon = True
            t.start()
            t.join(int(get_timeout(self.id()) / 5))
            self.assertTrue(t.is_alive())

    def _test_nccl_errors_blocking(self, func):
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=self.op_timeout_sec),
        )
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            with self.assertRaisesRegex(RuntimeError, self.blocking_wait_error_msg):
                # Operation would time out in blocking mode.
                work.wait()
            # Run some GPU operations to make sure cuda has not gotten stuck.
            # It was observed cuda could get stuck if NCCL communicators were
            # not properly aborted before throwing RuntimeError.
            a = torch.rand(10).cuda(self.rank)
        elif self.rank == 1:
            # Clean up structures (ex: files for FileStore before going down)
            del process_group
            func()
        else:
            # Wait for timeout
            time.sleep(2 * self.op_timeout_sec)

            # Now verify communicators on this rank have been aborted by the watchdog thread.
            self._wait_for_comm_abort(process_group)

    @requires_nccl()
    @requires_nccl_version(2400, "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_errors_blocking_clean_exit(self):
        self._test_nccl_errors_blocking(lambda: sys.exit(0))

    @requires_nccl()
    @requires_nccl_version(2400, "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_errors_blocking_nonzero_exit(self):
        self._test_nccl_errors_blocking(lambda: sys.exit(1))

    @requires_nccl()
    @requires_nccl_version(2400, "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_errors_blocking_abort(self):
        self._test_nccl_errors_blocking(lambda: os.abort())

    @requires_nccl()
    @requires_nccl_version(2400, "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_errors_blocking_sigkill(self):
        self._test_nccl_errors_blocking(lambda: os.kill(os.getpid(), signal.SIGKILL))

    @requires_nccl()
    @requires_nccl_version(2400, "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_errors_blocking_sigterm(self):
        self._test_nccl_errors_blocking(lambda: os.kill(os.getpid(), signal.SIGTERM))

    @requires_nccl()
    @requires_nccl_version(2400, "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_blocking_wait_with_barrier(self):
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=self.op_timeout_sec),
        )
        process_group.barrier().wait()
        if self.rank == 0:
            with self.assertRaisesRegex(RuntimeError, self.blocking_wait_error_msg):
                # This should timeout
                process_group.barrier().wait()

    def _run_invalid_nccl_blocking_wait_env(self, val):
        os.environ["NCCL_BLOCKING_WAIT"] = val
        store = c10d.FileStore(self.file_name, self.world_size)
        with self.assertRaises(RuntimeError):
            process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_invalid_nccl_blocking_wait_env(self):
        self._run_invalid_nccl_blocking_wait_env("abc")
        self._run_invalid_nccl_blocking_wait_env("-1")
        self._run_invalid_nccl_blocking_wait_env("2147483647")
        self._run_invalid_nccl_blocking_wait_env("4294967295")

    def _wait_for_comm_abort(self, process_group):
        """
        Waits for the watchdog thread to abort communicators for the process group.
        """
        while True:
            try:
                process_group.allreduce(torch.rand(10).cuda(self.rank))
            except Exception as e:
                if "NCCL communicator was aborted" in str(e):
                    return
                else:
                    raise e
            time.sleep(1)

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_nccl_timeout(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

        # Initialize process_group.
        timeout = 1
        process_group = c10d.ProcessGroupNCCL(
            store, self.rank, self.world_size, timeout=timedelta(seconds=timeout)
        )
        process_group.allreduce(torch.rand(10).cuda(self.rank)).wait()

        if self.rank == 0:
            # This should timeout in about 1 second.
            start = time.time()
            # Watchdog may abort timed out work resulting in NCCL error instead of operation timed out.
            with self.assertRaisesRegex(RuntimeError, self.blocking_wait_error_msg):
                process_group.allreduce(torch.rand(10).cuda(self.rank)).wait()
        else:
            # Sleep to ensure timeout.
            time.sleep(2 * timeout)

            self._wait_for_comm_abort(process_group)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class CommTest(MultiProcessTestCase):
    def setUp(self):
        super(CommTest, self).setUp()
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def tearDown(self):
        super(CommTest, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return 2

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        half = torch.float16

        # No support for float16 for CPU tensors
        if device == torch.device("cpu"):
            half = torch.float32

        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # The tensors to pass to broadcast are idential to the target
        # only on the process that is the root of the broadcast.
        if self.rank == root_rank:
            tensors = list(tensor.clone() for tensor in target)
        else:
            tensors = list(torch.zeros_like(tensor) for tensor in target)

        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_nccl()
    @skip_if_not_multigpu
    def test_broadcast_coalesced_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        device = torch.device("cuda:%d" % self.rank)
        ranks = [0, 1]
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    @skip_if_not_multigpu
    def test_broadcast_coalesced_gloo_cuda(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [create_device(interface=LOOPBACK)]
        process_group = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, options
        )
        device = torch.device("cuda:%d" % self.rank)
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    def test_broadcast_coalesced_gloo_cpu(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        options = c10d.ProcessGroupGloo.Options()
        options.devices = [create_device(interface=LOOPBACK)]
        process_group = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, options
        )
        device = torch.device("cpu")
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store)

        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        c10d.all_reduce(t)
        expected_tensor = torch.tensor([3] * 10).cuda(2 * self.rank)
        self.assertEqual(expected_tensor, t)

        # Test with new_group
        pg = c10d.new_group([0, 1])
        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        pg.allreduce(t).wait()
        self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([0])
        if self.rank == 0:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([1])
        if self.rank == 1:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier_timeout(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        if self.rank == 0:
            with self.assertRaisesRegex(RuntimeError, "Timed out initializing process group"):
                c10d.init_process_group(
                    backend="nccl",
                    rank=self.rank,
                    world_size=self.world_size,
                    store=store,
                    timeout=timedelta(seconds=1))

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier_timeout_new_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            timeout=timedelta(seconds=1))

        if self.rank == 0:
            with self.assertRaisesRegex(RuntimeError, "Timed out initializing process group"):
                c10d.new_group([0, 1], timeout=timedelta(seconds=1))

            with self.assertRaisesRegex(RuntimeError, "Timed out initializing process group"):
                c10d.new_group([0], timeout=timedelta(seconds=1))

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier_timeout_new_group_non_member(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            timeout=timedelta(seconds=1))

        if self.rank == 1:
            with self.assertRaisesRegex(RuntimeError, "Timed out initializing process group"):
                c10d.new_group([0, 1], timeout=timedelta(seconds=1))

            with self.assertRaisesRegex(RuntimeError, "Timed out initializing process group"):
                c10d.new_group([0], timeout=timedelta(seconds=1))

    @requires_nccl()
    @skip_if_not_multigpu
    def test_nccl_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store)

        c10d.barrier(device_ids=[self.rank])

    @requires_nccl()
    @skip_if_not_multigpu
    def test_nccl_barrier_device_ids_function_argument(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store)

        with self.assertRaisesRegex(RuntimeError, "Invalid function argument"):
            c10d.barrier(device_ids=self.rank)

    @requires_gloo()
    def test_gloo_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store)

        with self.assertRaisesRegex(RuntimeError, "device_ids not supported"):
            c10d.barrier(device_ids=[self.rank])

if __name__ == '__main__':
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
