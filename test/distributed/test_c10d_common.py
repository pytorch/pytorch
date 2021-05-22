import copy
import os
import random
import sys
import tempfile
import threading
import time
import traceback
import unittest
from datetime import timedelta
from itertools import product
from sys import platform

import torch
import torch.distributed as c10d

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch._six import string_classes
from torch.nn.parallel import DistributedDataParallel
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
    TEST_WITH_TSAN,
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


class AbstractTimeoutTest(object):
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


class AbstractProcessGroupWrapperTest(MultiProcessTestCase):
    def setUp(self):
        super(AbstractProcessGroupWrapperTest, self).setUp()
        # For Windows platform, Python does not support fork, change it to spawn here.
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def _test_collective_hang(self, wrapper_pg, use_cuda=False):
        # All ranks besides 1 call allreduce and wrapper_pg should detect a hang
        # and report an issue with rank 1.
        faulty_rank = 1
        if self.rank != faulty_rank:
            tensor = torch.randn(20, 10)
            if use_cuda:
                tensor = tensor.to(self.rank)

            if self.rank == 0:
                # Rank 0 reports faulty ranks
                err = f"Ranks {faulty_rank} failed to pass monitoredBarrier"
            else:
                err = "Please check rank 0 logs for faulty rank"
            with self.assertRaisesRegex(RuntimeError, err):
                wrapper_pg.allreduce([tensor])

    def _test_collectives_op_mismatch(self, wrapper_pg, use_cuda=False):
        tensor = torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        works = []
        # Run a few successful collectives
        for _ in range(10):
            work = wrapper_pg.allreduce([tensor])
            works.append(work)

        for w in works:
            w.wait()

        # Simulate mismatch: allreduce vs reduce.
        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            if self.rank == 0:
                wrapper_pg.allreduce([tensor])
            else:
                wrapper_pg.reduce([tensor])

        # Check additional mismatches

        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            if self.rank == 0:
                wrapper_pg.reduce([tensor])
            else:
                wrapper_pg.barrier()

        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            scatter_result = [torch.ones(4) * i for i in range(self.world_size)]
            scattered_tensor = torch.empty(4)
            if self.rank == 0:
                wrapper_pg.scatter(scattered_tensor, scatter_result, 0)
            else:
                wrapper_pg.reduce_scatter(scattered_tensor, scatter_result)

        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            if self.rank == 0:
                wrapper_pg.broadcast(tensor, 0)
            else:
                output_tensors = [
                    torch.zeros_like(tensor) for _ in range(self.world_size)
                ]
                wrapper_pg.allgather([output_tensors], [tensor])

    def _test_collective_shape_mismatch(self, wrapper_pg, use_cuda=False):
        wrapper_pg.barrier()
        dim = 2 if self.rank == 0 else 10
        tensor = torch.randn(20, dim)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, "Error when verifying shape tensors"):
            wrapper_pg.allreduce([tensor])
        # Check errors are raised when dimensionality of shapes is different
        tensor = torch.randn(20, 10, 2) if self.rank == 0 else torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, "Error when verifying shape tensors"):
            wrapper_pg.allreduce([tensor])

        # Check shape errors with scatter
        input = [
            torch.tensor(
                [self.rank] if self.rank == 0 else [self.rank, self.rank],
                device=self.rank if use_cuda else "cpu",
            )
            for _ in range(self.world_size)
        ]
        outputs = [
            torch.tensor(
                [-1] if self.rank == 0 else [-1, -1],
                device=self.rank if use_cuda else "cpu",
            )
            for _ in range(self.world_size)
        ]
        root_rank = 0
        opts = c10d.ScatterOptions()
        opts.rootRank = root_rank
        with self.assertRaisesRegex(RuntimeError, "Error when verifying shape tensors"):
            if self.rank == root_rank:
                wrapper_pg.scatter([outputs[self.rank]], [input], opts).wait()
            else:
                wrapper_pg.scatter([outputs[self.rank]], [], opts).wait()

class AbstractDistributedDataParallelTest(object):
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
        device = devices[0] if devices else torch.device("cuda:%d" % self.rank)
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).to(device),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        model.to(device)

        input = torch.randn(global_batch_size, 2).to(device)
        target = torch.randn(global_batch_size, 4).to(device)

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
        local_batch_size = 1 if devices is None else len(devices)
        global_batch_size = self.world_size * local_batch_size

        if multi_device:
            model, ddp_model, input, target = self._prepare_multi_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertTrue(ddp_logging_data.get("is_multi_device_module"))
        else:
            model, ddp_model, input, target = self._prepare_single_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertFalse(ddp_logging_data.get("is_multi_device_module"))

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
                    self.rank * local_batch_size: (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size: (self.rank + 1) * local_batch_size
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
            self, state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future:
        fut = torch.futures.Future()
        fut.set_result([torch.ones_like(bucket.get_tensor())])

        def fut_then(fut):
            # Add ones to fut's result.
            return [t + torch.ones_like(t) for t in fut.value()]

        return fut.then(fut_then)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class DistributedDataParallelTest(AbstractDistributedDataParallelTest, MultiProcessTestCase):

    def setUp(self):
        super(DistributedDataParallelTest, self).setUp()
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def test_invalid_powerSGD_state(self):
        for start_powerSGD_iter, use_error_feedback, warm_start in product(
                [0, 1], [True, False], [True, False]
        ):
            if not use_error_feedback and not warm_start:
                continue
            with self.assertRaisesRegex(
                    ValueError,
                    "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                    "because PowerSGD can only be applied after the first two iterations in DDP.",
            ):
                state = powerSGD.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=1,
                    start_powerSGD_iter=start_powerSGD_iter,
                    use_error_feedback=use_error_feedback,
                    warm_start=warm_start,
                )


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


class AbstractCommTest(object):

    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return 2

    def _verify_sequence_number_across_pg(self, pg, verify_pg):

        seq_num = pg._get_sequence_number_for_group()
        obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
        # We use a separate pg to verify the sequence numbers, otherwise these
        # collectives will themselves increment the sequence number.
        dist.all_gather_object(obj_list, seq_num, group=verify_pg)
        self.assertEqual(len(set(obj_list)), 1)
        return obj_list[0]

    def _test_sequence_num_incremented(self, process_group, ranks):
        # verify initial sequence numbers. Use a distinct process group for
        # verification to keep counts as expected with respect to process_group.
        verify_pg = dist.new_group(
            ranks=ranks,
            backend="gloo",
        )
        assert dist.get_world_size(process_group) == dist.get_world_size(verify_pg)

        initial_num = (
            self._verify_sequence_number_across_pg(
                pg=process_group, verify_pg=verify_pg
            )
            if not c10d.distributed_c10d._rank_not_in_group(process_group)
            else -1
        )

        # Verify sequence numbers are appropriately incremented
        for i in range(10):
            t = torch.ones(1, device=torch.cuda.current_device())
            dist.all_reduce(t, group=process_group)
            if not c10d.distributed_c10d._rank_not_in_group(process_group):
                seq_num = self._verify_sequence_number_across_pg(
                    pg=process_group,
                    verify_pg=verify_pg,
                )
                self.assertEqual(initial_num + i + 1, seq_num)

        if dist.get_world_size(process_group) > 2:
            # Test when certain ranks don't call collectives
            if dist.get_rank(process_group) not in [0, 2]:
                dist.all_reduce(t, group=process_group, async_op=True)
            # Now ranks 0 and 2 should be lagging by 1.
            if not c10d.distributed_c10d._rank_not_in_group(process_group):
                seq_num = process_group._get_sequence_number_for_group()
                rank = dist.get_rank(process_group)
                obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
                dist.all_gather_object(obj_list, (rank, seq_num), group=verify_pg)
                rank_to_seq_num = {rank: num for (rank, num) in obj_list}
                self.assertEqual(len(set(rank_to_seq_num.values())), 2)
                self.assertEqual(rank_to_seq_num[0], rank_to_seq_num[2])
                expected_same = {
                    rank_to_seq_num[i]
                    for i in rank_to_seq_num.keys()
                    if i not in [0, 2]
                }
                self.assertEqual(len(expected_same), 1)
                self.assertEqual(rank_to_seq_num[0] + 1, rank_to_seq_num[1])

    def _test_sequence_num_incremented_default_group(self, backend_name):
        torch.cuda.set_device(self.rank)
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        self._test_sequence_num_incremented(
            c10d.distributed_c10d._get_default_group(),
            ranks=list(i for i in range(dist.get_world_size())),
        )

    def _test_sequence_num_incremented_subgroup(self, backend_name):
        torch.cuda.set_device(self.rank)
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        subgroup_ranks = [0, 1, 2]
        subgroup = dist.new_group(subgroup_ranks)
        self._test_sequence_num_incremented(subgroup, subgroup_ranks)

    def _test_sequence_num_set_default_pg(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        default_pg = c10d.distributed_c10d._get_default_group()
        seq_num = default_pg._get_sequence_number_for_group()
        obj_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(obj_list, seq_num)
        self.assertEqual(len(set(obj_list)), 1)

    def _test_sequence_num_set_new_group(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        subgroup = dist.new_group([0, 1])

        if not c10d.distributed_c10d._rank_not_in_group(subgroup):
            subgroup_seq = subgroup._get_sequence_number_for_group()
            obj_list = [None for _ in range(dist.get_world_size(subgroup))]
            dist.all_gather_object(obj_list, subgroup_seq, group=subgroup)
            self.assertEqual(len(set(obj_list)), 1)


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class CommTest(AbstractCommTest, MultiProcessTestCase):

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

    def test_distributed_debug_mode(self):
        # Default should be off
        default_debug_mode = dist._get_debug_mode()
        self.assertEqual(default_debug_mode, dist._DistributedDebugLevel.OFF)
        mapping = {
            "OFF": dist._DistributedDebugLevel.OFF,
            "INFO": dist._DistributedDebugLevel.INFO,
            "DETAIL": dist._DistributedDebugLevel.DETAIL,
        }
        invalid_debug_modes = ["foo", 0, 1, -1]

        for mode in mapping.keys():
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = str(mode)
            set_debug_mode = dist._get_debug_mode()
            self.assertEqual(
                set_debug_mode,
                mapping[mode],
                f"Expected {mode} to map to {mapping[mode]} but got {set_debug_mode}",
            )

        for mode in invalid_debug_modes:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = str(mode)
            with self.assertRaisesRegex(RuntimeError, "to be one of"):
                dist._get_debug_mode()


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
