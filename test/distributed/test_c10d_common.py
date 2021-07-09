import copy
import os
import sys
import tempfile
import threading
import time
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
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import (
    TestCase,
    load_tests,
    run_tests,
    TEST_WITH_TSAN,
)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"

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
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

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
