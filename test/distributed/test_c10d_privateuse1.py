# Owner(s): ["oncall: distributed"]

import copy
import os
import random
import sys
from itertools import product

import torch
import torch.distributed as c10d
import torch.distributed as dist
import torch.distributed._functional_collectives as _functional_collectives
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_ROCM,
)


if torch.accelerator.current_accelerator() is None:
    print("No accelerator available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

import test_c10d_common
from test_c10d_common import gpus_for_rank, ModuleForDdpCommHook


DEVICE_TYPE = torch.accelerator.current_accelerator().type
BACKEND = dist.get_default_backend_for_device(DEVICE_TYPE)
device_module = torch.get_device_module(DEVICE_TYPE)
BFLOAT16_AVAILABLE = hasattr(device_module, "is_bf16_supported") and (
    device_module.is_bf16_supported()
)


class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        if DEVICE_TYPE == "cuda" and c10d.is_nccl_available():
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _get_process_group(self):
        store = self._get_store()
        c10d.init_process_group(
            BACKEND, store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_backend_single_device_module_device_ids_None(self):
        self._test_backend(None, None)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_backend_single_device_module_empty_device_ids(self):
        # This tests the backward compatibility of accepting an empty list as `device_ids`,
        # although we no longer document this in favor of the default value of `None`,
        # which is consistent with multi-device modules and CPU modules.
        self._test_backend(None, [])

    def _test_fp16(self, gradient_as_bucket_view=False):
        process_group = self._get_process_group()

        dev_ids = gpus_for_rank(self.world_size)[self.rank]
        device = torch.device(DEVICE_TYPE, dev_ids[0])
        model = nn.Linear(1, 1, bias=False).to(device).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[dev_ids[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.tensor([[2**15]]).to(device).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16(self):
        self._test_fp16()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16_grad_is_view(self):
        self._test_fp16(gradient_as_bucket_view=True)

    def _test_arbitrary_forward_return_value(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class ForwardReturnValueModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
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

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value(self):
        self._test_arbitrary_forward_return_value()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_with_lazy_parameters(self):
        process_group = self._get_process_group()
        with self.assertRaisesRegex(
            RuntimeError, "Modules with uninitialized parameters"
        ):
            DistributedDataParallel(
                torch.nn.LazyLinear(10), process_group=process_group
            )

    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class MultipleOutputModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()

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

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        self._test_multiple_outputs_multiple_backward()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_no_grad(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class NoGradModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
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

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_failure_recovery(self):
        process_group = self._get_process_group()

        # need to create a separate file for the recovered FileStore, because
        # the original one will be deleted when destructing the first FileStore.
        recovery_filename = self.file_name + "_recovery"

        if self.rank == 0:
            # the file will be deleted by the recovered FileStore
            open(recovery_filename, "w").close()

        # not necessary to run barrier here, as DDP will synchronize

        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
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
        c10d.destroy_process_group(process_group)

        store = c10d.FileStore(recovery_filename, self.world_size)
        c10d.init_process_group(
            BACKEND, store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
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

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_pass_default_pg(self):
        dist.init_process_group(
            BACKEND,
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        default_pg = c10d.distributed_c10d._get_default_group()
        dist.destroy_process_group(default_pg)
        self.assertFalse(dist.is_initialized())

    def _gpu_model_with_ddp_comm_hook(
        self,
        process_group,
        hook=None,
        gradient_as_bucket_view=False,
        state=None,
        static_graph=False,
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing(self):
        """
        This unit test verifies whether the Future object is passed properly using accelerator backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    def _test_ddp_comm_hook_allreduce_hook(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
        """
        This unit test verifies whether a DDP communication hook that just calls
        allreduce gives the same result with the case of no hook registered.
        Without the then callback, the future_value in reducer is no longer
        a PyObject, and this unit test verifies future_value is properly checked.
        """
        process_group = self._get_process_group()

        def allreduce_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            tensors = [bucket.buffer() / self.world_size]
            return (
                process_group.allreduce(tensors)
                .get_future()
                .then(lambda fut: fut.value()[0])
            )

        # Get GPU model with allreduce_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_hook, gradient_as_bucket_view, static_graph
        )

        # check whether the grads are equal to what DDP without hook would return.
        self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_default_ddp_comm_hooks(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether default Python DDP communication hooks ALLREDUCE, FP16_COMPRESS
        and BF16_COMPRESS, can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        # For these default DDP comm hooks, the only state is process group.
        state = process_group
        hook_options = [default.allreduce_hook, default.fp16_compress_hook]
        if (
            not TEST_WITH_ROCM
            and BFLOAT16_AVAILABLE
            and c10d.is_nccl_available()
            and DEVICE_TYPE == "cuda"
            and torch.cuda.nccl.version() >= (2, 10)
        ):
            hook_options.append(default.bf16_compress_hook)
        for hook in hook_options:
            # Get GPU model with the hook registered.
            # The first arg 'process_group' is used for initializing the test environment,
            # so it cannot be replaced by 'state', although they have the same value.
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group, hook, gradient_as_bucket_view, state
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_fp16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the FP16_WRAPPER can give the same result as when there is no hook registered.
        """
        process_group = self._get_process_group()
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        hook_args = [
            (powerSGD.powerSGD_hook, powerSGD_state),
            (default.allreduce_hook, process_group),
        ]

        for hook, state in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.fp16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state,
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_powerSGD_ddp_comm_hook(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether Python DDP communication hook POWER_SGD
        can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        # Get GPU model with the hook registered.
        # Test the hook with different algorithmic configs.
        for use_error_feedback, warm_start, batch_tensors_with_same_shape in product(
            [True, False],
            [True, False],
            [True, False],
        ):
            state = powerSGD.PowerSGDState(
                process_group=process_group,
                matrix_approximation_rank=1,
                use_error_feedback=use_error_feedback,
                warm_start=warm_start,
                batch_tensors_with_same_shape=batch_tensors_with_same_shape,
            )
            for hook in [powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook]:
                gpu_model = self._gpu_model_with_ddp_comm_hook(
                    process_group, hook, gradient_as_bucket_view, state
                )

                # check whether the grads are equal to what DDP without hook would return.
                self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_builtin_ddp_comm_hooks(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether built-in C++ DDP communication hooks ALLREDUCE and FP16_COMPRESS
        can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

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

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_future(self):
        self._test_ddp_comm_hook_allreduce_hook()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks(self):
        self._test_default_ddp_comm_hooks()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_hook_wrapper(self):
        self._test_fp16_compress_wrapper()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks(self):
        self._test_builtin_ddp_comm_hooks()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook(self):
        self._test_powerSGD_ddp_comm_hook()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_future_grad_is_view(self):
        self._test_ddp_comm_hook_allreduce_hook(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_future_static_graph(self):
        self._test_ddp_comm_hook_allreduce_hook(static_graph=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_is_view(self):
        self._test_default_ddp_comm_hooks(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_is_view(self):
        self._test_fp16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_grad_is_view(self):
        self._test_builtin_ddp_comm_hooks(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_grad_is_view(self):
        self._test_powerSGD_ddp_comm_hook(gradient_as_bucket_view=True)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_comm_hook_allreduce_with_then(self):
        """
        This unit test verifies whether a DDP communication hook that calls allreduce and then
        multiplies the result by ten and divides by two gives the expected result.
        """
        process_group = self._get_process_group()

        def allreduce_with_then_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            tensors = [bucket.buffer() / self.world_size]
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                # Multiply the result by 10.
                return 10 * fut.value()[0]

            def div(fut):
                # Divide the result by 2.
                return 0.5 * fut.value()

            return fut.then(mult).then(div)

        # Get GPU model with allreduce_with_then_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_with_then_hook
        )

        # check whether the grads are equal to what allreduce returns multiplied by 5.
        # without the comm_hook, result would be still 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))

    class AcceptsParam(torch.nn.Module):
        def __init__(self, p, factor):
            super().__init__()
            self.a = p
            self.f = factor

        def forward(self, input):
            return input + self.a * self.f

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_weight_sharing(self):
        process_group = self._get_process_group()

        size = 2048 * 2048
        dev = self.rank
        world = self.world_size

        p = torch.nn.Parameter(torch.randn(size, requires_grad=True))

        for try_set_to_none, use_bucket_view in product((False, True), (False, True)):
            m = torch.nn.Sequential(
                self.AcceptsParam(p, dev + 1), self.AcceptsParam(p, dev + 1)
            ).to(torch.device(DEVICE_TYPE, dev))

            m = torch.nn.parallel.DistributedDataParallel(
                m,
                bucket_cap_mb=1,
                gradient_as_bucket_view=use_bucket_view,
                device_ids=[dev],
                process_group=process_group,
            )

            for _ in range(3):
                m.zero_grad(set_to_none=try_set_to_none)
                m(1).sum().backward()

                # Each param value is multiplied by "rank + 1" twice in forward, so the grad
                # values produced by a particular rank should be 2. * (rank + 1).
                # Summing these over ranks and dividing by world size gives the expected result:
                analytic = torch.full_like(
                    p, 2.0 * (world * (world + 1.0) / 2.0) / world, device=dev
                )
                for name, p in m.named_parameters():
                    self.assertEqual(
                        p.grad,
                        analytic,
                        "mismatch at "
                        + name
                        + ".grad for "
                        + f"set_to_none = {try_set_to_none}, use_bucket_view = {use_bucket_view}",
                    )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_ddp_packed_sequence(self):
        """
        Tests that DDP with ``device_ids`` specified can run a forward and
        backward pass with ``PackedSequence`` s with parity compared to a local
        version of the model.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = dist.init_process_group(
            BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        seqs = ["sequence_sequence", "seq", "sequence"]
        vocab = ["<pad>"] + sorted({ch for seq in seqs for ch in seq})
        vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
        # Set the seed to make the embedding and LSTM deterministic (even
        # across ranks since DDP broadcasts parameters from rank 0)
        torch.manual_seed(0)
        embed = nn.Embedding(len(vocab), 4)  # keep on CPU
        lstm = nn.LSTM(input_size=4, hidden_size=2, batch_first=True).to(self.rank)
        lstm_ddp = DistributedDataParallel(
            copy.deepcopy(lstm),
            device_ids=[self.rank],
            process_group=process_group,
        )
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.module.parameters()):
            self.assertEqual(p1, p2)
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_tensor = torch.Tensor(
            torch.zeros((len(vectorized_seqs), seq_lengths.max()))
        ).long()
        for i, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[i, :seq_len] = torch.LongTensor(seq)
        seq_lengths, permutation_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[permutation_idx]
        embedded_seq_tensor = embed(seq_tensor)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor,
            seq_lengths,
            batch_first=True,
        )
        packed_input_ddp = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor.detach().clone(),
            seq_lengths,
            batch_first=True,
        )
        # Move the input to GPU explicitly for the local model
        packed_output, (ht, ct) = lstm(packed_input.to(self.rank))
        # Let DDP move the input to GPU internally
        packed_output_ddp, (ht_ddp, ct_ddp) = lstm_ddp(packed_input_ddp)
        self.assertEqual(packed_output.data, packed_output_ddp.data)
        self.assertEqual(ht, ht_ddp)
        self.assertEqual(ct, ct_ddp)
        packed_output.data.sum().backward()
        packed_output_ddp.data.sum().backward()
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_channels_last_contig(self):
        process_group = self._get_process_group()
        device = torch.device(f"{DEVICE_TYPE}:{self.rank}")
        tensor = torch.ones((2, 16, 768, 1152), dtype=torch.float32, device=device).to(
            memory_format=torch.channels_last
        )
        process_group.broadcast([tensor]).wait()

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_bucket_cap_mb_list_initialization(self):
        """Test that bucket_cap_mb_list is properly converted to bucket_bytes_cap_list"""
        process_group = self._get_process_group()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]

        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        ).to(device_id)

        bucket_cap_mb_list = [10, 25, 50]

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
            bucket_cap_mb_list=bucket_cap_mb_list,
        )

        # Verify bucket_bytes_cap_list was set correctly
        expected_bucket_bytes_cap_list = [
            int(cap * 1024 * 1024) for cap in bucket_cap_mb_list
        ]
        self.assertEqual(
            ddp_model.bucket_bytes_cap_list,
            expected_bucket_bytes_cap_list,
            lambda msg: f"{msg}\nbucket_bytes_cap_list should be {expected_bucket_bytes_cap_list}",
        )

        # Verify bucket_bytes_cap is set to max value
        expected_bucket_bytes_cap = int(max(bucket_cap_mb_list) * 1024 * 1024)
        self.assertEqual(
            ddp_model.bucket_bytes_cap,
            expected_bucket_bytes_cap,
            lambda msg: f"{msg}\nbucket_bytes_cap should be {expected_bucket_bytes_cap}",
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_bucket_cap_mb_list_default_behavior(self):
        """Test that default bucket_cap_mb is used when neither parameter is provided"""
        process_group = self._get_process_group()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        ).to(device_id)

        # Neither bucket_cap_mb nor bucket_cap_mb_list provided
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        # Default should be 25 MB
        expected_bucket_bytes_cap = int(25 * 1024 * 1024)
        self.assertEqual(
            ddp_model.bucket_bytes_cap,
            expected_bucket_bytes_cap,
            "Default bucket_bytes_cap should be 25 MB",
        )

        # bucket_bytes_cap_list should be empty
        self.assertEqual(
            ddp_model.bucket_bytes_cap_list,
            [],
            "bucket_bytes_cap_list should be empty when not provided",
        )

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_bucket_cap_mb_alone(self):
        """Test that bucket_cap_mb works correctly when provided alone"""
        process_group = self._get_process_group()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        ).to(device_id)

        bucket_cap_mb = 100

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
        )

        # Verify bucket_bytes_cap is set correctly
        expected_bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        self.assertEqual(
            ddp_model.bucket_bytes_cap,
            expected_bucket_bytes_cap,
            lambda msg: f"{msg}\nbucket_bytes_cap should be {expected_bucket_bytes_cap}",
        )

        # bucket_bytes_cap_list should be empty
        self.assertEqual(
            ddp_model.bucket_bytes_cap_list,
            [],
            "bucket_bytes_cap_list should be empty when bucket_cap_mb_list not provided",
        )


class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

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

        # The tensors to pass to broadcast are identical to the target
        # only on the process that is the root of the broadcast.
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            tensors = [torch.zeros_like(tensor) for tensor in target]

        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend=BACKEND, store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"{DEVICE_TYPE}:{self.rank}")
        ranks = [0, 1]
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_coalesced_manager_op_integrity(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend=BACKEND, store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"{DEVICE_TYPE}:{self.rank:d}")
        with self.assertRaisesRegex(
            RuntimeError,
            "Coalescing manager requires all collectives to be the same type",
        ):
            with torch.distributed._coalescing_manager(
                group=process_group, async_ops=True
            ):
                t = torch.ones(60, device=device) * (self.rank + 1)
                torch.distributed.all_reduce(t)
                output = torch.zeros(60 * self.world_size, device=device)
                torch.distributed.all_gather_single(output, t)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_nccl_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend=BACKEND, rank=self.rank, world_size=self.world_size, store=store
        )

        c10d.barrier(device_ids=[self.rank])

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_wait_tensor(self) -> None:
        # Verify that c10d_functional.wait_tensor() can be invoked on
        # output tensor of non-functional collective
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend=BACKEND, rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: under context manager (i.e. work is registered in registry)
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            input1 = torch.full(
                (10, 10), float(self.rank), device=f"{DEVICE_TYPE}:{self.rank}"
            )
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            torch.ops.c10d_functional.wait_tensor(input1)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

        # Case 2: not under context manager (i.e. work is not registered in registry)
        input1 = torch.full(
            (10, 10), float(self.rank), device=f"{DEVICE_TYPE}:{self.rank}"
        )
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        # this does not take effect, since the underlying wait_tensor() logic would not
        # be able to find the corresponding work object (because it's not registered in registry)
        torch.ops.c10d_functional.wait_tensor(input1)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

        input2 = torch.full(
            (10, 10), float(self.rank), device=f"{DEVICE_TYPE}:{self.rank}"
        )
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work.wait()
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        self.assertEqual(input1, input2)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_nncl_rank_membership(self):
        self._test_rank_membership(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_mismatch(self):
        self._test_tensor_dtype_mismatch(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensor = torch.zeros(2, dtype=torch.int64).to(self.rank)
        input_tensors = torch.arange(self.world_size * 2, dtype=torch.int64).to(
            self.rank
        )
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        dist.reduce_scatter_single(output_tensor, input_tensors)
        self.assertEqual(output_tensor, input_tensors[self.rank] * self.world_size)


class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        if DEVICE_TYPE == "cuda" and c10d.is_nccl_available():
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync(self):
        self._test_new_group_local_sync(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_sanity_check(self):
        self._test_new_group_local_sync_sanity_check(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_duplicated_pg(self):
        self._test_new_group_local_sync_duplicate_pg(backend=BACKEND)

    @requires_accelerator_dist_backend()
    @skip_if_lt_x_gpu(4)
    def test_new_group_ordered(self):
        self._test_new_group_ordered(backend=BACKEND)


instantiate_parametrized_tests(CommTest)
instantiate_parametrized_tests(LargeCommTest)


if __name__ == "__main__":
    run_tests()
