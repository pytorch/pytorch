# Owner(s): ["oncall: distributed"]

import itertools
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from _test_fully_shard_common import (
    get_active_memory_bytes,
    GPT,
    GPTConfig,
    ModelWithParamsAndBuffers,
)
from torch.distributed._composable.fsdp import (
    FSDP,
    fully_shard,
    InitPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)

from torch.distributed._tensor import DeviceMesh, Replicate
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, FSDPTestMultiThread
from torch.testing._internal.common_utils import run_tests


class TestFullyShardInit(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_registered_params(self):
        """
        Tests that ``fully_shard`` shards parameters on dim-0 and exposes them
        from ``nn.Module.named_parameters()``, preserving their original order.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "lin_shapes": [
                    [(16, 15), (15, 8)],
                    [(7, 15), (15, 3)],
                ],
                "device_type": ["cuda"],
            },
            self._test_fully_shard_registered_params,
        )

    def _test_fully_shard_registered_params(
        self,
        reshard_after_forward: Union[bool, int],
        lin_shapes: List[Tuple[int, int]],
        device_type: str,
    ):
        assert len(lin_shapes) == 2, f"Expects two linear shapes but got {lin_shapes}"
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(*lin_shapes[0]), nn.ReLU(), nn.Linear(*lin_shapes[1])
        )
        ref_named_params = [
            (n, p.detach().clone()) for n, p in model.named_parameters()
        ]
        shard_mesh_dim = 0
        fully_shard(model, device=device_type)
        # Initialize the mesh still for downstream usage
        mesh = DeviceMesh(device_type, torch.arange(self.world_size))
        sharded_param_shapes = [p._local_tensor.shape for p in model.parameters()]
        for (_, ref_param), sharded_shape in zip(
            ref_named_params, sharded_param_shapes
        ):
            expected_shape = torch.chunk(ref_param, mesh.size(shard_mesh_dim), dim=0)[
                self.rank
            ].shape
            self.assertEqual(expected_shape, sharded_shape)
        for (ref_param_name, ref_param), (param_name, param) in zip(
            ref_named_params, model.named_parameters()
        ):
            self.assertEqual(ref_param_name, param_name)
            replicate_param = param.redistribute(mesh, [Replicate()]).to_local()
            self.assertEqual(ref_param.shape, replicate_param.shape)
            self.assertEqual(ref_param, replicate_param)

    @skip_if_lt_x_gpu(2)
    def test_enable_wrap(self):
        """
        Tests compatibility with ``enable_wrap()`` to do module-by-module
        initialization and sharding.
        """
        self.run_subtests(
            {"use_mp_policy": [False, True]},
            self._test_enable_wrap,
        )

    def _test_enable_wrap(self, use_mp_policy: bool):
        param_dtype = torch.bfloat16 if use_mp_policy else None
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype)
        kwargs = {"reshard_after_forward": True, "mp_policy": mp_policy}
        # Use some smaller hyperparameters for unit testing
        config = GPTConfig(n_layer=6, vocab_size=2048)
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        before_active_memory_bytes = get_active_memory_bytes()
        with enable_wrap(wrapper_cls=fully_shard, **kwargs):
            model = GPT(config)
            model = wrap(model)
        after_active_memory_bytes = get_active_memory_bytes()
        sharded_active_memory_usage_bytes = (
            after_active_memory_bytes - before_active_memory_bytes
        )
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        before_active_memory_bytes = get_active_memory_bytes()
        ref_model = GPT(config).cuda()
        after_active_memory_bytes = get_active_memory_bytes()
        unsharded_active_memory_usage_bytes = (
            after_active_memory_bytes - before_active_memory_bytes
        )
        # The expected memory usage is the sharded parameter memory plus the
        # max unsharded parameter memory (in this case, from the block). Due to
        # initial segments in the caching allocator, the active memory used may
        # be slightly lower than the real active memory contribution.
        block_numel = sum(p.numel() for p in ref_model.transformer["h"][0].parameters())
        total_numel = sum(p.numel() for p in ref_model.parameters())
        bytes_per_numel = torch.float32.itemsize
        expected_unsharded_bytes = total_numel * bytes_per_numel
        assert expected_unsharded_bytes >= unsharded_active_memory_usage_bytes, (
            "Sanity check failed: expected the reference model to use at most "
            f"{expected_unsharded_bytes} but got {unsharded_active_memory_usage_bytes}"
        )
        expected_sharded_bytes = (
            (total_numel + self.world_size - 1) // self.world_size + block_numel
        ) * bytes_per_numel
        if use_mp_policy:
            # For mixed precision, we cast the unsharded parameter to the low
            # precision before freeing, increasing the peak memory from holding
            # both copies in memory at once
            expected_sharded_bytes += block_numel * param_dtype.itemsize
        self.assertTrue(expected_sharded_bytes >= sharded_active_memory_usage_bytes)
        # Unshard all parameters to check for parity with the reference model
        for module in model.modules():
            if isinstance(module, FSDP):
                module.unshard()
        for (n, p), (ref_n, ref_p) in zip(
            model.named_parameters(), ref_model.named_parameters()
        ):
            self.assertEqual(n, ref_n)
            self.assertEqual(p.shape, ref_p.shape)
            if use_mp_policy:
                ref_p = ref_p.to(param_dtype)
            self.assertEqual(p, ref_p)

    @skip_if_lt_x_gpu(2)
    def test_init_meta_device_module_to_cuda(self):
        """
        Tests initializing a meta-device module to CUDA both with the default
        :meth:`reset_parameters` path and with a user-specified
        ``param_init_fn``.
        """

        def param_init_fn(module: nn.Module) -> None:
            if any(param.is_meta for param in module.parameters(recurse=False)) or any(
                buf.is_meta for buf in module.buffers(recurse=False)
            ):
                module.to_empty(device=torch.device("cuda"), recurse=False)
                module.reset_parameters()

        self.run_subtests(
            {"param_init_fn": [None, param_init_fn]},
            self._test_meta_device_module,
        )

    def _test_meta_device_module(
        self,
        param_init_fn: Optional[Callable],
    ):
        ref_model = ModelWithParamsAndBuffers(device=torch.device("cuda"))
        for module in (
            ref_model.l3.lin1,
            ref_model.l3.lin2,
            ref_model.l3.buf_mod,
            ref_model.l4,
            ref_model,
        ):
            fully_shard(module)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        model = ModelWithParamsAndBuffers(device=torch.device("meta"))
        init_policy = InitPolicy(param_init_fn=param_init_fn)
        for module in (model.l3.lin1, model.l3.lin2, model.l3.buf_mod, model.l4, model):
            fully_shard(module, init_policy=init_policy)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        for (param_name, param), (ref_param_name, ref_param) in zip(
            model.named_parameters(), ref_model.named_parameters()
        ):
            self.assertEqual(param_name, ref_param_name)
            self.assertEqual(param, ref_param)

        torch.manual_seed(42 + self.rank)
        inp = torch.randn((4, 3), device="cuda")
        for iter_idx in range(5):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp))
                losses[-1].sum().backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_sync_module_states_cuda(self):
        """
        Tests passing ``sync_module_states=True`` when initializing on CUDA.
        """
        torch.manual_seed(42)
        # Initialize the reference model on CPU and move to CUDA in case CPU
        # vs. CUDA randomness differs
        ref_model = ModelWithParamsAndBuffers(device=torch.device("cpu")).cuda()

        torch.manual_seed(42)
        model = ModelWithParamsAndBuffers(device=torch.device("cpu"))
        if self.rank != 0:
            for tensor in itertools.chain(model.parameters(), model.buffers()):
                tensor.detach().zero_()
        init_policy = InitPolicy(sync_module_states=True)
        for module in (model.l3.lin1, model.l3.lin2, model.l3.buf_mod, model.l4, model):
            fully_shard(module, init_policy=init_policy)

        for module in model.modules():
            if isinstance(module, FSDP):
                module.unshard()
        for (param_name, param), (ref_param_name, ref_param) in zip(
            model.named_parameters(), ref_model.named_parameters()
        ):
            self.assertEqual(param_name, ref_param_name)
            self.assertEqual(param, ref_param)
        for (buf_name, buf), (ref_buf_name, ref_buf) in zip(
            model.named_buffers(), ref_model.named_buffers()
        ):
            self.assertEqual(buf_name, ref_buf_name)
            self.assertEqual(buf, ref_buf)


class TestFSDPInitMultiThread(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_device_mesh_mismatch_raises(self):
        module = nn.Linear(3, 3)
        cuda_mesh = dist._tensor.init_device_mesh("cuda", (self.world_size,))
        with self.assertRaisesRegex(
            ValueError,
            "device and mesh must be of the same type but got cpu for device and cuda for mesh",
        ):
            fully_shard(module, mesh=cuda_mesh, device="cpu")

        module = nn.Linear(3, 3)
        cpu_mesh = dist._tensor.init_device_mesh("cpu", (self.world_size,))
        with self.assertRaisesRegex(
            ValueError,
            "device and mesh must be of the same type but got cuda for device and cpu for mesh",
        ):
            fully_shard(module, mesh=cpu_mesh, device="cuda")

    @skip_if_lt_x_gpu(2)
    def test_invalid_param_init_fn_raises(self):
        model = nn.Linear(3, 3, device="meta")
        with self.assertRaisesRegex(
            ValueError, "Expects param_init_fn to be a callable but got <class 'int'>"
        ):
            fully_shard(model, init_policy=InitPolicy(param_init_fn=42))

        def param_init_fn(module: nn.Module):
            module.register_parameter("p", nn.Parameter(torch.empty((3,))))

        model = nn.Linear(3, 3, device="meta")
        with self.assertRaisesRegex(
            AssertionError,
            r"Calling param_init_fn changed the module's registered parameters "
            r"or buffers \(before 2 vs. after 3\), which is unsupported",
        ):
            fully_shard(model, init_policy=InitPolicy(param_init_fn=param_init_fn))

    @skip_if_lt_x_gpu(2)
    def test_cuda_device(self):
        """
        Tests passing a CUDA device to the ``device`` argument, where the input
        module's device is either CPU or CUDA.
        - If CPU offloading is enabled, then parameters should be on CPU, and
          buffers should be on GPU.
        - If CPU offloading is not enabled, then parameters and buffers should
          be on GPU.
        """
        self.run_subtests(
            {
                "device": [
                    None,
                    "cuda",
                    torch.device("cuda"),
                    torch.device("cuda", torch.cuda.current_device()),
                    torch.cuda.current_device(),
                ],
                "input_device": [
                    torch.device("cpu"),
                    torch.device("cuda", torch.cuda.current_device()),
                ],
                "offload_policy": [OffloadPolicy(), OffloadPolicy("cpu")],
            },
            self._test_cuda_device,
        )

    def _test_cuda_device(
        self,
        device: Optional[Union[str, torch.device, int]],
        input_device: torch.device,
        offload_policy: OffloadPolicy,
    ):
        model = ModelWithParamsAndBuffers(device=input_device)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            self.assertEqual(tensor.device, input_device)
        for module in (
            model.l3.lin1,
            model.l3.lin2,
            model.l3.buf_mod,
            model.l4,
            model,
        ):
            if device is None:
                fully_shard(module, offload_policy=offload_policy)
            else:
                fully_shard(module, device=device, offload_policy=offload_policy)
        expected_param_device = (
            torch.device("cuda", torch.cuda.current_device())
            if offload_policy.offload_type != "cpu"
            else torch.device("cpu")
        )
        expected_buffer_device = torch.device("cuda", torch.cuda.current_device())
        for param in model.parameters():
            self.assertEqual(param.device, expected_param_device)
        for buffer in model.buffers():
            self.assertEqual(buffer.device, expected_buffer_device)

    @skip_if_lt_x_gpu(1)
    def test_apply_methods(self):
        """
        Tests that ``_apply`` methods raise an error if they actually affect
        module states.
        """
        error_regex = (
            "FSDP does not support _apply methods that change tensor storage or device"
        )
        model = ModelWithParamsAndBuffers(device="cpu")
        fully_shard(model, device="cuda")
        with self.assertRaisesRegex(NotImplementedError, error_regex):
            model.cpu()
        model = ModelWithParamsAndBuffers(device="cpu")
        fully_shard(model, device="cpu")
        with self.assertRaisesRegex(NotImplementedError, error_regex):
            model.cuda()


if __name__ == "__main__":
    run_tests()
