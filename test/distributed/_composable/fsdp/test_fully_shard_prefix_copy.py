# Owner(s): ["oncall: distributed"]

import math
import unittest
from collections import Counter
from unittest.mock import Mock

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGatherInput
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_all_gather_output_fn,
    _default_reduce_scatter_input_fn,
    AllGatherResult,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    _get_all_gather_output_layout,
    _normalize_all_gather_inputs,
    FSDPParam,
    ShardedState,
)
from torch.distributed.tensor import Shard
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


class _OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[func] += 1
        return func(*args, **(kwargs or {}))


@unittest.skipIf(IS_WINDOWS, "FSDP2 is not supported on Windows")
@unittest.skipIf(not dist.is_available(), "distributed not available")
class TestPrefixCopy(TestCase):
    def _make_extension_param(self, world_size, shard_dim, padded_size, inputs):
        param = FSDPParam.__new__(FSDPParam)
        param.mesh_info = Mock(spec=FSDPMeshInfo, shard_mesh_size=world_size)
        param.fsdp_placement = Shard(shard_dim)
        param.padded_sharded_param_size = torch.Size(padded_size)
        param.sharded_state = ShardedState.SHARDED
        param.offload_to_cpu = False
        param._shard_mesh = Mock()
        local_tensor = Mock(spec=["fsdp_pre_all_gather", "fsdp_post_all_gather"])
        local_tensor.fsdp_pre_all_gather = lambda mesh: (inputs, None)
        param.sharded_param = Mock(_local_tensor=local_tensor)
        param._init_extensions()
        param.all_gather_outputs = []
        return param

    @parametrize(
        "dim,output_size,match",
        [
            (2, None, "dim 2 is invalid"),
            (-3, None, "dim -3 is invalid"),
            (0, torch.Size((-1, 12)), "must be nonnegative"),
            (1, torch.Size((2, 5)), "must contain 12 elements"),
        ],
    )
    def test_all_gather_input_invalid(self, device, dim, output_size, match):
        tensor = torch.empty(2, 3, device=device)
        with self.assertRaisesRegex(ValueError, match):
            _normalize_all_gather_inputs(
                (AllGatherInput(tensor, dim, output_size),),
                world_size=2,
                shard_dim=1,
                padded_sharded_size=tensor.size(),
                require_padding=False,
            )

    @parametrize("explicit_layout", [False, True])
    def test_all_gather_input_padding(self, device, explicit_layout):
        tensor = torch.empty(2, 3, device=device)
        inputs = (AllGatherInput(tensor, dim=1) if explicit_layout else tensor,)
        kwargs = {
            "world_size": 2,
            "shard_dim": 0,
            "padded_sharded_size": torch.Size((4, 3)),
            "require_padding": True,
        }
        if explicit_layout:
            tensors, layouts = _normalize_all_gather_inputs(inputs, **kwargs)
            self.assertIs(tensors[0], tensor)
            self.assertEqual(layouts[0].output_size, (2, 6))
        else:
            with self.assertRaisesRegex(AssertionError, "padded sharded size"):
                _normalize_all_gather_inputs(inputs, **kwargs)

    @parametrize("world_size", [1, 2])
    def test_legacy_all_gather_input_size(self, device, world_size):
        tensor = torch.empty(2, 3, device=device)
        kwargs = {
            "world_size": world_size,
            "shard_dim": 1,
            "padded_sharded_size": torch.Size((4, 3)),
            "require_padding": False,
        }
        if world_size == 1:
            tensors, layouts = _normalize_all_gather_inputs((tensor,), **kwargs)
            self.assertIs(tensors[0], tensor)
            self.assertEqual(tensor.view(layouts[0].output_size).size(), (2, 3))
            self.assertEqual(layouts[0].outer_size, 1)
        else:
            with self.assertRaisesRegex(
                RuntimeError, "Shard.*all-gather output must have.*elements"
            ):
                _normalize_all_gather_inputs((tensor,), **kwargs)

    @parametrize("input_size", [None, (0, 3), (3, 0)])
    def test_empty_all_gather_inputs(self, device, input_size):
        kwargs = {
            "world_size": 2,
            "shard_dim": 1,
            "padded_sharded_size": torch.Size((4, 3)),
            "require_padding": False,
        }
        if input_size is None:
            self.assertEqual(_normalize_all_gather_inputs((), **kwargs), ([], ()))
        else:
            tensor = torch.empty(input_size, device=device)
            tensors, layouts = _normalize_all_gather_inputs((tensor,), **kwargs)
            self.assertIs(tensors[0], tensor)
            self.assertEqual(
                tensor.view(layouts[0].output_size).size(),
                (input_size[0] * kwargs["world_size"], *input_size[1:]),
            )
            self.assertEqual(layouts[0].outer_size, 1)

    def test_all_gather_mixed_empty_inputs(self, device):
        world_size = 2
        expected = make_tensor((2, 4, 3), device=device, dtype=torch.float32)
        shards = [shard.contiguous() for shard in expected.chunk(world_size, dim=1)]
        empty = torch.empty(0, 3, device=device)
        tensors, layouts = _normalize_all_gather_inputs(
            (shards[0], empty),
            world_size=world_size,
            shard_dim=1,
            padded_sharded_size=shards[0].size(),
            require_padding=False,
        )
        self.assertIs(tensors[0], shards[0])
        self.assertIs(tensors[1], empty)
        self.assertEqual([layout.outer_size for layout in layouts], [2, 1])
        self.assertEqual([layout.dim for layout in layouts], [1, 0])
        param = Mock(all_gather_outputs=[], all_gather_copy_layouts=layouts)
        param.init_all_gather_outputs = FSDPParam.init_all_gather_outputs.__get__(param)
        param.alloc_all_gather_outputs = FSDPParam.alloc_all_gather_outputs.__get__(
            param
        )
        result = AllGatherResult(
            torch.cat([shard.flatten() for shard in shards]),
            None,
            None,
            [[expected.dtype, empty.dtype]],
            [[shards[0].numel(), 0]],
            [shards[0].numel(), 0],
        )
        with torch.no_grad(), _OpCounter() as counter:
            _default_all_gather_output_fn([param], result, world_size)
        output, empty_output = param.all_gather_outputs
        self.assertEqual(output.view_as(expected), expected, atol=0, rtol=0)
        self.assertEqual(empty_output.view(layouts[1].output_size).size(), (0, 3))
        self.assertEqual(counter.counts[torch.ops.aten.cat.out], 1)

    @parametrize("nonzero_shards", [False, True])
    @parametrize("world_size", [1, 4])
    @parametrize("mixed_layout", [False, True])
    @dtypes(torch.bfloat16)
    def test_reduce_scatter_preparation(
        self, device, dtype, nonzero_shards, world_size, mixed_layout
    ):
        shapes = [
            (world_size * 3 - 1, 5),
            (2, world_size * 3, 5),
            (2, 3, world_size * 2),
        ]
        grads = [make_tensor(shape, device=device, dtype=dtype) for shape in shapes]
        if mixed_layout:
            grads[2] = grads[2].transpose(0, 1).contiguous().transpose(0, 1)
        shard_dims = list(range(len(grads))) if nonzero_shards else [0] * len(grads)
        shards = []
        for dim, grad in zip(shard_dims, grads):
            shape = list(grad.shape)
            shape[dim] = math.ceil(shape[dim] / world_size) * world_size
            padded = grad.new_zeros(shape)
            padded.narrow(dim, 0, grad.size(dim)).copy_(grad)
            shards.append(padded.chunk(world_size, dim))
        expected = torch.cat(
            [shard[rank].flatten() for rank in range(world_size) for shard in shards]
        ).float()
        params = [Mock(fsdp_placement=Shard(dim)) for dim in shard_dims]
        prepared = _default_reduce_scatter_input_fn(params, grads, world_size)
        sizes = prepared.padded_unsharded_sizes
        self.assertIs(prepared.copy_in, foreach_reduce_scatter_copy_in)
        self.assertEqual(len(sizes), len(params))
        self.assertEqual(sum(size.numel() for size in sizes), expected.numel())
        output = torch.empty_like(expected)
        with _OpCounter() as counter:
            prepared.copy_in(grads, output, world_size)
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(counter.counts[torch.ops.fsdp.chunk_cat.default], 1)

    @parametrize(
        "layout",
        [
            "shard0",
            "shard1",
            "singleton_outer_size",
            "mixed",
            "extension",
            "post_forward",
            "post_forward_nonzero",
            "zero_outer_size",
            "mixed_fallback",
        ],
    )
    def test_all_gather_output(self, device, layout):
        self._test_all_gather_output(device, layout)

    def test_all_gather_empty_output(self, device):
        self._test_all_gather_output(device, "all_empty")

    def _test_all_gather_output(self, device, layout):
        world_size = 4
        layouts = {
            "all_empty": ("zero_outer_size",),
            "mixed": ("shard0", "shard1"),
            "zero_outer_size": ("zero_outer_size", "shard0"),
            "mixed_fallback": (
                "shard1",
                "extension",
                "post_forward",
                "zero_outer_size",
            ),
        }.get(layout, (layout,))
        params, expected, shards, outputs = [], [], [], []
        for kind in layouts:
            dim = 0 if kind in ("shard0", "post_forward") else 1
            shape = {
                "shard0": (12, 5),
                "singleton_outer_size": (1, 12, 5),
                "post_forward": (16, 5),
                "post_forward_nonzero": (16, 8),
                "zero_outer_size": (0, 12, 5),
            }.get(kind, (2, 12, 5))
            dtype = torch.bfloat16 if layout == "mixed" and dim == 1 else torch.float32
            tensor = make_tensor(shape, device=device, dtype=dtype)
            is_post_forward = kind in ("post_forward", "post_forward_nonzero")
            if is_post_forward:
                rank_shards = tensor.flatten().chunk(world_size)
                shard_size = list(tensor.size())
                shard_size[dim] //= world_size * 2
                state = ShardedState.SHARDED_POST_FORWARD
            else:
                rank_shards = tensor.chunk(world_size, dim=dim)
                shard_size = list(rank_shards[0].size())
                state = ShardedState.SHARDED
            output = tensor.new_empty(tensor.numel())
            params.append(
                Mock(
                    fsdp_placement=Shard(dim),
                    padded_sharded_param_size=torch.Size(shard_size),
                    sharded_state=state,
                    _sharded_local_tensor=(
                        Mock(spec=["fsdp_pre_all_gather"])
                        if kind == "extension"
                        else rank_shards[0]
                    ),
                    _sharded_post_forward_param_data=(
                        rank_shards[0] if is_post_forward else None
                    ),
                    all_gather_outputs=[output],
                    all_gather_copy_layouts=[
                        _get_all_gather_output_layout(
                            rank_shards[0].size(),
                            0 if is_post_forward else dim,
                            world_size,
                        )
                    ],
                )
            )
            expected.append(tensor)
            shards.append(rank_shards)
            outputs.append(output)

        gather_dtype = torch.uint8 if layout == "mixed" else torch.float32
        packed = torch.cat(
            [
                shard[rank].contiguous().view(gather_dtype).flatten()
                for rank in range(world_size)
                for shard in shards
            ]
        )
        splits = [
            tensor.nbytes // world_size
            if gather_dtype == torch.uint8
            else tensor.numel() // world_size
            for tensor in expected
        ]
        result = AllGatherResult(
            packed,
            None,
            None,
            [[tensor.dtype] for tensor in expected],
            [[tensor.numel() // world_size] for tensor in expected],
            splits,
        )
        versions = [output._version for output in outputs]
        with torch.no_grad(), _OpCounter() as counter:
            _default_all_gather_output_fn(params, result, world_size)

        self.assertEqual(
            [output.view(tensor.shape) for output, tensor in zip(outputs, expected)],
            expected,
            atol=0,
            rtol=0,
        )
        self.assertEqual([output._version for output in outputs], versions)
        self.assertEqual(
            counter.counts[torch.ops.fsdp.split_with_sizes_copy.default],
            int(packed.numel() > 0),
        )
        reorder_layouts = ("shard1", "singleton_outer_size", "extension")
        num_reorders = sum(kind in reorder_layouts for kind in layouts)
        self.assertEqual(counter.counts[torch.ops.aten.cat.out], num_reorders)

    @parametrize("shard_dim", [1, 2])
    @parametrize(
        "cached_output,payload_matches_param",
        [(True, False), (True, True), (False, False)],
    )
    @dtypes(torch.float32, torch.bfloat16)
    def test_all_gather_byte_input(
        self,
        device,
        dtype,
        shard_dim,
        cached_output,
        payload_matches_param,
    ):
        world_size = 2
        expected = make_tensor((2, 4, 8), device=device, dtype=dtype)
        shards = expected.chunk(world_size, dim=shard_dim)
        inputs = [shard.contiguous().view(torch.uint8).flatten() for shard in shards]
        padded_size = list(shards[0].size())
        if payload_matches_param:
            padded_size[shard_dim] *= expected.element_size()
        param = self._make_extension_param(
            world_size, shard_dim, padded_size, (inputs[0],)
        )
        if cached_output:
            param.init_all_gather_outputs(
                [shards[0].numel()], [dtype], world_size, expected.device
            )
            (output,) = param.all_gather_outputs
            version = output._version
        result = AllGatherResult(
            torch.cat(inputs),
            None,
            None,
            [[torch.uint8]],
            [[inputs[0].numel()]],
            [inputs[0].numel()],
        )
        if not cached_output:
            with self.assertRaisesRegex(
                RuntimeError, "Shard.*all-gather output must have.*elements"
            ):
                _ = param.all_gather_inputs
            return
        _ = param.all_gather_inputs
        self.assertEqual(param._unflatten_all_gather_outputs()[0].size(), output.size())
        with torch.no_grad(), _OpCounter() as counter:
            _default_all_gather_output_fn([param], result, world_size)
        self.assertIs(param.all_gather_outputs[0], output)
        self.assertEqual(output.dtype, dtype)
        self.assertEqual(output.view_as(expected), expected, atol=0, rtol=0)
        self.assertEqual(output._version, version)
        self.assertEqual(counter.counts[torch.ops.aten.cat.out], 1)

    @parametrize("shard_dim", [0, 1, 2])
    def test_all_gather_changing_payload_size(self, device, shard_dim):
        world_size = 2
        local = torch.arange(2**shard_dim, device=device, dtype=torch.float32)
        local = local.repeat_interleave(2).view((2,) * (shard_dim + 1))
        outer_size = math.prod(local.shape[:shard_dim])
        expected = torch.cat([local] * world_size, dim=shard_dim)
        param = self._make_extension_param(world_size, shard_dim, local.size(), ())
        param.sharded_param.requires_grad = False
        param.param_dtype, param.orig_dtype = None, local.dtype
        param._orig_size = expected.size()
        param._contiguous_orig_stride = expected.stride()
        param.is_spmd_types = False
        param._unsharded_dtensor_spec = None

        def post_hook(outputs, length, dtype, *, out=None):
            packed = outputs[0].view(outer_size, world_size, -1).transpose(0, 1)
            encoded = packed.flatten()[: world_size * length].view(world_size, -1)
            decoded = encoded.repeat_interleave(local.numel() // length, dim=1)
            decoded = torch.cat(
                decoded.view(world_size, *local.shape).unbind(), dim=shard_dim
            )
            if out is not None:
                with torch.autograd._unsafe_preserve_version_counter(out):
                    out.copy_(decoded)
                return None
            return decoded, (decoded,)

        inner = param.sharded_param._local_tensor
        inner.fsdp_pre_all_gather = lambda mesh: ((payload,), payload.numel())
        inner.fsdp_post_all_gather = post_hook
        output = None
        for compressed in (False, True, False):
            local.add_(10)
            payload = local.flatten()[::2].contiguous() if compressed else local
            expected = torch.cat([local] * world_size, dim=shard_dim)
            inputs = param.all_gather_inputs
            result = AllGatherResult(
                torch.cat(inputs * world_size),
                None,
                None,
                [[local.dtype]],
                [[payload.numel()]],
                [payload.numel()],
            )
            with torch.no_grad():
                _default_all_gather_output_fn([param], result, world_size)
                param.init_unsharded_param()
            self.assertEqual(param.unsharded_param, expected, atol=0, rtol=0)
            if output is None:
                (output,) = param.all_gather_outputs
                pointer, version = output.data_ptr(), output._version
            self.assertIs(param.all_gather_outputs[0], output)
            self.assertEqual(output.numel(), expected.numel())
            self.assertEqual(output.data_ptr(), pointer)
            self.assertEqual(output._version, version)


instantiate_device_type_tests(TestPrefixCopy, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
