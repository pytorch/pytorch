# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import math
import threading
import weakref
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.autograd.grad_mode import _unsafe_preserve_version_counter
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp.experimental import (
    all_gather_output_fn_with_intermediate_copy,
    AllGatherInput,
)
from torch.distributed.tensor import Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import TorchDispatchMode


device_type = torch.device(get_devtype())


def two_tensor_fsdp_pre_all_gather_v1(
    self, mesh: DeviceMesh
) -> tuple[tuple[torch.Tensor, ...], Any]:
    all_gather_inputs = (self.a, self.b)
    metadata = None
    return all_gather_inputs, metadata


def two_tensor_fsdp_pre_all_gather_v2(
    self,
    mesh: DeviceMesh,
    outer_size: torch.Size,
    outer_stride: tuple[int, ...],
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy,
) -> tuple[tuple[torch.Tensor, ...], Any]:
    all_gather_inputs = (self.a, self.b)
    metadata = None
    return all_gather_inputs, metadata


def two_tensor_fsdp_post_all_gather(
    self,
    all_gather_outputs: tuple[torch.Tensor, ...],
    metadata: Any,
    param_dtype: torch.dtype,
    *,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
    if metadata is not None:
        raise AssertionError(f"Expected metadata to be None, got {metadata}")
    a, b = all_gather_outputs
    if out is not None:
        if not isinstance(out, TwoTensor):
            raise AssertionError(f"Expected TwoTensor, got {type(out)}")
        if a.dtype == param_dtype:
            if a.untyped_storage().data_ptr() != out.a.untyped_storage().data_ptr():
                raise AssertionError("a storage data_ptr mismatch with out.a")
            if b.untyped_storage().data_ptr() != out.b.untyped_storage().data_ptr():
                raise AssertionError("b storage data_ptr mismatch with out.b")
        else:
            if out.a.dtype != param_dtype:
                raise AssertionError(f"out.a dtype {out.a.dtype} != {param_dtype}")
            if out.b.dtype != param_dtype:
                raise AssertionError(f"out.b dtype {out.b.dtype} != {param_dtype}")
            out.a.copy_(a)
            out.b.copy_(b)
        return
    tensors_to_free = (a, b)
    # If the cast is real, then the all-gather outputs will not alias the
    # returned `TwoTensor`'s `a` and `b`
    two_tensor = TwoTensor(a, b).to(param_dtype)
    return two_tensor, tensors_to_free


class BFloat16AllGatherTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor, pad_in_pre_all_gather: bool = True):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            data.stride(),
            data.storage_offset(),
            dtype=data.dtype,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, pad_in_pre_all_gather: bool = True):
        self._data = data
        self._pad_in_pre_all_gather = pad_in_pre_all_gather

    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ) -> tuple[tuple[torch.Tensor, ...], Any]:
        if mesh.ndim != 1:
            raise AssertionError(f"Expected mesh.ndim == 1, got {mesh.ndim}")
        mesh_size = mesh.size()
        requires_padding = outer_size[0] % mesh_size != 0
        if requires_padding and self._pad_in_pre_all_gather:
            sharded_padded_size = list(outer_size)
            sharded_padded_size[0] = math.ceil(outer_size[0] / mesh_size)
            padded_out = torch.empty(
                sharded_padded_size, dtype=torch.bfloat16, device=self.device
            )
            padded_out[: self._data.size(0)].copy_(self._data)
            return (padded_out,), None
        else:
            return self._data.to(torch.bfloat16), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
        if metadata is not None:
            raise AssertionError(f"Expected metadata to be None, got {metadata}")
        (tensor,) = all_gather_outputs
        if tensor.dtype != torch.bfloat16:
            raise AssertionError(
                f"Expected tensor.dtype == torch.bfloat16, got {tensor.dtype}"
            )
        if out is not None:
            with _unsafe_preserve_version_counter(out):
                out.copy_(tensor)
            return
        upcast_tensor = tensor.to(param_dtype)
        return upcast_tensor, (tensor, upcast_tensor)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        pad_in_pre_all_gather = None

        def unwrap(x: cls):
            nonlocal pad_in_pre_all_gather
            if pad_in_pre_all_gather is None:
                pad_in_pre_all_gather = x._pad_in_pre_all_gather
            else:
                if pad_in_pre_all_gather != x._pad_in_pre_all_gather:
                    raise AssertionError(
                        f"pad_in_pre_all_gather mismatch: {pad_in_pre_all_gather} vs {x._pad_in_pre_all_gather}"
                    )
            return x._data

        out = func(
            *pytree.tree_map_only(cls, unwrap, args),
            **pytree.tree_map_only(cls, unwrap, kwargs),
        )
        return pytree.tree_map_only(
            torch.Tensor, lambda x: cls(x, pad_in_pre_all_gather), out
        )

    def __tensor_flatten__(self):
        return ["_data"], None

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors, outer_size: torch.Size, outer_stride: tuple[int, ...]
    ):
        return inner_tensors["_data"]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"


class TestFullyShardAllGatherExtensionsCommon:
    @property
    def world_size(self) -> int:
        return 2

    @contextlib.contextmanager
    def _patch_two_tensor_fsdp_all_gather(self, pre_all_gather_version: int):
        lock = threading.Lock()
        if pre_all_gather_version == 1:
            TwoTensor.fsdp_pre_all_gather = two_tensor_fsdp_pre_all_gather_v1
        elif pre_all_gather_version == 2:
            TwoTensor.fsdp_pre_all_gather = two_tensor_fsdp_pre_all_gather_v2
        TwoTensor.fsdp_post_all_gather = two_tensor_fsdp_post_all_gather
        dist.barrier()
        try:
            yield
        finally:
            dist.barrier()
            with lock:  # only one thread needs to delete
                if hasattr(TwoTensor, "fsdp_pre_all_gather"):
                    delattr(TwoTensor, "fsdp_pre_all_gather")
                if hasattr(TwoTensor, "fsdp_post_all_gather"):
                    delattr(TwoTensor, "fsdp_post_all_gather")

    def _init_two_tensor_mlp(self) -> nn.Module:
        # Disable bias because the reference model will end up with a bias
        # gradient that is a `TwoTensor`, whereas the FSDP model does not
        model = nn.Sequential(*[MLP(8, bias=False) for _ in range(3)])
        for mlp in model:
            mlp.in_proj.weight = nn.Parameter(
                TwoTensor(mlp.in_proj.weight, mlp.in_proj.weight.clone())
            )
            mlp.out_proj.weight = nn.Parameter(
                TwoTensor(mlp.out_proj.weight, mlp.out_proj.weight.clone())
            )
        return model


@instantiate_parametrized_tests
class TestFullyShardAllGatherExtensionsMultiProcess(
    TestFullyShardAllGatherExtensionsCommon, FSDPTest
):
    @skip_if_lt_x_gpu(2)
    @parametrize("shard_dim", [1, 2])
    @parametrize("invalid_payload", [False, True])
    def test_legacy_all_gather_direct_copy(self, shard_dim: int, invalid_payload: bool):
        self._test_legacy_all_gather_direct_copy(
            shard_dim, invalid_payload, self.world_size
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("shard_dim", [1, 2])
    def test_legacy_all_gather_single_rank(self, shard_dim: int):
        self._test_legacy_all_gather_direct_copy(shard_dim, False, 1)

    def _test_legacy_all_gather_direct_copy(
        self,
        shard_dim: int,
        invalid_payload: bool,
        shard_world_size: int,
    ):
        expected = torch.arange(48, device=device_type).float().view(2, 4, 6) / 64
        post_out_ids: list[int | None] = []
        byte_inputs = False

        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(expected.clone())

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return self.weight * input

        class CopyCounter(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.copies = 0
                self.reorders = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.fsdp._all_gather_copy_out_.default:
                    self.copies += 1
                elif func == torch.ops.aten.cat.out:
                    self.reorders += 1
                return func(*args, **(kwargs or {}))

        def fsdp_pre_all_gather(
            local_tensor,
            mesh: DeviceMesh,
            outer_size: torch.Size,
            outer_stride: tuple[int, ...],
            module: nn.Module,
            mp_policy: MixedPrecisionPolicy,
        ) -> tuple[tuple[torch.Tensor, ...], Any]:
            del outer_stride, module, mp_policy
            weight = local_tensor.view(-1, 6)
            auxiliary = (local_tensor.to(torch.bfloat16) + 1).view(-1, 4)
            if invalid_payload:
                auxiliary = auxiliary[: auxiliary.size(0) // 2]
            if byte_inputs:
                weight = weight.view(torch.uint8)
                auxiliary = auxiliary.view(torch.uint8)
            return (weight, auxiliary), (outer_size, mesh.size())

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
            del local_tensor
            weight, auxiliary = all_gather_outputs
            weight_tail = 24 if byte_inputs else 6
            auxiliary_tail = 8 if byte_inputs else 4
            self.assertEqual(metadata, (expected.size(), shard_world_size))
            self.assertEqual(weight.dtype, param_dtype)
            self.assertEqual(
                weight.size(), (expected.numel() // weight_tail, weight_tail)
            )
            self.assertEqual(weight, expected.view(-1, weight_tail), atol=0, rtol=0)
            self.assertEqual(auxiliary.dtype, torch.bfloat16)
            self.assertEqual(
                auxiliary.size(),
                (expected.numel() // auxiliary_tail, auxiliary_tail),
            )
            self.assertEqual(
                auxiliary,
                (expected + 1).to(torch.bfloat16).view(-1, auxiliary_tail),
                atol=0,
                rtol=0,
            )
            post_out_ids.append(None if out is None else id(out))
            weight = weight.view(expected.size())
            if out is not None:
                with _unsafe_preserve_version_counter(out):
                    out.copy_(weight)
                return None
            return weight, (weight,)

        model = Scale()
        ref_model = copy.deepcopy(model)
        mesh = None
        if shard_world_size == 1:
            mesh = init_device_mesh(
                device_type.type,
                (self.world_size, 1),
                mesh_dim_names=("outer", "shard"),
            )["shard"]
        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=lambda _: Shard(shard_dim),
            reshard_after_forward=True,
        )
        local_weight = model.weight._local_tensor
        local_weight.fsdp_pre_all_gather = fsdp_pre_all_gather.__get__(local_weight)
        local_weight.fsdp_post_all_gather = fsdp_post_all_gather.__get__(local_weight)
        inp = torch.arange(48, device=device_type).float().view_as(expected) / 32
        if invalid_payload:
            with self.assertRaisesRegex(
                RuntimeError, "Shard.*all-gather output must have.*elements"
            ):
                model(inp)
            return
        for iteration in range(2):
            byte_inputs = iteration == 1 and shard_world_size > 1
            model.zero_grad(set_to_none=True)
            ref_model.zero_grad(set_to_none=True)
            with CopyCounter() as counter:
                output = model(inp)
            ref_output = ref_model(inp)
            self.assertEqual(output, ref_output, atol=0, rtol=0)
            self.assertEqual(counter.copies, int(shard_world_size > 1))
            self.assertEqual(counter.reorders, 0)
            output.sum().backward()
            ref_output.sum().backward()
            check_sharded_parity(self, ref_model, model)
        self.assertEqual(len(post_out_ids), 4)
        self.assertIsNotNone(post_out_ids[1])
        self.assertEqual(post_out_ids, [None] + [post_out_ids[1]] * 3)

    @skip_if_lt_x_gpu(2)
    def test_all_gather_extensions_train_parity(self):
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=1):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_train_parity,
            )
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=2):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_train_parity,
            )

    def _test_all_gather_extensions_train_parity(self, reshard_after_forward: bool):
        torch.manual_seed(42)
        model = self._init_two_tensor_mlp()
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=True)
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        check_sharded_parity(self, ref_model, model)

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device=device_type)
        for iter_idx in range(10):
            losses: list[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()
                if _model is ref_model:
                    for _, param in _model.named_parameters():
                        dist.all_reduce(param.grad)
                        param.grad.detach().div_(self.world_size)
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            check_sharded_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(2)
    def test_all_gather_input_layouts(self):
        self.run_subtests(
            {
                "shard_world_size": [1, 2],
                "intermediate_copy": [False, True],
                "release_outputs": [False, True],
            },
            self._test_all_gather_input_layouts,
        )

    def _test_all_gather_input_layouts(
        self, shard_world_size: int, intermediate_copy: bool, release_outputs: bool
    ):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size // shard_world_size, shard_world_size),
            mesh_dim_names=("replicate", "shard"),
        )["shard"]
        expected_weight = torch.arange(64, device=device_type).float().view(8, 8) / 64
        tags = torch.arange(6, dtype=torch.bfloat16, device=device_type).view(2, 3)
        post_out_ids: list[int | None] = []
        payload_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        storage_observations: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

        def fsdp_pre_all_gather(
            local_tensor,
            mesh: DeviceMesh,
            outer_size: torch.Size,
            outer_stride: tuple[int, ...],
            module: nn.Module,
            mp_policy: MixedPrecisionPolicy,
        ) -> tuple[tuple[AllGatherInput, ...], Any]:
            del outer_stride, module, mp_policy
            rank = mesh.get_local_rank()
            payload_tags = tags + rank * 8
            rank_tag = torch.tensor(rank, dtype=torch.int64, device=local_tensor.device)
            payload_refs.extend((weakref.ref(payload_tags), weakref.ref(rank_tag)))
            return (
                AllGatherInput(local_tensor, dim=-1, output_size=torch.Size((2, 4, 8))),
                AllGatherInput(payload_tags),
                AllGatherInput(rank_tag),
            ), (outer_size, mesh.size())

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
            del local_tensor
            weight, gathered_tags, ranks = all_gather_outputs
            self.assertEqual(metadata, (torch.Size((8, 8)), shard_world_size))
            self.assertEqual(weight.shape, (2, 4, 8))
            self.assertEqual(weight.dtype, param_dtype)
            self.assertEqual(weight, expected_weight.view(2, 4, 8))
            self.assertEqual(gathered_tags.shape, (2 * shard_world_size, 3))
            self.assertEqual(gathered_tags.dtype, torch.bfloat16)
            self.assertEqual(
                gathered_tags,
                torch.cat([tags + rank * 8 for rank in range(shard_world_size)]),
            )
            self.assertEqual(ranks.shape, (shard_world_size,))
            self.assertEqual(ranks.dtype, torch.int64)
            self.assertEqual(ranks, torch.arange(shard_world_size, device=device_type))
            post_out_ids.append(None if out is None else id(out))
            weight = weight.view(8, 8)
            if out is not None:
                with _unsafe_preserve_version_counter(out):
                    out.copy_(weight)
                return None
            transformed = weight.clone()
            return transformed, (transformed,)

        def fsdp_should_release_all_gather_outputs_after_post_all_gather(
            local_tensor,
        ) -> bool:
            del local_tensor
            return release_outputs

        class InspectLinear(nn.Linear):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                param_group = fully_shard.state(self)._fsdp_param_group
                if param_group is None:
                    raise AssertionError("Expected an FSDP parameter group")
                (param,) = param_group.fsdp_params
                raw = param.all_gather_outputs
                inner = param._unsharded_inner_tensors
                storage_observations.append(
                    (
                        tuple(t.untyped_storage().size() for t in raw),
                        tuple(t.untyped_storage().size() for t in inner),
                    )
                )
                return super().forward(input)

        model = InspectLinear(8, 8, bias=False, device=device_type)
        ref_model = nn.Linear(8, 8, bias=False, device=device_type)
        with torch.no_grad():
            model.weight.copy_(expected_weight)
            ref_model.weight.copy_(expected_weight)
        fully_shard(
            model,
            mesh=mesh,
            shard_placement_fn=lambda _: Shard(1),
            reshard_after_forward=True,
        )
        if intermediate_copy:
            model.set_all_gather_output_fn(all_gather_output_fn_with_intermediate_copy)
        local_weight = model.weight._local_tensor
        local_weight.fsdp_pre_all_gather = fsdp_pre_all_gather.__get__(local_weight)
        local_weight.fsdp_post_all_gather = fsdp_post_all_gather.__get__(local_weight)
        local_weight.fsdp_should_release_all_gather_outputs_after_post_all_gather = (
            fsdp_should_release_all_gather_outputs_after_post_all_gather.__get__(
                local_weight
            )
        )

        inp = torch.arange(16, device=device_type).float().view(2, 8) / 16
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            ref_model.zero_grad(set_to_none=True)
            output = model(inp)
            ref_output = ref_model(inp)
            self.assertEqual(output, ref_output)
            raw_sizes, inner_sizes = storage_observations[-1]
            self.assertEqual(len(raw_sizes), 3)
            if release_outputs:
                self.assertEqual(raw_sizes, (0, 0, 0))
            else:
                self.assertTrue(all(size > 0 for size in raw_sizes))
            self.assertTrue(all(size > 0 for size in inner_sizes))
            output.sum().backward()
            ref_output.sum().backward()
            check_sharded_parity(self, ref_model, model)

        self.assertEqual(len(post_out_ids), 4)
        self.assertIsNotNone(post_out_ids[1])
        self.assertEqual(post_out_ids, [None] + [post_out_ids[1]] * 3)
        self.assertTrue(all(ref() is None for ref in payload_refs))


class TestFullyShardAllGatherExtensionsMultiThread(
    TestFullyShardAllGatherExtensionsCommon, FSDPTestMultiThread
):
    @property
    def world_size(self) -> int:
        return 8

    @property
    def device(self) -> torch.device:
        return torch.device(device_type)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extensions_end_to_end(self):
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=1):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_end_to_end,
            )
        with self._patch_two_tensor_fsdp_all_gather(pre_all_gather_version=2):
            self.run_subtests(
                {"reshard_after_forward": [True, False]},
                self._test_all_gather_extensions_end_to_end,
            )

    def _test_all_gather_extensions_end_to_end(self, reshard_after_forward: bool):
        # Check that we can run the meta-device initialization flow
        with torch.device("meta"):
            model = self._init_two_tensor_mlp()
        for param in model.parameters():
            self.assertEqual(param.device, torch.device("meta"))
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        model.to_empty(device=self.device)
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # Run a few iterations to check for errors
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device=device_type)
        for _ in range(3):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extensions_monkey_patch(self):
        tls = threading.local()
        tls.ran_pre_all_gather = False

        # Define a pre/post-all-gather pair that quantizes to bf16 for the
        # all-gather and de-quantizes back to the parameter dtype
        def fsdp_pre_all_gather(
            self,
            mesh: DeviceMesh,
            outer_size: torch.Size,
            outer_stride: tuple[int, ...],
            module: nn.Module,
            mp_policy: MixedPrecisionPolicy,
        ) -> tuple[tuple[torch.Tensor, ...], Any]:
            nonlocal tls
            tls.ran_pre_all_gather = True
            return (self.to(torch.bfloat16),), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            self,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
            (tensor,) = all_gather_outputs
            if metadata is not None:
                raise AssertionError(f"Expected metadata to be None, got {metadata}")
            if tensor.dtype != torch.bfloat16:
                raise AssertionError(
                    f"Expected tensor.dtype == torch.bfloat16, got {tensor.dtype}"
                )
            if out is not None:
                with _unsafe_preserve_version_counter(out):
                    out.copy_(tensor)
                return
            upcast_tensor = tensor.to(param_dtype)
            return upcast_tensor, (tensor, upcast_tensor)

        with torch.device("meta"):
            model = self._init_two_tensor_mlp()
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        model.to_empty(device=self.device)
        for param in model.parameters():
            nn.init.trunc_normal_(param)
        # Monkey patch the pre/post-all-gather functions *after* `to_empty()`
        # since the local tensor objects change from materialization
        self.assertGreater(sum("weight" in n for n, _ in model.named_parameters()), 0)
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                # Need to use `_local_tensor` to patch the tensor object
                local_param = param._local_tensor
                # Monkey patch on the `torch.Tensor` as instance methods to
                # show that the extension can work even without a subclass
                local_param.fsdp_pre_all_gather = fsdp_pre_all_gather.__get__(
                    local_param
                )
                local_param.fsdp_post_all_gather = fsdp_post_all_gather.__get__(
                    local_param
                )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)

        # Run a few iterations to check for errors
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 8), device=device_type)
        for _ in range(3):
            model(inp).sum().backward()
            optim.step()
            optim.zero_grad()
        if not tls.ran_pre_all_gather:
            raise AssertionError("Expected tls.ran_pre_all_gather to be True")

    @skip_if_lt_x_gpu(1)
    def test_release_all_gather_outputs_after_post_all_gather(self):
        self.run_subtests(
            {"reshard_after_forward": [True, False]},
            self._test_release_all_gather_outputs_after_post_all_gather,
        )

    def _test_release_all_gather_outputs_after_post_all_gather(
        self, reshard_after_forward: bool
    ):
        tls = threading.local()
        tls.num_post_all_gather_calls = 0

        def fsdp_pre_all_gather(
            self,
            mesh: DeviceMesh,
            outer_size: torch.Size,
            outer_stride: tuple[int, ...],
            module: nn.Module,
            mp_policy: MixedPrecisionPolicy,
        ) -> tuple[tuple[torch.Tensor, ...], Any]:
            del mesh, outer_size, outer_stride, module, mp_policy
            return (self,), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            self,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
            del self
            if metadata is not None:
                raise AssertionError(f"Expected metadata to be None, got {metadata}")
            (tensor,) = all_gather_outputs
            if tensor.dtype != param_dtype:
                raise AssertionError(
                    f"Expected tensor dtype {param_dtype}, got {tensor.dtype}"
                )
            tls.num_post_all_gather_calls += 1
            if out is not None:
                with _unsafe_preserve_version_counter(out):
                    out.copy_(tensor)
                return None
            transformed = tensor.clone()
            return transformed, (transformed,)

        def fsdp_should_release_all_gather_outputs_after_post_all_gather(
            self,
        ) -> bool:
            del self
            return True

        test_device = self.device

        class InspectLinear(nn.Linear):
            def __init__(self) -> None:
                super().__init__(8, 8, device=test_device)
                self.storage_observations: list[
                    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
                ] = []

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                state = fully_shard.state(self)
                param_group = state._fsdp_param_group
                if param_group is None:
                    raise AssertionError("Expected an FSDP parameter group")
                fsdp_params = {
                    fsdp_param._module_info.param_name: fsdp_param
                    for fsdp_param in param_group.fsdp_params
                }
                weight = fsdp_params["weight"]
                bias = fsdp_params["bias"]
                self.storage_observations.append(
                    (
                        tuple(
                            tensor.untyped_storage().size()
                            for tensor in weight.all_gather_outputs
                        ),
                        tuple(
                            tensor.untyped_storage().size()
                            for tensor in weight._unsharded_inner_tensors
                        ),
                        tuple(
                            tensor.untyped_storage().size()
                            for tensor in bias.all_gather_outputs
                        ),
                    )
                )
                return super().forward(input)

        model = InspectLinear()
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        local_weight = model.weight._local_tensor
        local_weight.fsdp_pre_all_gather = fsdp_pre_all_gather.__get__(local_weight)
        local_weight.fsdp_post_all_gather = fsdp_post_all_gather.__get__(local_weight)
        local_weight.fsdp_should_release_all_gather_outputs_after_post_all_gather = (
            fsdp_should_release_all_gather_outputs_after_post_all_gather.__get__(
                local_weight
            )
        )

        inp = torch.randn((2, 8), device=device_type)
        for _ in range(2):
            output = model(inp)
            weight_outputs, weight_inner_tensors, bias_outputs = (
                model.storage_observations[-1]
            )
            self.assertTrue(all(size == 0 for size in weight_outputs))
            self.assertTrue(all(size > 0 for size in weight_inner_tensors))
            self.assertTrue(all(size > 0 for size in bias_outputs))

            output.sum().backward()
            state = fully_shard.state(model)
            param_group = state._fsdp_param_group
            if param_group is None:
                raise AssertionError("Expected an FSDP parameter group")
            for fsdp_param in param_group.fsdp_params:
                self.assertTrue(
                    all(
                        tensor.untyped_storage().size() == 0
                        for tensor in fsdp_param.all_gather_outputs
                    )
                )
                self.assertTrue(
                    all(
                        tensor.untyped_storage().size() == 0
                        for tensor in fsdp_param._unsharded_inner_tensors
                    )
                )

        expected_post_all_gather_calls = 4 if reshard_after_forward else 2
        self.assertEqual(tls.num_post_all_gather_calls, expected_post_all_gather_calls)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_outer_size_stride(self):
        """
        NOTE: We cannot easily test the incorrect case where the user-defined
        ``fsdp_pre_all_gather`` does not correctly pad the local tensor because
        only some ranks may require padding, in which case only those ranks
        will error out and the all-gather will timeout.
        """
        if self.world_size < 2:
            raise AssertionError(
                f"Assumes world size of at least 2 but got {self.world_size=}"
            )
        model = MLP(dim=3, dim_multiplier=3)
        for module in model.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if "weight" in param_name:
                    param = nn.Parameter(BFloat16AllGatherTensor(param))
                    setattr(module, param_name, param)
        # need to fix reshard_after_forward=True
        # https://github.com/pytorch/pytorch/issues/154836
        fully_shard(model, reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2, fused=True)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, 3), device=device_type)
        loss = model(inp).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_hsdp_mesh(self):
        tls = threading.local()
        replicate_size = 2
        shard_size = self.world_size // replicate_size
        mesh = init_device_mesh(
            device_type.type,
            (replicate_size, shard_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )

        def fsdp_pre_all_gather(
            self,
            mesh: DeviceMesh,
            outer_size: torch.Size,
            outer_stride: tuple[int, ...],
            module: nn.Module,
            mp_policy: MixedPrecisionPolicy,
        ) -> tuple[tuple[torch.Tensor, ...], Any]:
            nonlocal tls
            tls.mesh = mesh
            return (self,), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            self,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
            (tensor,) = all_gather_outputs
            if out is not None:
                return
            return tensor, (tensor,)

        model = self._init_two_tensor_mlp()
        for mlp in model:
            fully_shard(mlp, mesh=mesh)
        fully_shard(model, mesh=mesh)
        self.assertGreater(sum("weight" in n for n, _ in model.named_parameters()), 0)
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                # Need to use `_local_tensor` to patch the tensor object
                local_param = param._local_tensor
                # Monkey patch on the `torch.Tensor` as instance methods to
                # show that the extension can work even without a subclass
                local_param.fsdp_pre_all_gather = fsdp_pre_all_gather.__get__(
                    local_param
                )
                local_param.fsdp_post_all_gather = fsdp_post_all_gather.__get__(
                    local_param
                )

        inp = torch.randn((2, 8), device=device_type)
        model(inp)
        # Check that FSDP passes only the shard mesh to the pre-all-gather
        self.assertEqual(tls.mesh.ndim, 1)
        self.assertEqual(tls.mesh.size(), shard_size)


if __name__ == "__main__":
    run_tests()
