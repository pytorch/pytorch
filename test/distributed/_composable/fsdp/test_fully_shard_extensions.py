# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import math
import re
import threading
import weakref
from typing import Any

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.autograd.grad_mode import _unsafe_preserve_version_counter
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    _get_all_gather_output_layouts,
    ShardedState,
)
from torch.distributed.fsdp.experimental import AllGatherInput
from torch.distributed.tensor import Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor


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


class TestFullyShardAllGatherExtensionsMultiProcess(
    TestFullyShardAllGatherExtensionsCommon, FSDPTest
):
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


def _finish_post_all_gather(
    weight: torch.Tensor, out: torch.Tensor | None
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
    if out is None:
        return weight, (weight,)
    with _unsafe_preserve_version_counter(out):
        out.copy_(weight)
    return None


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

    def _patch_all_gather_extension(self, module: nn.Module, pre_fn, post_fn):
        local_weight = module.weight._local_tensor
        local_weight.fsdp_pre_all_gather = pre_fn.__get__(local_weight)
        local_weight.fsdp_post_all_gather = post_fn.__get__(local_weight)

    def _init_hsdp_mesh(self, shard_world_size: int) -> DeviceMesh:
        return init_device_mesh(
            device_type.type,
            (self.world_size // shard_world_size, shard_world_size),
            mesh_dim_names=("replicate", "shard"),
        )

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_single_rank_payloads(self):
        # A size-1 shard mesh skips the all-gather, so FSDP copies each payload
        # directly, including byte views into typed cached outputs
        mesh = self._init_hsdp_mesh(1)["shard"]
        byte_payloads = False
        offset = 1

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            payloads = (local_tensor, (local_tensor + offset).to(torch.bfloat16))
            if byte_payloads:
                payloads = tuple(t.view(torch.uint8) for t in payloads)
            return payloads, outer_size

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            # Typed outputs take the trailing dims of the byte payloads
            sizes = [(2, 64), (4, 32)] if byte_payloads else [(8, 16), (8, 16)]
            self.assertEqual([t.shape for t in all_gather_outputs], sizes)
            weight, auxiliary = (t.view(metadata) for t in all_gather_outputs)
            self.assertEqual(auxiliary, (weight + offset).to(torch.bfloat16))
            return _finish_post_all_gather(weight, out)

        model = nn.Linear(16, 8, bias=False, device=device_type)
        ref_model = copy.deepcopy(model)
        fully_shard(model, mesh=mesh)
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.randn((2, 16), device=device_type)
        for iteration in range(2):
            # A new offset per iteration so that stale outputs fail the check
            byte_payloads, offset = iteration == 1, iteration + 1
            output = model(inp)
            ref_output = ref_model(inp)
            self.assertEqual(output, ref_output)
            output.sum().backward()
            ref_output.sum().backward()
            check_sharded_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_zero_size_trailing_dim(self):
        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            empty = local_tensor.new_empty(local_tensor.size(0), 0)
            return (local_tensor, empty), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            weight, empty = all_gather_outputs
            self.assertEqual(empty.size(), (weight.size(0), 0))
            return _finish_post_all_gather(weight, out)

        model = nn.Linear(16, 8, bias=False, device=device_type)
        # Seeding is per process, not per thread
        dist.broadcast(model.weight.detach(), src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(model)
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.randn((2, 16), device=device_type)
        # The second all-gather reuses the outputs and passes out=
        for _ in range(2):
            output = model(inp)
            self.assertEqual(output, ref_model(inp))
            output.sum().backward()

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_errors_name_param(self):
        num_payloads = 1

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            return (local_tensor,) * num_payloads, None

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            return _finish_post_all_gather(all_gather_outputs[0], out)

        model = nn.Sequential(nn.Linear(16, 8, bias=False, device=device_type))
        fully_shard(model)
        self._patch_all_gather_extension(
            model[0], fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.randn((2, 16), device=device_type)
        # The second all-gather caches the layouts, so the third must revalidate
        for _ in range(2):
            model(inp).sum().backward()
        num_payloads = 2
        with self.assertRaisesRegex(
            ValueError,
            "fsdp_pre_all_gather for 0.weight returned 2 inputs, but 1 on its first",
        ):
            model(inp)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_uneven_sharding(self):
        # A (9, 16) weight over 8 ranks pads every shard to (2, 16)
        def init_model():
            model = nn.Linear(16, 9, bias=False, device=device_type)
            dist.broadcast(model.weight.detach(), src=0)
            return model, copy.deepcopy(model)

        def pad(local_tensor):
            padded = local_tensor.new_zeros(2, 16)
            padded[: local_tensor.size(0)] = local_tensor
            return padded

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            return (pad(local_tensor).view(-1),), None

        def legacy_fsdp_pre_all_gather(local_tensor, mesh):
            ones = torch.ones(1, device=local_tensor.device)
            return (pad(local_tensor), ones), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            weight, ones = all_gather_outputs
            self.assertEqual(ones, torch.ones_like(ones))
            # A copy, since the unpadded rows alone cannot own the output's storage
            return _finish_post_all_gather(weight[:9].clone(), out)

        # Flat inputs lack the padded size on every rank, so no rank enters the
        # collective
        model, _ = init_model()
        fully_shard(model)
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.randn((2, 16), device=device_type)
        msg = "weight is unevenly sharded, so its Tensor all-gather inputs must have the padded sharded size (2, 16), but input 0 has size (32,)"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            model(inp)
        # The legacy hook has no outer_size to pad with, so FSDP does not check it
        model, ref_model = init_model()
        fully_shard(model)
        self._patch_all_gather_extension(
            model, legacy_fsdp_pre_all_gather, fsdp_post_all_gather
        )
        for _ in range(2):
            output = model(inp)
            self.assertEqual(output, ref_model(inp))
            output.sum().backward()

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_layout_cache(self):
        # FSDP caches a parameter's layouts per combination of its inputs' kinds,
        # dims, sizes and dtypes. Shard(1) reassembles the rows of a plain tensor
        # but concatenates a dim-0 record of the same size.
        payload = "tensor"
        payloads = {
            # Each rank's (8, 2) shard of the (8, 16) weight
            "tensor": lambda t: t,
            "dim0": lambda t: AllGatherInput(t),
            "dim1": lambda t: AllGatherInput(t, dim=1),
            "flat": lambda t: AllGatherInput(t.view(1, 16), dim=1),
            "column": lambda t: AllGatherInput(t.view(16, 1)),
            "float_dim": lambda t: AllGatherInput(t, dim=1.0),
            "int32": lambda t: AllGatherInput(t.view(torch.int32), dim=1),
        }
        # Output shapes, and whether the outputs hold the ranks' shards in turn
        outputs = {
            "tensor": ((64, 2), False),
            "dim0": ((64, 2), True),
            "dim1": ((8, 16), False),
            "flat": ((1, 128), True),
            "column": ((128, 1), True),
        }

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            return (payloads[payload](local_tensor),), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            (output,) = all_gather_outputs
            shape, rank_major = outputs[payload]
            self.assertEqual(output.shape, shape)
            weight = output.view(8, 8, 2).transpose(0, 1) if rank_major else output
            return _finish_post_all_gather(weight.reshape(8, 16), out)

        model = nn.Linear(16, 8, bias=False, device=device_type)
        dist.broadcast(model.weight.detach(), src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(
            model, reshard_after_forward=True, shard_placement_fn=lambda _: Shard(1)
        )
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        fsdp_param = fully_shard.state(model)._fsdp_param_group.fsdp_params[0]
        inp = torch.randn((2, 16), device=device_type)
        # Layouts are cached once the outputs that they are checked against exist
        model(inp)
        cached_layouts = {}
        for payload in ("tensor", "dim0", "tensor", "dim1", "flat", "dim1", "dim0"):
            self.assertEqual(model(inp), ref_model(inp))
            layouts = fsdp_param.all_gather_output_layouts
            self.assertIs(cached_layouts.setdefault(payload, layouts), layouts)
        # A fifth combination exceeds the cache, so an earlier one is checked again
        for payload in ("column", "tensor"):
            self.assertEqual(model(inp), ref_model(inp))
        layouts = fsdp_param.all_gather_output_layouts
        self.assertIsNot(layouts, cached_layouts["tensor"])
        self.assertLessEqual(len(fsdp_param._extension_layouts_by_key), 4)
        # dim 1.0 is rejected also while dim 1 is cached
        payload = "dim1"
        self.assertEqual(model(inp), ref_model(inp))
        for payload, msg in (
            ("float_dim", "has dim 1.0, which is invalid"),
            ("int32", "must keep the dtype torch.float32"),
        ):
            with self.assertRaisesRegex(ValueError, msg):
                model(inp)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_payload_errors(self):
        # Inputs that FSDP rejects before the collective, after a first call with
        # each rank's (2, 8) shard of the (16, 8) weight
        msg = "AllGatherInput 0 of weight has dim 2, which is invalid for size (2, 8)"
        self._check_bad_payload(lambda t: AllGatherInput(t, dim=2), msg)
        # fsdp_post_all_gather would view the output as (-1, 3)
        msg = "All-gather input 0 of weight has trailing dims (3,), which do not divide the 128 elements of its output"
        self._check_bad_payload(lambda t: t.reshape(-1)[:6].view(2, 3), msg)

    def _check_bad_payload(self, bad_payload, msg: str):
        payload = None

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            return (payload(local_tensor) if payload else local_tensor,), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            return _finish_post_all_gather(all_gather_outputs[0], out)

        model = nn.Linear(8, 16, bias=False, device=device_type)
        fully_shard(model, reshard_after_forward=True)
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.randn((2, 8), device=device_type)
        model(inp)
        payload = bad_payload
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            model(inp)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_post_forward_reshard(self):
        # Extensions with one input reshard to a smaller mesh after forward, but
        # FSDP cannot all-gather their post-forward shards. Those with more fail
        # before any parameter of their group is resharded.
        num_inputs = 1

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            return (local_tensor, torch.ones_like(local_tensor))[:num_inputs], None

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            return _finish_post_all_gather(all_gather_outputs[0], out)

        inp = torch.randn((2, 8), device=device_type)
        all_gather_msg = "All-gathering the post-forward shard of 1.weight is not supported for all-gather extensions yet"
        reshard_msg = "Resharding 1.weight to smaller mesh is not supported for all-gather extensions with 2 inputs yet"
        for num_inputs, msg in ((1, all_gather_msg), (2, reshard_msg)):
            model = nn.Sequential(
                nn.Linear(8, 8, bias=False, device=device_type),
                nn.Linear(8, 8, bias=False, device=device_type),
            )
            fully_shard(model, reshard_after_forward=2)
            self._patch_all_gather_extension(
                model[1], fsdp_pre_all_gather, fsdp_post_all_gather
            )
            fsdp_params = fully_shard.state(model)._fsdp_param_group.fsdp_params
            with self.assertRaisesRegex(NotImplementedError, re.escape(msg)):
                model(inp).sum().backward()
            if num_inputs == 2:
                states = [fsdp_param.sharded_state for fsdp_param in fsdp_params]
                self.assertEqual(states, [ShardedState.UNSHARDED] * 2)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_shard_dim1_byte_views(self):
        # Byte views of typed outputs are reassembled in the outputs' layout,
        # whatever their shapes
        byte_shape = None

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            if byte_shape is None:
                return (local_tensor,), None
            return (local_tensor.view(torch.uint8).view(byte_shape),), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            return _finish_post_all_gather(all_gather_outputs[0].view(8, 16), out)

        # Shard(1) of an (8, 16) weight over 8 ranks: (8, 2) fp32 shards
        model = nn.Linear(16, 8, bias=False, device=device_type)
        dist.broadcast(model.weight.detach(), src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(
            model, reshard_after_forward=True, shard_placement_fn=lambda _: Shard(1)
        )
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.randn((2, 16), device=device_type)
        for byte_shape in (None, (8, 8), (4, 16), (64,)):
            self.assertEqual(model(inp), ref_model(inp))

    @skip_if_lt_x_gpu(1)
    def test_all_gather_input_layouts(self):
        self.run_subtests(
            {
                "shard_world_size": [2, 1],
            },
            self._test_all_gather_input_layouts,
        )

    def _test_all_gather_input_layouts(self, shard_world_size: int):
        mesh = self._init_hsdp_mesh(shard_world_size)
        expected_weight = torch.arange(64, device=device_type).float().view(8, 8) / 64
        tags = torch.arange(12, dtype=torch.bfloat16, device=device_type).view(2, 3, 2)
        post_out_ids: list[int | None] = []
        payload_refs: list[weakref.ReferenceType[torch.Tensor]] = []

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
            payload_tags = tags + rank * 16
            rank_tag = torch.tensor(rank, dtype=torch.int64, device=local_tensor.device)
            payload_refs.extend((weakref.ref(payload_tags), weakref.ref(rank_tag)))
            return (
                AllGatherInput(local_tensor, dim=-1),
                # Gathered along a dim with leading dims other than the shard dim
                AllGatherInput(payload_tags, dim=2),
                AllGatherInput(rank_tag),
                # Tensor payloads follow the parameter's shard layout
                local_tensor,
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
            weight, gathered_tags, ranks, tensor_payload = all_gather_outputs
            self.assertEqual(metadata, (torch.Size((8, 8)), shard_world_size))
            self.assertEqual(weight.shape, (8, 8))
            self.assertEqual(weight.dtype, param_dtype)
            self.assertEqual(weight, expected_weight)
            self.assertEqual(gathered_tags.shape, (2, 3, 2 * shard_world_size))
            self.assertEqual(gathered_tags.dtype, torch.bfloat16)
            expected_tags = [tags + rank * 16 for rank in range(shard_world_size)]
            self.assertEqual(gathered_tags, torch.cat(expected_tags, dim=2))
            self.assertEqual(ranks.shape, (shard_world_size,))
            self.assertEqual(ranks.dtype, torch.int64)
            self.assertEqual(ranks, torch.arange(shard_world_size, device=device_type))
            self.assertEqual(tensor_payload.view(8, 8), expected_weight)
            post_out_ids.append(None if out is None else id(out))
            if out is not None:
                with _unsafe_preserve_version_counter(out):
                    out.copy_(weight)
                return None
            transformed = weight.clone()
            return transformed, (transformed,)

        model = nn.Linear(8, 8, bias=False, device=device_type)
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
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.arange(16, device=device_type).float().view(2, 8) / 16
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            ref_model.zero_grad(set_to_none=True)
            output = model(inp)
            ref_output = ref_model(inp)
            self.assertEqual(output, ref_output)
            output.sum().backward()
            ref_output.sum().backward()
            check_sharded_parity(self, ref_model, model)

        self.assertEqual(len(post_out_ids), 4)
        self.assertIsNotNone(post_out_ids[1])
        self.assertEqual(post_out_ids, [None] + [post_out_ids[1]] * 3)
        # FSDP keeps no references to the payloads of the 4 all-gathers
        self.assertEqual([ref() for ref in payload_refs], [None] * 8)

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_smaller_payload(self):
        self.run_subtests(
            {"shard_world_size": [2, 1]},
            self._test_all_gather_extension_smaller_payload,
        )

    def _test_all_gather_extension_smaller_payload(self, shard_world_size: int):
        # A Tensor payload of a Shard(0) parameter may shrink after its first
        # call and then fills a prefix of its cached output, also when a size-1
        # shard mesh copies it directly. The bf16 tags make the group gather bytes.
        mesh = self._init_hsdp_mesh(shard_world_size)
        num_tags = 4

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            rank = mesh.get_local_rank()
            tags = torch.arange(num_tags, device=local_tensor.device) + 10.0 * rank
            return (local_tensor, tags.bfloat16()), (outer_size, num_tags)

        @torch.no_grad()
        def fsdp_post_all_gather(
            local_tensor, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            outer_size, num_rank_tags = metadata
            weight, tags = all_gather_outputs
            expected_tags = [
                torch.arange(num_rank_tags, device=tags.device) + 10.0 * rank
                for rank in range(shard_world_size)
            ]
            self.assertEqual(
                tags[: num_rank_tags * shard_world_size],
                torch.cat(expected_tags).bfloat16(),
            )
            return _finish_post_all_gather(weight.view(outer_size), out)

        model = nn.Linear(8, 16, bias=False, device=device_type)
        # Seeding is per process, not per thread
        dist.broadcast(model.weight.detach(), src=0)
        ref_model = copy.deepcopy(model)
        fully_shard(model, mesh=mesh)
        self._patch_all_gather_extension(
            model, fsdp_pre_all_gather, fsdp_post_all_gather
        )
        inp = torch.randn((2, 8), device=device_type)
        for num_tags in (4, 2):
            output = model(inp)
            self.assertEqual(output, ref_model(inp))
            output.sum().backward()


class TestAllGatherInputValidation(TestCase):
    # Shard(0) of an (8, 4) parameter over 2 ranks, or Shard(1) of a (4, 8) one
    # with shard_dim=1
    padded_sharded_size = torch.Size((4, 4))

    def _get_layouts(self, inputs, outputs=(), **kwargs):
        kwargs = {
            "shard_world_size": 2,
            "shard_dim": 0,
            "padded_sharded_size": self.padded_sharded_size,
            "require_padded_size": False,
            "param_name": "lin.weight",
            **kwargs,
        }
        return _get_all_gather_output_layouts(inputs, list(outputs), **kwargs)

    def test_record_dims(self):
        # Tensor dims would sync on every lookup of the cached layouts
        for dim in (2, -3, 1.0, None, torch.tensor(1)):
            msg = f"AllGatherInput 0 of lin.weight has dim {dim!r}, which is invalid for size (4, 4)"
            with self.assertRaisesRegex(ValueError, re.escape(msg)):
                self._get_layouts([AllGatherInput(torch.ones(4, 4), dim=dim)])
        # Other integers count by value, and scalars count as shape (1,)
        layouts = self._get_layouts([AllGatherInput(torch.ones(4, 4), dim=1)])
        for dim in (np.int64(1), True):
            inputs = [AllGatherInput(torch.ones(4, 4), dim=dim)]
            self.assertEqual(self._get_layouts(inputs), layouts)
        self._get_layouts([AllGatherInput(torch.ones(()), dim=-1)])

    def test_output_layouts(self):
        inputs = [
            AllGatherInput(torch.ones(2, 3, 4), dim=1),
            AllGatherInput(torch.ones(())),
            AllGatherInput(torch.ones(2, 0), dim=1),
            torch.ones(4, 4),
            torch.ones(4, 0),
        ]
        layouts = self._get_layouts(inputs, shard_dim=1)
        self.assertEqual(
            [tuple(layout) for layout in layouts],
            [((2, 6, 4), 2), ((2,), 1), ((2, 0), 1), ((-1, 4), 4), ((8, 0), 1)],
        )
        # A single rank needs no reassembly
        layouts = self._get_layouts(inputs, shard_dim=1, shard_world_size=1)
        self.assertEqual([layout.outer_size for layout in layouts], [1] * 5)

    def test_payload_count_change(self):
        with self.assertRaisesRegex(
            ValueError, "lin.weight returned 2 inputs, but 1 on its first call"
        ):
            self._get_layouts([torch.ones(4, 4)] * 2, [torch.empty(32)])

    def test_records_keep_numel_and_dtype(self):
        output = torch.empty(32)
        for tensor in (torch.ones(4, 4), torch.ones(2, 8)):
            self._get_layouts([AllGatherInput(tensor)], [output])
        msg = "input 0 of lin.weight must match the size of its first call, 16 torch.float32 per rank, but has 8 torch.float32"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            self._get_layouts([AllGatherInput(torch.ones(2, 4))], [output])
        msg = "input 0 of lin.weight must keep the dtype torch.float32 of its first call, but has dtype torch.uint8"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            byte_view = torch.ones(4, 4).view(torch.uint8)
            self._get_layouts([AllGatherInput(byte_view)], [output])

    def test_tensor_inputs_shrink_or_become_byte_views(self):
        output = torch.empty(32)
        byte_view = torch.ones(4, 4).view(torch.uint8)
        for tensor in (torch.ones(2, 4), torch.ones(0, 4), byte_view):
            self._get_layouts([tensor], [output])
        with self.assertRaisesRegex(ValueError, "must fit in the size of its first"):
            self._get_layouts([torch.ones(8, 4)], [output])
        with self.assertRaisesRegex(ValueError, "must keep the dtype torch.float32"):
            self._get_layouts([torch.ones(4, 4, dtype=torch.bfloat16)], [output])
        # Only empty outputs can be viewed with a zero trailing dim
        self._get_layouts([torch.ones(4, 0)], [torch.empty(0)])
        with self.assertRaisesRegex(ValueError, "must match the size of its first"):
            self._get_layouts([torch.ones(4, 0)], [output])
        # fsdp_post_all_gather receives each output viewed as (-1, *size[1:])
        msg = "input 0 of lin.weight has trailing dims (3,), which do not divide the 32 elements of its output"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            self._get_layouts([torch.ones(2, 3)], [output])

    def test_shard_dim_tensor_inputs_keep_rows(self):
        # The copy-out reads Shard(1) Tensor inputs as rows of the padded sharded
        # parameter, so they need its element count or leading dims, and cannot
        # shrink
        for tensor in (
            torch.ones(16),
            torch.ones(2, 8),
            torch.ones(4, 4).view(torch.uint8),
            torch.ones(4, 2, dtype=torch.uint8),  # e.g. packed 4-bit values
            torch.ones(0, 4),
        ):
            self._get_layouts([tensor], shard_dim=1)
        # e.g. bytes of a wrongly shaped payload, whose ranks would be mixed up
        msg = "Tensor all-gather input 0 of lin.weight is gathered along its Shard(1) dim, so it must have the 16 elements of the padded sharded size (4, 4) or its dims (4,) before that dim, but has size (2, 16)"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            self._get_layouts([torch.ones(2, 4).view(torch.uint8)], shard_dim=1)
        # Bytes of a typed output are reassembled in the output's layout
        output = torch.empty(32)
        byte_view = torch.ones(4, 4).view(torch.uint8)
        for tensor in (byte_view.view(-1), byte_view.view(2, 32)):
            self._get_layouts([tensor], [output], shard_dim=1)
        for tensor in (torch.ones(4, 2), torch.ones(4, 0)):
            with self.assertRaisesRegex(ValueError, "must match the size of its"):
                self._get_layouts([tensor], [output], shard_dim=1)

    def test_uneven_tensor_inputs_keep_padded_size(self):
        msg = "lin.weight is unevenly sharded, so its Tensor all-gather inputs must have the padded sharded size (4, 4), but input 0 has size (3, 4)"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            self._get_layouts([torch.ones(3, 4)], require_padded_size=True)
        self._get_layouts([AllGatherInput(torch.ones(3, 4))], require_padded_size=True)


if __name__ == "__main__":
    run_tests()
