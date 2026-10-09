# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import math
import re
import threading
import warnings
import weakref
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.autograd.grad_mode import _unsafe_preserve_version_counter
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGatherInput
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_all_gather_output_fn,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    _get_all_gather_output_layouts,
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
from torch.testing._internal.common_utils import run_tests, TestCase
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
        test = self

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
            weight, auxiliary = (t.view(metadata) for t in all_gather_outputs)
            test.assertEqual(auxiliary, (weight + offset).to(torch.bfloat16))
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
        test = self

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
            test.assertEqual(empty.size(), (weight.size(0), 0))
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
        model(inp).sum().backward()
        num_payloads = 2
        with self.assertRaisesRegex(
            ValueError,
            "fsdp_pre_all_gather for 0.weight returned 2 inputs, but 1 on its first",
        ):
            model(inp)

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
        # shard mesh copies it directly
        mesh = self._init_hsdp_mesh(shard_world_size)
        num_tags = 4
        test = self

        def fsdp_pre_all_gather(
            local_tensor, mesh, outer_size, outer_stride, module, mp_policy
        ):
            rank = mesh.get_local_rank()
            tags = torch.arange(num_tags, device=local_tensor.device) + 10.0 * rank
            return (local_tensor, tags), (outer_size, num_tags)

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
            test.assertEqual(
                tags[: num_rank_tags * shard_world_size], torch.cat(expected_tags)
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
    # Shard(1) of a (4, 8) parameter over 2 ranks
    padded_sharded_size = torch.Size((4, 4))

    def _get_layouts(self, inputs, outputs=(), **kwargs):
        tensors = [
            inp.tensor if isinstance(inp, AllGatherInput) else inp for inp in inputs
        ]
        kwargs = {
            "shard_world_size": 2,
            "shard_dim": 0,
            "padded_sharded_size": self.padded_sharded_size,
            "unevenly_sharded": False,
            "param_name": "lin.weight",
            **kwargs,
        }
        return _get_all_gather_output_layouts(inputs, tensors, list(outputs), **kwargs)

    def test_dim_out_of_range(self):
        for dim in (2, -3):
            with self.assertRaisesRegex(ValueError, f"dim {dim} is invalid"):
                AllGatherInput(torch.ones(4, 4), dim=dim)
        # Scalars count as shape (1,)
        AllGatherInput(torch.ones(()), dim=-1)

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

    def test_records_keep_size_and_dtype(self):
        output = torch.empty(32)
        self._get_layouts([AllGatherInput(torch.ones(4, 4))], [output])
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

    def test_shard_dim_tensor_inputs_keep_rows(self):
        # The copy-out reassembles the rows of Shard(1) Tensor inputs, so they
        # need the padded sharded size or its leading dims, and cannot shrink
        for tensor in (
            torch.ones(16),
            torch.ones(4, 4).view(torch.uint8),
            torch.ones(4, 2, dtype=torch.uint8),  # e.g. packed 4-bit values
        ):
            self._get_layouts([tensor], shard_dim=1)
        # e.g. bytes of a wrongly shaped payload, whose ranks would be mixed up
        msg = "Tensor all-gather input 0 of lin.weight must have the padded sharded size (4, 4) or its dims before the Shard(1) dim, but has size (2, 16)"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            self._get_layouts([torch.ones(2, 4).view(torch.uint8)], shard_dim=1)
        for tensor in (torch.ones(4, 2), torch.ones(4, 0)):
            with self.assertRaisesRegex(ValueError, "must match the size of its"):
                self._get_layouts([tensor], [torch.empty(32)], shard_dim=1)

    def test_uneven_tensor_inputs_keep_padded_size(self):
        msg = "lin.weight is unevenly sharded, so its Tensor all-gather inputs must have the padded sharded size (4, 4), but input 0 has size (3, 4)"
        with self.assertRaisesRegex(ValueError, re.escape(msg)):
            self._get_layouts([torch.ones(3, 4)], unevenly_sharded=True)
        self._get_layouts([AllGatherInput(torch.ones(3, 4))], unevenly_sharded=True)


class TestDefaultAllGatherOutputFn(TestCase):
    def test_smaller_payload_fills_prefix(self):
        # Two ranks gather two elements each into an output sized for four each
        output = torch.full((8,), -1.0)
        with warnings.catch_warnings():
            # e.g. the deprecated resize of a mismatched out=
            warnings.simplefilter("error")
            _default_all_gather_output_fn(torch.arange(4.0), [output], [2], [1], 2)
        self.assertEqual(output, torch.tensor([0.0, 1, 2, 3, -1, -1, -1, -1]))

    def test_byte_buffer_reassembles_typed_rows(self):
        # Mixed dtypes gather bytes: per rank, a (2, 2) chunk of a parameter
        # gathered along dim 1, then two bf16 tags
        chunks = [torch.arange(4.0).view(2, 2) + 10 * rank for rank in range(2)]
        tags = [torch.full((2,), rank, dtype=torch.bfloat16) for rank in range(2)]
        all_gather_output = torch.cat(
            [t.view(torch.uint8).view(-1) for pair in zip(chunks, tags) for t in pair]
        )
        outputs = [torch.empty(8), torch.empty(4, dtype=torch.bfloat16)]
        cat_dtypes: list[torch.dtype] = []

        class RecordCatDtypes(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func == torch.ops.aten.cat.out:
                    cat_dtypes.append(kwargs["out"].dtype)
                return func(*args, **kwargs)

        with RecordCatDtypes():
            _default_all_gather_output_fn(
                all_gather_output, outputs, [16, 4], [2, 1], 2
            )
        self.assertEqual(outputs[0].view(2, 4), torch.cat(chunks, dim=1))
        self.assertEqual(outputs[1], torch.cat(tags))
        # Typed rows get vectorized cat kernels that bytes may not
        self.assertEqual(cat_dtypes, [torch.float32])


if __name__ == "__main__":
    run_tests()
