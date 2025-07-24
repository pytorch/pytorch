# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import math
import threading
from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.autograd.grad_mode import _unsafe_preserve_version_counter
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
)
from torch.testing._internal.common_utils import run_tests
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
    out: Optional[torch.Tensor] = None,
) -> Union[tuple[torch.Tensor, tuple[torch.Tensor, ...]], None]:
    assert metadata is None, f"{metadata}"
    a, b = all_gather_outputs
    if out is not None:
        assert isinstance(out, TwoTensor), f"{type(out)}"
        if a.dtype == param_dtype:
            assert a.untyped_storage().data_ptr() == out.a.untyped_storage().data_ptr()
            assert b.untyped_storage().data_ptr() == out.b.untyped_storage().data_ptr()
        else:
            assert out.a.dtype == param_dtype, f"{out.a.dtype} {param_dtype}"
            assert out.b.dtype == param_dtype, f"{out.b.dtype} {param_dtype}"
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
        assert mesh.ndim == 1, f"{mesh.ndim}"
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
        out: Optional[torch.Tensor] = None,
    ) -> Union[tuple[torch.Tensor, tuple[torch.Tensor, ...]], None]:
        assert metadata is None, f"{metadata}"
        (tensor,) = all_gather_outputs
        assert tensor.dtype == torch.bfloat16, f"{tensor.dtype}"
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
                assert pad_in_pre_all_gather == x._pad_in_pre_all_gather
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
            tls.ran_pre_all_gather = True
            return (self.to(torch.bfloat16),), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            self,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: Optional[torch.Tensor] = None,
        ) -> Union[tuple[torch.Tensor, tuple[torch.Tensor, ...]], None]:
            (tensor,) = all_gather_outputs
            assert metadata is None, f"{metadata}"
            assert tensor.dtype == torch.bfloat16, f"{tensor.dtype}"
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
        assert tls.ran_pre_all_gather

    @skip_if_lt_x_gpu(1)
    def test_all_gather_extension_outer_size_stride(self):
        """
        NOTE: We cannot easily test the incorrect case where the user-defined
        ``fsdp_pre_all_gather`` does not correctly pad the local tensor because
        only some ranks may require padding, in which case only those ranks
        will error out and the all-gather will timeout.
        """
        assert self.world_size >= 2, (
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
            tls.mesh = mesh
            return (self,), None

        @torch.no_grad()
        def fsdp_post_all_gather(
            self,
            all_gather_outputs: tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: Optional[torch.Tensor] = None,
        ) -> Union[tuple[torch.Tensor, tuple[torch.Tensor, ...]], None]:
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
