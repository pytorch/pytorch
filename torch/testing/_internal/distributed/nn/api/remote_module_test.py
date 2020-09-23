#!/usr/bin/python3
import enum
from typing import Tuple

import torch
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch import Tensor, nn
from torch._jit_internal import Future
from torch.distributed.nn import RemoteModule
from torch.distributed.nn.api.remote_module import _RemoteModule
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


_PARAM_VAL = torch.nn.Parameter(torch.ones(1))


# RPC handler for querying the device on the destination worker.
def remote_device(module_rref):
    for param in module_rref.local_value().parameters():
        return param.device


class ModuleCreationMode(enum.Enum):
    MODULE_CTOR_WITH_INTERFACE = "module_ctor_with_interface"
    MODULE_CTOR = "module_ctor"


@torch.jit.interface
class MyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        pass


@torch.jit.interface
class RemoteMyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        pass

    def forward_async(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Future[Tuple[str, int, Tensor]]:
        pass


class MyModule(nn.Module):
    def __init__(self, first_arg, first_kwarg=-1):
        super().__init__()
        self.param1 = _PARAM_VAL

    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        return word, number, tensor


class BadModule:
    def __init__(self, first_arg, first_kwarg=-1):
        pass


def create_scripted_module(first_arg, first_kwarg=-1):
    module = MyModule(first_arg, first_kwarg=first_kwarg)
    scripted_module = torch.jit.script(module)
    return scripted_module


class RemoteModuleTest(RpcAgentTestFixture):
    @property
    def world_size(self):  # Override setting in RpcAgentTestFixture
        return 2

    @staticmethod
    def _create_remote_module_iter(dst_worker_name, device="cpu", modes=None):
        if modes is None:
            modes = ModuleCreationMode.__members__.values()

        args = (1,)
        kwargs = dict(first_kwarg=2)

        if ModuleCreationMode.MODULE_CTOR in modes:
            remote_module = RemoteModule(
                dst_worker_name, device, MyModule, args, kwargs
            )
            yield remote_module

        if ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE in modes:
            remote_module = _RemoteModule(
                dst_worker_name,
                device,
                create_scripted_module,
                args,
                kwargs,
                _module_interface_cls=MyModuleInterface,
            )
            scripted_remote_module = torch.jit.script(remote_module)
            yield scripted_remote_module

    @dist_utils.dist_init
    def test_bad_module(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (1,)
        kwargs = dict(first_kwarg=2)

        with self.assertRaisesRegex(
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            RemoteModule(dst_worker_name, "cpu", BadModule, args, kwargs)

        with self.assertRaisesRegex(
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            RemoteModule(dst_worker_name, "cpu", BadModule, args, kwargs)

    @dist_utils.dist_init
    def test_forward_async(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2, "3")
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret_fut = remote_module.forward_async(*args)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_forward_async_script(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        scripted_remote_module = next(
            self._create_remote_module_iter(
                dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
            )
        )

        @torch.jit.script
        def run_forward_async(scripted_remote_module: RemoteMyModuleInterface):
            ret_fut = scripted_remote_module.forward_async(torch.ones(1), 2, "3")
            ret = ret_fut.wait()
            return ret

        ret = run_forward_async(scripted_remote_module)

        self.assertEqual(ret, ("3", 2, torch.ones(1)))

    @dist_utils.dist_init
    def test_forward_sync(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2, "3")
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret = remote_module.forward(*args)
            self.assertEqual(ret, tuple(reversed(args)))

    @dist_utils.dist_init
    def test_forward_sync_script(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        scripted_remote_module = next(
            self._create_remote_module_iter(
                dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
            )
        )

        @torch.jit.script
        def run_forward(scripted_remote_module: MyModuleInterface):
            ret = scripted_remote_module.forward(torch.ones(1), 2, "3")
            return ret

        ret = run_forward(scripted_remote_module)

        self.assertEqual(ret, ("3", 2, torch.ones(1)))

    @dist_utils.dist_init
    def test_forward_with_kwargs(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
        args = (torch.ones(1), 2)
        kwargs = dict(word="3")
        # Only test Python nn.Module, because script module methods don't support taking kwargs.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            ret_fut = remote_module.forward_async(*args, **kwargs)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args + ("3",))))

            ret = remote_module.forward(*args, **kwargs)
            self.assertEqual(ret, tuple(reversed(args + ("3",))))

    @dist_utils.dist_init
    def test_remote_parameters(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        # Only test Python nn.Module, because script module methods don't support ``remote_parameters``.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            param_rrefs = remote_module.remote_parameters()
            self.assertEqual(len(param_rrefs), 1)
            self.assertTrue(torch.equal(param_rrefs[0].to_here(), _PARAM_VAL))

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_valid_device(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        for remote_module in self._create_remote_module_iter(
            dst_worker_name, device="cuda:0", modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            device = rpc.rpc_sync(
                dst_worker_name, remote_device, (remote_module.module_rref,)
            )
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, 0)

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_invalid_devices(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected one of cpu, cuda, mkldnn, opengl, opencl, ideep, hip, msnpu, xla device type at start of device string",
        ):
            list(
                self._create_remote_module_iter(
                    dst_worker_name,
                    device="foo",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(
            RuntimeError, r"CUDA error: invalid device ordinal"
        ):
            list(
                self._create_remote_module_iter(
                    dst_worker_name,
                    device="cuda:100",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(RuntimeError, r"Invalid device string: 'cpu2'"):
            list(
                self._create_remote_module_iter(
                    dst_worker_name,
                    modes=[ModuleCreationMode.MODULE_CTOR],
                    device="cpu2",
                )
            )

        with self.assertRaisesRegex(
            RuntimeError, r"CPU device index must be -1 or zero, got 2"
        ):
            list(
                self._create_remote_module_iter(
                    dst_worker_name,
                    device="cpu:2",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

    @dist_utils.dist_init
    def test_unsupported_methods(self):
        if self.rank != 0:
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)

        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            with self.assertRaisesRegex(
                ValueError, r"Method ``register_buffer`` not supported for RemoteModule"
            ):
                remote_module.register_buffer("buffer", torch.ones(5))
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_parameter`` not supported for RemoteModule",
            ):
                remote_module.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(1))
                )
            with self.assertRaisesRegex(
                ValueError, r"Method ``add_module`` not supported for RemoteModule"
            ):
                remote_module.add_module("empty", None)

            with self.assertRaisesRegex(
                ValueError, r"Method ``apply`` not supported for RemoteModule"
            ):
                fn = torch.rand((3, 3), requires_grad=False)
                remote_module.apply(fn)

            with self.assertRaisesRegex(
                ValueError, r"Method ``cuda`` not supported for RemoteModule"
            ):
                remote_module.cuda()
            with self.assertRaisesRegex(
                ValueError, r"Method ``cpu`` not supported for RemoteModule"
            ):
                remote_module.cpu()
            with self.assertRaisesRegex(
                ValueError, r"Method ``type`` not supported for RemoteModule"
            ):
                remote_module.type(torch.FloatTensor)
            with self.assertRaisesRegex(
                ValueError, r"Method ``float`` not supported for RemoteModule"
            ):
                remote_module.float()
            with self.assertRaisesRegex(
                ValueError, r"Method ``double`` not supported for RemoteModule"
            ):
                remote_module.double()
            with self.assertRaisesRegex(
                ValueError, r"Method ``bfloat16`` not supported for RemoteModule"
            ):
                remote_module.bfloat16()
            with self.assertRaisesRegex(
                ValueError, r"Method ``to`` not supported for RemoteModule"
            ):
                remote_module.to("cpu", dtype=torch.int32)

            def hook(module, grad_input, grad_output):
                pass

            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_backward_hook`` not supported for RemoteModule",
            ):
                remote_module.register_backward_hook(hook)
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_forward_pre_hook`` not supported for RemoteModule",
            ):
                remote_module.register_forward_pre_hook(hook)
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``register_forward_hook`` not supported for RemoteModule",
            ):
                remote_module.register_forward_hook(hook)

            with self.assertRaisesRegex(
                ValueError, r"Method ``state_dict`` not supported for RemoteModule"
            ):
                remote_module.state_dict()
            with self.assertRaisesRegex(
                ValueError, r"Method ``load_state_dict`` not supported for RemoteModule"
            ):
                remote_module.load_state_dict({})

            with self.assertRaisesRegex(
                ValueError,
                r"Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead.",
            ):
                remote_module.parameters()
            with self.assertRaisesRegex(
                ValueError,
                r"Method ``named_parameters`` not supported for RemoteModule",
            ):
                remote_module.named_parameters()
            with self.assertRaisesRegex(
                ValueError, r"Method ``buffers`` not supported for RemoteModule"
            ):
                remote_module.buffers()
            with self.assertRaisesRegex(
                ValueError, r"Method ``named_buffers`` not supported for RemoteModule"
            ):
                remote_module.named_buffers()
            with self.assertRaisesRegex(
                ValueError, r"Method ``children`` not supported for RemoteModule"
            ):
                remote_module.children()
            with self.assertRaisesRegex(
                ValueError, r"Method ``named_children`` not supported for RemoteModule"
            ):
                remote_module.named_children()
            with self.assertRaisesRegex(
                ValueError, r"Method ``modules`` not supported for RemoteModule"
            ):
                remote_module.modules()
            with self.assertRaisesRegex(
                ValueError, r"Method ``named_modules`` not supported for RemoteModule"
            ):
                remote_module.named_modules()

            with self.assertRaisesRegex(
                ValueError, r"Method ``train`` not supported for RemoteModule"
            ):
                remote_module.train()
            with self.assertRaisesRegex(
                ValueError, r"Method ``eval`` not supported for RemoteModule"
            ):
                remote_module.eval()
            with self.assertRaisesRegex(
                ValueError, r"Method ``requires_grad_`` not supported for RemoteModule"
            ):
                remote_module.requires_grad_()
            with self.assertRaisesRegex(
                ValueError, r"Method ``zero_grad`` not supported for RemoteModule"
            ):
                remote_module.zero_grad()
            with self.assertRaisesRegex(
                ValueError, r"Method ``share_memory`` not supported for RemoteModule"
            ):
                remote_module.share_memory()
            with self.assertRaisesRegex(
                ValueError, r"Method ``extra_repr`` not supported for RemoteModule"
            ):
                remote_module.extra_repr()
