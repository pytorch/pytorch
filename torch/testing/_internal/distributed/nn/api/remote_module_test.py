#!/usr/bin/python3
import enum
import unittest
from typing import Tuple

import torch
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch import Tensor, nn
from torch._jit_internal import Future
from torch.distributed.nn import RemoteModule
from torch.distributed.nn.api.remote_module import _REMOTE_MODULE_PICKLED_ATTRIBUTES
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


# RPC handler for querying __dict__ on the destination worker.
def remote_module_attributes(remote_module):
    return remote_module.__dict__


# RPC handler for running forward on the destination worker.
def remote_forward(remote_module, args):
    return remote_module.forward(*args)


# RPC handler for running forward_async on the destination worker.
def remote_forward_async(remote_module, args):
    # Since future cannot be pickled and sent over the RPC layer,
    # have to wait and behave just like ``forward_sync``.
    return remote_module.forward_async(*args).wait()


class ModuleCreationMode(enum.Enum):
    MODULE_CTOR_WITH_INTERFACE = "module_ctor_with_interface"
    MODULE_CTOR = "module_ctor"


@torch.jit.interface
class MyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
        pass


@torch.jit.interface
class RemoteMyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[str, int, Tensor]:
        # pyre-ignore[7]: Pyre and torch.jit.interface don't mix well
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


# Common utils for both CPU and CUDA test suites
class CommonRemoteModuleTest(RpcAgentTestFixture):
    @property
    def world_size(self):  # Override setting in RpcAgentTestFixture
        return 2

    @staticmethod
    def _create_remote_module_iter(remote_device, modes=None):
        if modes is None:
            modes = ModuleCreationMode.__members__.values()

        args = (1,)
        kwargs = dict(first_kwarg=2)

        if ModuleCreationMode.MODULE_CTOR in modes:
            remote_module = RemoteModule(remote_device, MyModule, args, kwargs)  # type: ignore[arg-type]
            yield remote_module

        if ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE in modes:
            remote_module = _RemoteModule(  # type: ignore[assignment]
                remote_device,
                create_scripted_module,  # type: ignore[arg-type]
                args,
                kwargs,
                _module_interface_cls=MyModuleInterface,
            )
            scripted_remote_module = torch.jit.script(remote_module)
            yield scripted_remote_module


class RemoteModuleTest(CommonRemoteModuleTest):
    @dist_utils.dist_init
    def test_bad_module(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]
        remote_device = "{}/cpu".format(dst_worker_name)
        args = (1,)
        kwargs = dict(first_kwarg=2)

        with self.assertRaisesRegex(  # type: ignore[attr-defined]
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            RemoteModule(remote_device, BadModule, args, kwargs)  # type: ignore[arg-type]

        with self.assertRaisesRegex(  # type: ignore[attr-defined]
            ValueError,
            r"Expect `module_cls\(\*args, \*\*kwargs\)` returns an instance of <class nn.Module>,",
        ):
            RemoteModule(remote_device, BadModule, args, kwargs)  # type: ignore[arg-type]

    @dist_utils.dist_init
    def test_forward_async(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]
        args = (torch.ones(1), 2, "3")
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret_fut = remote_module.forward_async(*args)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args)))  # type: ignore[attr-defined]

    @dist_utils.dist_init
    def test_forward_async_script(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

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

        self.assertEqual(ret, ("3", 2, torch.ones(1)))  # type: ignore[attr-defined]

    @dist_utils.dist_init
    def test_forward_sync(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]
        args = (torch.ones(1), 2, "3")
        for remote_module in self._create_remote_module_iter(dst_worker_name):
            ret = remote_module.forward(*args)
            self.assertEqual(ret, tuple(reversed(args)))  # type: ignore[attr-defined]

    @dist_utils.dist_init
    def test_forward_sync_script(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

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

        self.assertEqual(ret, ("3", 2, torch.ones(1)))  # type: ignore[attr-defined]

    @dist_utils.dist_init
    def test_forward_with_kwargs(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]
        args = (torch.ones(1), 2)
        kwargs = dict(word="3")
        # Only test Python nn.Module, because script module methods don't support taking kwargs.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            ret_fut = remote_module.forward_async(*args, **kwargs)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args + ("3",))))  # type: ignore[attr-defined]

            ret = remote_module.forward(*args, **kwargs)
            self.assertEqual(ret, tuple(reversed(args + ("3",))))  # type: ignore[attr-defined]

    @dist_utils.dist_init
    def test_remote_parameters(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        # Only test Python nn.Module, because script module methods don't support ``remote_parameters``.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            param_rrefs = remote_module.remote_parameters()
            self.assertEqual(len(param_rrefs), 1)  # type: ignore[attr-defined]
            self.assertTrue(torch.equal(param_rrefs[0].to_here(), _PARAM_VAL))  # type: ignore[attr-defined]

    @dist_utils.dist_init
    def test_get_module_rref(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        # Only test Python nn.Module, because script module methods don't support ``get_module_rref``.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            rref = remote_module.get_module_rref()
            self.assertEqual(rref, remote_module.module_rref)  # type: ignore[attr-defined]
            for param in rref.to_here().parameters():
                self.assertTrue(torch.equal(param, _PARAM_VAL))  # type: ignore[attr-defined]

    @dist_utils.dist_init
    def test_unsupported_methods(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``register_buffer`` not supported for RemoteModule"
            ):
                remote_module.register_buffer("buffer", torch.ones(5))
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError,
                r"Method ``register_parameter`` not supported for RemoteModule",
            ):
                remote_module.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(1))
                )
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``add_module`` not supported for RemoteModule"
            ):
                remote_module.add_module("empty", None)

            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``apply`` not supported for RemoteModule"
            ):
                fn = torch.rand((3, 3), requires_grad=False)
                remote_module.apply(fn)

            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``cuda`` not supported for RemoteModule"
            ):
                remote_module.cuda()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``cpu`` not supported for RemoteModule"
            ):
                remote_module.cpu()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``type`` not supported for RemoteModule"
            ):
                remote_module.type(torch.FloatTensor)
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``float`` not supported for RemoteModule"
            ):
                remote_module.float()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``double`` not supported for RemoteModule"
            ):
                remote_module.double()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``bfloat16`` not supported for RemoteModule"
            ):
                remote_module.bfloat16()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``to`` not supported for RemoteModule"
            ):
                remote_module.to("cpu", dtype=torch.int32)

            def hook(module, grad_input, grad_output):
                pass

            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError,
                r"Method ``register_backward_hook`` not supported for RemoteModule",
            ):
                remote_module.register_backward_hook(hook)
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError,
                r"Method ``register_forward_pre_hook`` not supported for RemoteModule",
            ):
                remote_module.register_forward_pre_hook(hook)
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError,
                r"Method ``register_forward_hook`` not supported for RemoteModule",
            ):
                remote_module.register_forward_hook(hook)

            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``state_dict`` not supported for RemoteModule"
            ):
                remote_module.state_dict()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``load_state_dict`` not supported for RemoteModule"
            ):
                remote_module.load_state_dict({})

            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError,
                r"Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead.",
            ):
                remote_module.parameters()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError,
                r"Method ``named_parameters`` not supported for RemoteModule",
            ):
                remote_module.named_parameters()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``buffers`` not supported for RemoteModule"
            ):
                remote_module.buffers()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``named_buffers`` not supported for RemoteModule"
            ):
                remote_module.named_buffers()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``children`` not supported for RemoteModule"
            ):
                remote_module.children()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``named_children`` not supported for RemoteModule"
            ):
                remote_module.named_children()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``modules`` not supported for RemoteModule"
            ):
                remote_module.modules()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``named_modules`` not supported for RemoteModule"
            ):
                remote_module.named_modules()

            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``train`` not supported for RemoteModule"
            ):
                remote_module.train()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``eval`` not supported for RemoteModule"
            ):
                remote_module.eval()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``requires_grad_`` not supported for RemoteModule"
            ):
                remote_module.requires_grad_()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``zero_grad`` not supported for RemoteModule"
            ):
                remote_module.zero_grad()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``share_memory`` not supported for RemoteModule"
            ):
                remote_module.share_memory()
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                ValueError, r"Method ``extra_repr`` not supported for RemoteModule"
            ):
                remote_module.extra_repr()

    @dist_utils.dist_init
    def test_send_remote_module_with_a_new_attribute_over_the_wire(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        # If add a new attribute is added to this RemoteModule, which will be sent over the wire by RPC,
        # this new field must be added to either _REMOTE_MODULE_PICKLED_ATTRIBUTES or _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING
        # to avoid runtime error.
        for remote_module in self._create_remote_module_iter(
            dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            new_attr_name = "new_attr"
            setattr(remote_module, new_attr_name, 1)
            with self.assertRaisesRegex(  # type: ignore[attr-defined]
                RuntimeError,
                "Attribute ``{}`` of RemoteModule must be either in "
                "``_REMOTE_MODULE_PICKLED_ATTRIBUTES``  or ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``".format(
                    new_attr_name
                ),
            ):
                rpc.rpc_sync(
                    dst_worker_name, remote_module_attributes, (remote_module,)
                )


class ThreeWorkersRemoteModuleTest(CommonRemoteModuleTest):
    @property
    def world_size(self):  # Override setting in CommonRemoteModuleTest
        return 3

    @dist_utils.dist_init
    def test_send_remote_module_over_the_wire(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)  # type: ignore[attr-defined]

        # Unpickled attribtes include both the inherent attributes of RemoteModule
        # (not inherited from the superclass) and two installed methods.
        expected_unpickled_attrs = list(_REMOTE_MODULE_PICKLED_ATTRIBUTES)
        expected_unpickled_attrs.append("forward_async")
        expected_unpickled_attrs.append("forward")

        # Create a remote module on worker1 and then pass it to worker2 over the RPC layer.
        for remote_module in self._create_remote_module_iter(
            dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            # Test querying some simple attributes from worker2.
            attrs = rpc.rpc_sync(
                dst_worker2_name, remote_module_attributes, (remote_module,)
            )
            self.assertListEqual(list(attrs.keys()), expected_unpickled_attrs)  # type: ignore[attr-defined]
            self.assertEqual(attrs["on"], "worker1")  # type: ignore[attr-defined]
            self.assertEqual(attrs["device"], "cpu")  # type: ignore[attr-defined]
            self.assertFalse(attrs["is_device_map_set"])  # type: ignore[attr-defined]
            self.assertFalse(attrs["is_scriptable"])  # type: ignore[attr-defined]

            # Test the installed methods on worker1's can be initiated by worker2 over RPC layer.
            # NOTE: In practice a remote module should be directly stored on the worker that runs ``forward``` or ``foward_async``,
            # not have another worker to initiate forward over the RPC layer.
            args = (torch.ones(1), 2, "3")
            ret1 = rpc.rpc_sync(dst_worker2_name, remote_forward, (remote_module, args))
            self.assertEqual(ret1, tuple(reversed(args)))  # type: ignore[attr-defined]
            ret2 = rpc.rpc_sync(
                dst_worker2_name, remote_forward_async, (remote_module, args)
            )
            self.assertEqual(ret2, tuple(reversed(args)))  # type: ignore[attr-defined]

    @unittest.skip(
        "Script RemoteModule cannot be sent over RPC at this time. See #57865"
    )
    @dist_utils.dist_init
    def test_send_remote_module_over_the_wire_script(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker1_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]
        dst_worker2_name = dist_utils.worker_name((self.rank + 2) % self.world_size)  # type: ignore[attr-defined]

        # Unpickled attribtes include both the inherent attributes of RemoteModule
        # (not inherited from the superclass) and two installed methods.
        expected_unpickled_attrs = list(_REMOTE_MODULE_PICKLED_ATTRIBUTES)
        expected_unpickled_attrs.append("forward_async")
        expected_unpickled_attrs.append("forward")

        # Create a remote module on worker1 and then pass it to worker2 over the RPC layer.
        for remote_module in self._create_remote_module_iter(
            dst_worker1_name, modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE]
        ):
            # Test querying some simple attributes from worker2.
            attrs = rpc.rpc_sync(
                dst_worker2_name, remote_module_attributes, (remote_module,)
            )
            self.assertListEqual(list(attrs.keys()), expected_unpickled_attrs)  # type: ignore[attr-defined]
            self.assertEqual(attrs["on"], "worker1")  # type: ignore[attr-defined]
            self.assertEqual(attrs["device"], "cpu")  # type: ignore[attr-defined]
            self.assertFalse(attrs["is_device_map_set"])  # type: ignore[attr-defined]
            self.assertFalse(attrs["is_scriptable"])  # type: ignore[attr-defined]

            # Test the installed methods on worker1's can be initiated by worker2 over RPC layer.
            # NOTE: In practice a remote module should be directly stored on the worker that runs ``forward``` or ``foward_async``,
            # not have another worker to initiate forward over the RPC layer.
            args = (torch.ones(1), 2, "3")
            ret1 = rpc.rpc_sync(dst_worker2_name, remote_forward, (remote_module, args))
            self.assertEqual(ret1, tuple(reversed(args)))  # type: ignore[attr-defined]
            ret2 = rpc.rpc_sync(
                dst_worker2_name, remote_forward_async, (remote_module, args)
            )
            self.assertEqual(ret2, tuple(reversed(args)))  # type: ignore[attr-defined]


class CudaRemoteModuleTest(CommonRemoteModuleTest):
    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_valid_device(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        for remote_module in self._create_remote_module_iter(
            "{}/cuda:0".format(dst_worker_name), modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            device = rpc.rpc_sync(
                dst_worker_name, remote_device, (remote_module.module_rref,)
            )
            self.assertEqual(device.type, "cuda")  # type: ignore[attr-defined]
            self.assertEqual(device.index, 0)  # type: ignore[attr-defined]

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_invalid_devices(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        with self.assertRaisesRegex(  # type: ignore[attr-defined]
            RuntimeError,
            r"Expected one of .+ device type at start of device string",
        ):
            list(
                self._create_remote_module_iter(
                    "{}/foo".format(dst_worker_name),
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(  # type: ignore[attr-defined]
            RuntimeError, r"CUDA error: invalid device ordinal"
        ):
            list(
                self._create_remote_module_iter(
                    "{}/cuda:100".format(dst_worker_name),
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(RuntimeError, r"Invalid device string: 'cpu2'"):  # type: ignore[attr-defined]
            list(
                self._create_remote_module_iter(
                    "{}/cpu2".format(dst_worker_name),
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(RuntimeError, r"Device string must not be empty"):  # type: ignore[attr-defined]
            list(
                self._create_remote_module_iter(
                    "{}/".format(dst_worker_name),
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(  # type: ignore[attr-defined]
            RuntimeError,
            r"Could not parse remote_device: worker1/cuda:0/cuda:1. The valid format is '<workername>/<device>'",
        ):
            list(
                self._create_remote_module_iter(
                    "{}/cuda:0/cuda:1".format(dst_worker_name),
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(  # type: ignore[attr-defined]
            RuntimeError,
            r"The workername in remote_device '/' cannot be empty. The valid format is '<workername>/<device>'",
        ):
            list(
                self._create_remote_module_iter(
                    "/",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

        with self.assertRaisesRegex(  # type: ignore[attr-defined]
            RuntimeError,
            r"The workername in remote_device '/cuda:0' cannot be empty. The valid format is '<workername>/<device>'",
        ):
            list(
                self._create_remote_module_iter(
                    "/cuda:0",
                    modes=[ModuleCreationMode.MODULE_CTOR],
                )
            )

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_input_moved_to_cuda_device(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        # These two CPU tensors (in args and kwargs) should be implicitly moved to an appropriate cuda device.
        t1 = torch.ones(1)
        args = (t1, 2)
        t2 = t1 * 2
        kwargs = dict(word=t2)

        # Only test Python nn.Module, because script module methods don't support taking kwargs.
        for remote_module in self._create_remote_module_iter(
            "{}/cuda:0".format(dst_worker_name), modes=[ModuleCreationMode.MODULE_CTOR]
        ):
            ret_fut = remote_module.forward_async(*args, **kwargs)
            ret = ret_fut.wait()
            self.assertEqual(ret, tuple(reversed(args + (t2,))))  # type: ignore[attr-defined]
            # TODO: Once the RPC backend can support directly sending GPU tensors, the expected device type should be "cuda:0".
            self.assertEqual(ret[0].device.type, "cpu")  # type: ignore[attr-defined]
            self.assertEqual(ret[2].device.type, "cpu")  # type: ignore[attr-defined]

            ret = remote_module.forward(*args, **kwargs)
            self.assertEqual(ret, tuple(reversed(args + (t2,))))  # type: ignore[attr-defined]
            # TODO: Once the RPC backend can support directly sending GPU tensors, the expected device type should be "cuda:0".
            self.assertEqual(ret[0].device.type, "cpu")  # type: ignore[attr-defined]
            self.assertEqual(ret[2].device.type, "cpu")  # type: ignore[attr-defined]

    @skip_if_lt_x_gpu(1)
    @dist_utils.dist_init
    def test_input_moved_to_cuda_device_script(self):
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)  # type: ignore[attr-defined]

        scripted_remote_module = next(
            self._create_remote_module_iter(
                "{}/cuda:0".format(dst_worker_name),
                modes=[ModuleCreationMode.MODULE_CTOR_WITH_INTERFACE],
            )
        )

        @torch.jit.script
        def run_forward(scripted_remote_module: MyModuleInterface):
            ret = scripted_remote_module.forward(torch.ones(1), 2, "3")
            return ret

        ret = run_forward(scripted_remote_module)

        self.assertEqual(ret, ("3", 2, torch.ones(1)))  # type: ignore[attr-defined]
        # TODO: Once the RPC backend can support directly sending GPU tensors, the expected device type should be "cuda:0".
        self.assertEqual(ret[2].device.type, "cpu")  # type: ignore[attr-defined]
