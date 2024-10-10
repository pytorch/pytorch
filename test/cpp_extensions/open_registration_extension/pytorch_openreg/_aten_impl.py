import logging

import torch
from torch.utils._pytree import tree_any


log = logging.getLogger(__name__)

from ._device_daemon import driver
from ._meta_parser import prepare_for_sending, to_device_no_copy


_IMPL_REGISTRY = {}


# Define all the implementations in the registry
def _register_same_name(name, with_log=False):
    def _(*args, **kwargs):
        if with_log:
            log.info("Calling hook %s", name)
        return driver.exec(name, *args, **kwargs)

    _IMPL_REGISTRY[name] = _


_register_same_name("deviceCount")
_register_same_name("getDevice")
_register_same_name("uncheckedSetDevice")
_register_same_name("exchangeDevice")
_register_same_name("malloc", True)
_register_same_name("free", True)
_register_same_name("isPinnedPtr", True)
_register_same_name("hostMalloc", True)
_register_same_name("hostFree", True)
_register_same_name("getNewStream")
_register_same_name("queryStream")
_register_same_name("getStream")
_register_same_name("exchangeStream")
_register_same_name("synchronizeStream")


# TODO: replace it with implementing torch.openreg.device
class DeviceContext:
    def __init__(self, device):
        self.idx = device.index

    def __enter__(self):
        self.prev = driver.exec("exchangeDevice", self.idx)

    def __exit__(self, *args):
        driver.exec("uncheckedSetDevice", self.prev)


def _openreg_kernel_fallback(op, *args, **kwargs):
    def get_tensor_device(*args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "openreg":
                return arg.device

    device = get_tensor_device(*args)
    if device is None:
        return _kernel_fallback(op, *args, **kwargs)

    # Mimicks the DeviceGuard system we have in aten
    with DeviceContext(device):
        return _kernel_fallback(op, *args, **kwargs)


def _kernel_fallback(op, *args, **kwargs):
    log.info("Calling kernel %s", op)

    op_name = None
    post_process = None
    if "out" in op._overloadname:
        # Note that all structured native op will call here
        if isinstance(kwargs["out"], tuple):
            raise RuntimeError(f"out= variant {op} with tuple out= not supported")
        if kwargs["out"].nelement() == 0:
            # Out variant that needs a resize, convert to an out of place
            # and handle generically below
            orig_out = kwargs["out"]
            del kwargs["out"]
            if op._overloadname != "out":
                raise RuntimeError(
                    "Cannot retranslate non-default out= variant form 0 size"
                )
            op = op.overloadpacket.default

            def _post_process():
                nonlocal real_res
                orig_out.set_(real_res)
                real_res = orig_out

            post_process = _post_process

        else:
            # No metadata update to do, just run the op on the device
            op_name = op.overloadpacket._qualified_op_name
            real_res = kwargs["out"]
    elif not tree_any(lambda obj: isinstance(obj, torch.Tensor), (args, kwargs)):
        # No Tensor argument means factory function
        # They should decompose and be handled in our c++ side directly
        raise RuntimeError(f"{op} not handled yet.")
    elif op._schema.is_mutable or op is torch.ops.aten._copy_from.default:
        # Only handle inplace ops returning their first arg
        assert len(args) >= 1, f"Inplace {op} needs at least one arg"
        assert (
            len(op._schema.returns) == 1
        ), f"NYI Inplace {op} with more than one return"
        op_name = op.overloadpacket._qualified_op_name
        real_res = args[0]
    elif any(r.alias_info is not None for r in op._schema.returns):
        # View ops
        if op is torch.ops.aten.view.default:
            return torch.ops.aten._unsafe_view(*args, **kwargs)
        raise RuntimeError(f"{op} view op is not handled yet")

    if op_name is None:
        # 1. Compute updated metadata
        if torch.Tag.dynamic_output_shape not in op.tags:
            # Usual case: run the meta op to see the output metadata
            meta_args, meta_kwargs = to_device_no_copy("meta", args, kwargs)
            meta_res = op(*meta_args, **meta_kwargs)

            # 2. Allocate the output
            real_res, _ = to_device_no_copy("openreg", meta_res, {})
        else:
            # Slow version for data-dependent functions:
            # Run the op on the device just to get the output shape
            args_, kwargs_ = prepare_for_sending(args, kwargs)
            shape = driver.exec(
                "get_op_output_shape",
                op.overloadpacket._qualified_op_name,
                args_,
                kwargs_,
            )

            # 2. Allocate the output
            real_res = args[0].new(shape)

        # 3. Move to out variant
        kwargs["out"] = real_res
        # Let overload resolution find the out= overload
        op_name = op.overloadpacket._qualified_op_name

    # 4. Run the compute and populate the output on the device
    args, kwargs = prepare_for_sending(args, kwargs)
    driver.exec("run_op", op_name, args, kwargs)

    if post_process is not None:
        post_process()

    return real_res


def copy_from_device(from_):
    with DeviceContext(from_.device):
        args, _ = prepare_for_sending((from_,), {})
        return driver.exec("send_data", *args)


def copy_from_host_to_device(from_, to_):
    with DeviceContext(to_.device):
        args, _ = prepare_for_sending((to_,), {})
        driver.exec("recv_data", from_, *args)
    return to_


def _copy_from(from_, to_):
    if from_.device.type == to_.device.type:
        assert from_.device.type == "openreg"
        if from_.device.index == to_.device.index:
            op = torch.ops.aten.copy_.default
            return _openreg_kernel_fallback(op, to_, from_)
        else:
            host_mem = copy_from_device(from_)
            return copy_from_host_to_device(host_mem, to_)
    elif from_.device.type == "openreg":
        host_mem = copy_from_device(from_)
        return to_.copy_(host_mem)
    elif to_.device.type == "openreg":
        return copy_from_host_to_device(from_, to_)
    else:
        raise RuntimeError("Should not happen")


def _set_source_tensor(ten1, ten2):
    return torch.ops.aten.set_.source_Storage_storage_offset(
        ten1,
        ten2.untyped_storage(),
        ten2.storage_offset(),
        ten2.size(),
        ten2.stride(),
    )


def _local_scalar_dense(ten):
    host_mem = copy_from_device(ten)
    return host_mem.item()


_openreg_lib = torch.library.Library("_", "IMPL")
_openreg_lib.fallback(_openreg_kernel_fallback, dispatch_key="PrivateUse1")

_openreg_lib_aten = torch.library.Library("aten", "IMPL")
_openreg_lib_aten.impl("_copy_from", _copy_from, dispatch_key="PrivateUse1")
_openreg_lib_aten.impl(
    "set_.source_Tensor", _set_source_tensor, dispatch_key="PrivateUse1"
)
_openreg_lib_aten.impl(
    "_local_scalar_dense", _local_scalar_dense, dispatch_key="PrivateUse1"
)
