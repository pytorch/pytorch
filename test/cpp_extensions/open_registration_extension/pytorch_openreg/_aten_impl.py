import logging

import torch
from torch.utils._pytree import tree_any


log = logging.getLogger(__name__)

from ._device_daemon import daemon
from ._meta_parser import prepare_for_sending, to_device_no_copy


_IMPL_REGISTRY = {}


# Define all the implementations in the registry
def _register_same_name(name, with_log=False):
    def _(*args, **kwargs):
        if with_log:
            log.info("Calling hook %s", name)
        return daemon.exec(name, *args, **kwargs)

    _IMPL_REGISTRY[name] = _


_register_same_name("deviceCount")
_register_same_name("getDevice")
_register_same_name("uncheckedSetDevice")
_register_same_name("exchangeDevice")
_register_same_name("malloc", True)
_register_same_name("free", True)

_openreg_lib = torch.library.Library("_", "IMPL")


def _openreg_kernel_fallback(op, *args, **kwargs):
    log.info("Calling kernel %s", op)

    # Special ops needed to avoid infinite recursion
    if op is torch.ops.aten._copy_from.default:
        from_, to_ = args
        if from_.device.type == to_.device.type:
            assert from_.device.type == "openreg"
            op = torch.ops.aten.copy_.default
            # handled below as a regular copy
        elif from_.device.type == "openreg":
            args, _ = prepare_for_sending((from_,), {})
            host_mem = daemon.exec("send_data", *args)
            return to_.copy_(host_mem)
        elif to_.device.type == "openreg":
            args, _ = prepare_for_sending((to_,), {})
            daemon.exec("recv_data", from_, *args)
            return to_
        else:
            raise RuntimeError("Should not happen")
    elif op is torch.ops.aten.set_.source_Tensor:
        return torch.ops.aten.set_.source_Storage_storage_offset(
            args[0],
            args[1].untyped_storage(),
            args[1].storage_offset(),
            args[1].size(),
            args[1].stride(),
        )
    elif op is torch.ops.aten._local_scalar_dense.default:
        args, _ = prepare_for_sending(args, {})
        host_mem = daemon.exec("send_data", *args)
        return host_mem.item()

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
            shape = daemon.exec(
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
    daemon.exec("run_op", op_name, args, kwargs)

    if post_process is not None:
        post_process()

    return real_res


_openreg_lib.fallback(_openreg_kernel_fallback, dispatch_key="PrivateUse1")
