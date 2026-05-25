from typing import Union

import torch
import torch.utils._pytree as pytree
from torch.export.exported_program import ExportedProgram


__all__ = ["move_to_device_pass"]


def move_to_device_pass(
    ep: ExportedProgram, location: torch.device | str | dict[str, str]
) -> ExportedProgram:
    """
    Move the exported program to the given device.

    Args:
        ep (ExportedProgram): The exported program to move.
        location (Union[torch.device, str, Dict[str, str]]): The device to move the exported program to.
            If a string, it is interpreted as a device name.
            If a dict, it is interpreted as a mapping from
            the existing device to the intended one

    Returns:
        ExportedProgram: The moved exported program.
    """

    def _get_new_device(
        curr_device: torch.device,
        location: torch.device | str | dict[str, str],
    ) -> str:
        if isinstance(location, dict):
            if str(curr_device) in location:
                return location[str(curr_device)]
            else:
                return str(curr_device)
        else:
            return str(location)

    def _move_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(_get_new_device(tensor.device, location))

    def _move_parameter(param: torch.nn.Parameter) -> torch.nn.Parameter:
        with torch.no_grad():
            param_applied = _move_tensor(param)
        if torch._has_compatible_shallow_copy_type(param, param_applied):
            param.data = param_applied
            out_param = param
        else:
            out_param = torch.nn.Parameter(param_applied, param.requires_grad)

        if param.grad is not None:
            with torch.no_grad():
                grad_applied = _move_tensor(param.grad)
            if torch._has_compatible_shallow_copy_type(param.grad, grad_applied):
                param.grad.data = grad_applied
                out_param.grad = param.grad
            else:
                out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)
        return out_param

    # move all the state_dict
    moved_parameters: dict[int, torch.nn.Parameter] = {}
    for k, v in ep.state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            param_id = id(v)
            if param_id not in moved_parameters:
                moved_parameters[param_id] = _move_parameter(v)
            ep._state_dict[k] = moved_parameters[param_id]
        else:
            ep._state_dict[k] = _move_tensor(v)

    # move all the constants
    for k, v in ep.constants.items():
        if isinstance(v, torch.Tensor):
            ep._constants[k] = _move_tensor(v)

    # move example_inputs if they exist
    if ep.example_inputs is not None:
        args, kwargs = ep.example_inputs
        moved_args = pytree.tree_map_only(
            torch.Tensor,
            lambda tensor: tensor.to(_get_new_device(tensor.device, location)),
            args,
        )
        moved_kwargs = pytree.tree_map_only(
            torch.Tensor,
            lambda tensor: tensor.to(_get_new_device(tensor.device, location)),
            kwargs,
        )
        ep._example_inputs = (moved_args, moved_kwargs)

    for m in ep.graph_module.modules():
        if isinstance(m, torch.fx.GraphModule):
            for node in m.graph.nodes:
                # move all the nodes kwargs with burnt-in device
                if "device" in node.kwargs:
                    kwargs = node.kwargs.copy()
                    kwargs["device"] = _get_new_device(kwargs["device"], location)
                    node.kwargs = kwargs

                if (
                    node.op == "call_function"
                    and node.target is torch.ops.aten.to.device
                ):
                    args = list(node.args)
                    # pyrefly: ignore [unsupported-operation]
                    args[1] = _get_new_device(args[1], location)
                    node.args = tuple(args)

                # move all the tensor metadata
                node.meta["val"] = pytree.tree_map(
                    lambda v: v.to(_get_new_device(v.device, location))
                    if isinstance(v, torch.Tensor)
                    else v,
                    node.meta.get("val"),
                )
            m.recompile()

    ep.validate()
    return ep
