from typing import Union

import torch
import torch.utils._pytree as pytree
from torch.export.exported_program import ExportedProgram


__all__ = ["move_to_device_pass"]


def move_to_device_pass(
    ep: ExportedProgram, location: Union[torch.device, str, dict[str, str]]
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
        location: Union[torch.device, str, dict[str, str]],
    ) -> str:
        if isinstance(location, dict):
            if str(curr_device) in location:
                return location[str(curr_device)]
            else:
                return str(curr_device)
        else:
            return str(location)

    # move all the state_dict
    for k, v in ep.state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            ep._state_dict[k] = torch.nn.Parameter(
                v.to(_get_new_device(v.device, location)),
                v.requires_grad,
            )
        else:
            ep._state_dict[k] = v.to(_get_new_device(v.device, location))

    # move all the constants
    for k, v in ep.constants.items():
        if isinstance(v, torch.Tensor):
            ep._constants[k] = v.to(_get_new_device(v.device, location))

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

    ep.validate()
    return ep
