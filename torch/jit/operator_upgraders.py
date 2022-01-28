"""
This is the centralized file for all PyTorch operator upgraders.
Each function definition here should satisfy following
requirements:
1. The body of the function must be Torchscriptable
2. The naming convention of the upgraders should be:
    <op_name>_<op_overload>_upgrader_<old_version>_<new_version>
3. The name of the upgrader must be present in
   torch/csrc/jit/operator_upgraders/version_map.h
"""
import torch
from typing import List, no_type_check, Optional, Union

# TODO (tugsuu) This context manager
# forbids the tests to override torch.jit.script
# Without it, torch.jit.load will fail due to
# circular dependency
with torch._jit_internal._disable_emit_hooks():
    @torch.jit.script
    def div_Tensor_0_3(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if (self.is_floating_point() or other.is_floating_point()):
            return self.true_divide(other)
        return self.divide(other, rounding_mode='trunc')

    @torch.jit.script
    def div_Scalar_0_3(self: torch.Tensor, other: Union[int, float]) -> torch.Tensor:
        if (self.is_floating_point() or isinstance(other, float)):
            return self.true_divide(other)
        return self.divide(other, rounding_mode='trunc')

    @torch.jit.script
    def div_out_0_3(self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
        if (self.is_floating_point() or other.is_floating_point() or out.is_floating_point()):
            return self.true_divide(other, out=out)
        return self.divide(other, rounding_mode='trunc', out=out)  # type: ignore[call-overload]

    @torch.jit.script
    def div__Tensor_0_3(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if (self.is_floating_point() or other.is_floating_point()):
            return self.true_divide_(other)
        return self.divide_(other, rounding_mode='trunc')

    @torch.jit.script
    def div__Scalar_0_3(self: torch.Tensor, other: Union[int, float]) -> torch.Tensor:
        if (self.is_floating_point() or isinstance(other, float)):
            return self.true_divide_(other)
        return self.divide_(other, rounding_mode='trunc')

    # TODO: since TS relies on typecheck comment of mypy
    # adding type: ignore at specific lines in this function
    # messes up our type refinement. For now, let's not check
    # type here.
    @no_type_check
    @torch.jit.script
    def full_0_4(size: List[int], fill_value: Union[int, float], *,
                 dtype: Optional[int], layout: Optional[int], device: Optional[torch.device],
                 pin_memory: Optional[bool]) -> torch.Tensor:
        if dtype is None:
            fill_value = float(fill_value)
        return torch.full(size, fill_value, dtype=dtype, layout=layout,
                          device=device, pin_memory=pin_memory)

    @torch.jit.script
    def full_out_0_4(size: List[int], fill_value: Union[int, float], *, out: torch.Tensor) -> torch.Tensor:
        return torch.full(size, fill_value, out=out)

def format_bytecode(table):
    # given a nested tuples, convert them to nested list
    def listify(content):
        if not isinstance(content, tuple):
            return content
        return [listify(i) for i in content]

    formatted_table = {}
    for entry in table:
        identifier = entry[0]
        content = entry[1]
        content = listify(content)
        formatted_table[identifier] = content
    return formatted_table

def collect_available_upgraders():
    # There needs to be 1 to 1 mapping between the
    # upgrader entries here and the list of upgraders
    # in the torch/csrc/operator_upgraders/version_map.h

    entries = globals()
    # ignore test operators
    version_map = {k : v for k, v in torch._C._get_operator_version_map().items()
                   if not k.startswith("aten::_test")}

    # 1. Check if everything in version_map.h is defined here
    available_upgraders_in_version_map = set()

    for op_name in version_map:
        for upgrader_entry in version_map[op_name]:
            if upgrader_entry.upgrader_name not in entries:
                raise AssertionError("Upgrader entry {} needs to be defined in python".format(upgrader_entry.upgrader_name))
            available_upgraders_in_version_map.add(upgrader_entry.upgrader_name)

    # 2. Check if everything in this file is registered in version_map.h
    for entry in entries:
        if isinstance(entries[entry], torch.jit.ScriptFunction):
            if entry not in available_upgraders_in_version_map:
                raise AssertionError("The upgrader {} is not registered in the version_map.h".format(entry))

    return available_upgraders_in_version_map

def generate_bytecode() -> List:
    upgrader_set = collect_available_upgraders()
    yaml_content = []
    for upgrader_name in upgrader_set:
        upgrader_graph = globals()[upgrader_name].graph
        upgrader_bytecode = torch._C._compile_graph_to_code_table(upgrader_name, upgrader_graph)
        entry = {upgrader_name: format_bytecode(upgrader_bytecode)}
        yaml_content.append(entry)
    return yaml_content

if __name__ == "__main__":
    raise RuntimeError("This file is not meant to be run directly")
