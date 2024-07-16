# mypy: allow-untyped-defs
import re
from collections import defaultdict
from typing import Any, Dict

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed._tensor.api import DTensor
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.module_tracker import ModuleTracker


funcol_native = torch.ops._c10d_functional
funcol_py = torch.ops.c10d_functional
funcol_autograd = torch.ops._c10d_functional_autograd
c10d_ops = torch.ops.c10d

NATIVE_TO_PY_MAPPING = {
    funcol_native.all_gather_into_tensor: funcol_py.all_gather_into_tensor,
    funcol_native.all_gather_into_tensor_coalesced: funcol_py.all_gather_into_tensor_coalesced,
    funcol_native.all_reduce: funcol_py.all_reduce,
    funcol_native.all_reduce_coalesced: funcol_py.all_reduce_coalesced,
    funcol_native.all_to_all_single: funcol_py.all_to_all_single,
    funcol_native.broadcast: funcol_py.broadcast,
    funcol_native.reduce_scatter_tensor: funcol_py.reduce_scatter_tensor,
    funcol_native.reduce_scatter_tensor_coalesced: funcol_py.reduce_scatter_tensor_coalesced,
    # functional ops
    funcol_autograd.all_to_all_single: funcol_py.all_to_all_single,
}

c10d_collective_ops = {
    c10d_ops._allgather_base_,
    c10d_ops._reduce_scatter_base_,
    c10d_ops.allgather_,
    c10d_ops.allgather_coalesced_,
    c10d_ops.allgather_into_tensor_coalesced_,
    c10d_ops.allreduce_,
    c10d_ops.allreduce_coalesced_,
    c10d_ops.alltoall_,
    c10d_ops.alltoall_base_,
    c10d_ops.broadcast_,
    c10d_ops.gather_,
    c10d_ops.scatter_,
    c10d_ops.reduce_,
    c10d_ops.reduce_scatter_,
    c10d_ops.reduce_scatter_tensor_coalesced_,
}


class CommModeModuleTracker(ModuleTracker):
    """
    Inherits ModuleTracker and expands on its functionality to track the
    parameters and sharding information of a model at a module-level
    """

    def __init__(self):
        super().__init__()
        self.module_depth_dict = {}
        self.module_parameters_dict = {}
        self.sharding_dict = {}
        self.name = ""

    def _fw_pre_hook(self, mod, input):
        """
        This function is called before the forward pass of a module. It
        collects the parameters and sharding information of a module and
        stores it in a dictionary.
        """
        self.name = super()._get_mod_name(mod)

        # contains information about module ordering and depth in the module tree
        self.module_depth_dict[self.name] = len(self.parents)
        # adds current sub-module to module tracker parent class
        super()._get_append_fn(self.name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            register_multi_grad_hook(tensors, super()._get_pop_fn(self.name, True))

        for param_name, param in mod.named_parameters(recurse=False):
            if self.name not in self.module_parameters_dict:
                self.module_parameters_dict[self.name] = {}

            self.module_parameters_dict[self.name][param_name] = param.data

            if isinstance(param.data, DTensor):
                key_name = self.name + "." + param_name
                self.sharding_dict[key_name] = param.data.placements

    def __enter__(self):
        self.module_parameters_dict.clear()
        self.sharding_dict.clear()
        self.module_depth_dict.clear()
        self.module_depth_dict["Global"] = 0
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(super()._fw_post_hook)

    def __exit__(self, *args):
        super().__exit__(*args)

    def print_paramater_info(self):
        print(self.module_parameters_dict)

    def print_sharding_info(self):
        for key, value in self.sharding_dict.items():
            print(key + ": " + str(value))


class CommDebugMode(TorchDispatchMode):
    """
    ``CommDebugMode`` is a context manager that counts the number of
    functional collectives within its context. It does this using a
    ``TorchDispatchMode``.

    NOTE: this mode only works for functional collective atm and the
    distributed_c10d collectives are not supported yet.

    Example usage

    .. code-block:: python

        mod = ...
        comm_mode = CommDebugMode()
        with comm_mode:
            mod.sum().backward()

    """

    def __init__(self):
        self.comm_counts: Dict[Any, int] = defaultdict(int)
        self.comm_module_counts = {}
        self.comm_registry = set()
        for native_op, py_op in NATIVE_TO_PY_MAPPING.items():
            self.comm_registry.add(native_op)
            self.comm_registry.add(py_op)

        self.comm_registry.add(torch.ops._dtensor.shard_dim_alltoall)
        self.advanced_module_tracker = CommModeModuleTracker()

    def generate_module_tracing_table(self):
        """
        Inspired by flop counter, generates a detailed table displaying collective tracing
        information on a module level
        """
        table = ""
        for fqn in self.advanced_module_tracker.module_depth_dict:
            indent = "  " * (self.advanced_module_tracker.module_depth_dict[fqn])
            table += f"{indent}{fqn}\n"

            # prints out all collectives in the respective sub-module
            if fqn in self.comm_module_counts:
                for collective, count in self.comm_module_counts[fqn].items():
                    collective_indent = "  " * (
                        (self.advanced_module_tracker.module_depth_dict[fqn]) + 1
                    )
                    table += (
                        f"\033[1;33m{collective_indent}*{collective}: {count}\033[0m\n"
                    )

        return table

    def get_total_counts(self) -> int:
        return sum(self.comm_counts.values())

    def get_comm_counts(self) -> Dict[Any, int]:
        """Returns the communication counts as a dictionary.

        Returns:
            Dict[Any, int]: The communication counts as a dictionary.
        """
        return self.comm_counts

    def get_comm_module_counts(self) -> Dict[str, Dict[Any, int]]:
        """
        Returns the communication counts at a module level as a dictionary.
        """
        return self.comm_module_counts

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        return self.advanced_module_tracker.module_parameters_dict

    def get_sharding_info(self) -> Dict[str, Dict[str, Any]]:
        return self.advanced_module_tracker.sharding_dict

    def __enter__(self):
        self.comm_counts.clear()
        self.comm_module_counts.clear()
        super().__enter__()
        self.advanced_module_tracker.__enter__()
        return self

    def __exit__(self, *args):
        self.advanced_module_tracker.__exit__()
        super().__exit__(*args)

    def log_module_tracing_table_to_file(self):
        # ansi_escape is used to remove ANSI escape sequences in table used to make terminal output more readable

        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        table = ansi_escape.sub("", self.generate_module_tracing_table())

        with open("output.txt", "w") as log_file:
            log_file.write(table)

    def print_paramater_info(self):
        self.advanced_module_tracker.print_paramater_info()

    def print_sharding_info(self):
        self.advanced_module_tracker.print_sharding_info()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into comms ops, before CommDebugMode sees them.
        if any(t == DTensor for t in types):
            return NotImplemented
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        # We have many tests that use CommDebugMode to verify the occurrence of
        # collectives. These tests do so by querying comm_counts with legacy
        # funcol ops as key. For the purpose of native funcol migration, we
        # need these tests to work for both legacy and native funcol. To avoid
        # the need to modify all tests to accommodate the two implementations,
        # we make CommDebugMode translate native funcol ops into legacy funcol
        # ops until the migration finishes.

        if func_packet in self.comm_registry or func_packet in c10d_collective_ops:
            if func_packet in NATIVE_TO_PY_MAPPING:
                func_packet = NATIVE_TO_PY_MAPPING[func_packet]
            self.comm_counts[func_packet] += 1

            # adds collective count to current module
            if self.advanced_module_tracker.name not in self.comm_module_counts:
                self.comm_module_counts[
                    self.advanced_module_tracker.name
                ] = defaultdict(int)
            self.comm_module_counts[self.advanced_module_tracker.name][func_packet] += 1

            # adds collective count to parent modules
            for par in self.advanced_module_tracker.parents:
                # makes sure we aren't double counting when current sub-module hasn't been removed from parents
                if par != self.advanced_module_tracker.name:
                    if par not in self.comm_module_counts:
                        self.comm_module_counts[par] = defaultdict(int)
                    self.comm_module_counts[par][func_packet] += 1

        return out
