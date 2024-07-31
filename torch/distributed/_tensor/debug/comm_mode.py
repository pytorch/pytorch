# mypy: allow-untyped-defs
import copy
import json
import re
import weakref

from collections import defaultdict
from typing import Any, Dict

import torch

import torch.nn

from torch.autograd.graph import register_multi_grad_hook
from torch.distributed._tensor.api import DTensor
from torch.distributed._tools.mod_tracker import ModTracker

from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
    register_module_full_backward_pre_hook,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

funcol_native = torch.ops._c10d_functional
funcol_py = torch.ops.c10d_functional
from torch._guards import detect_fake_mode

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

trivial_ops = {
    "aten.detach.default",
    "aten.t.default",
    "aten.view.default",
    "aten._to_copy.default",
    "aten.as_strided.default",
    "aten.transpose.int",
}


class CommModeModuleTracker(ModTracker):
    """
    Inherits ModuleTracker and expands on its functionality to track the
    parameters and sharding information of a model at a module-level
    """

    def __init__(self):
        super().__init__()
        self.module_helper_dict = {}
        self.module_parameters_dict = {}
        self.module_parents_dict = {}
        self.register_forward_hook_handles = {}
        self.parent_dict = {}
        self.parent_list = []
        self.sharding_dict = {}
        self.activation_checkpointing = False
        self.name = ""

    def _fw_set_module_hook(self, mod, input, output):
        """
        Updates the current module after module finishes running and
        all other hooks are resolved
        """

        if self.is_bw:
            self.activation_checkpointing = True
        else:
            self.activation_checkpointing = False

        if not self.activation_checkpointing:
            # module is no longer parent of next modules
            self.parent_list.pop()

            # set current module to previous parent module
            self.name = self.parent_list[-1]

    def _fw_pre_hook(self, mod, input):
        """
        This function is called before the forward pass of a module. It
        collects the parameters and sharding information of a module and
        stores it in a dictionary.
        """
        if self.is_bw:
            self.activation_checkpointing = True
        else:
            self.activation_checkpointing = False

        self.name = super()._get_mod_name(mod)
        w_mod = weakref.ref(mod)

        # adds current sub-module to module tracker parent class
        super()._get_append_fn(w_mod, self.name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw and tensors:
            register_multi_grad_hook(
                tensors, super()._get_pop_fn(w_mod, self.name, True)
            )

        if not self.activation_checkpointing:
            # contains information about module ordering and depth in the module tree
            if self.name not in self.module_helper_dict:
                self.module_helper_dict[self.name] = {}

            self.module_helper_dict[self.name]["module_type"] = (
                str(type(mod)).replace("<", "").replace(">", "")
            )
            self.module_helper_dict[self.name]["depth"] = len(self.parents) - 1

            for param_name, param in mod.named_parameters(recurse=False):
                if self.name not in self.module_parameters_dict:
                    self.module_parameters_dict[self.name] = {}

                self.module_parameters_dict[self.name][param_name] = param.data

                if isinstance(param.data, DTensor):
                    key_name = self.name + "." + param_name
                    self.sharding_dict[key_name] = param.data.placements

                    if "parameters" not in self.module_helper_dict[self.name]:
                        self.module_helper_dict[self.name]["parameters"] = {}

                    self.module_helper_dict[self.name]["parameters"][param_name] = str(
                        param.data.placements
                    )

            # used to store module's parents to ensure correctness in backward pass/checkpointing
            if self.name not in self.module_parents_dict:
                self.module_parents_dict[self.name] = copy.deepcopy(self.parents)

            # used to create parent-child module associations for json dumps
            parent = self.parent_list[-1]
            if parent not in self.parent_dict:
                self.parent_dict[parent] = []

            self.parent_dict[parent].append(self.name)
            self.parent_list.append(self.name)

            self.register_forward_hook_handles[self.name] = mod.register_forward_hook(
                self._fw_set_module_hook
            )

    def _fw_post_hook(self, mod, input, output):
        """
        This function is called when the forward pass of a module is called.
        It updates the module tracker and removes the module from parent data
        """

        super()._fw_post_hook(mod, input, output)

    def _bw_hook(self, mod, output):
        """
        This function is called when the backward pass of a module is called. It
        updates the current module for backward passes
        """
        self.activation_checkpointing = False
        self.name = super()._get_mod_name(mod)

    def __enter__(self):
        self.activation_checkpointing = False
        self.module_parameters_dict.clear()
        self.sharding_dict.clear()
        self.parent_dict.clear()
        self.parent_list = ["Global"]
        self.module_helper_dict.clear()
        self.module_helper_dict["Global"] = {"depth": 0}
        self.module_parents_dict.clear()
        self.module_parents_dict["Global"] = set()
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(self._fw_post_hook)
        self.register_forward_hook_handles.clear()
        self._bw_handle = register_module_full_backward_pre_hook(self._bw_hook)
        self.name = "Global"

    def __exit__(self, *args):
        super().__exit__(*args)
        self._bw_handle.remove()

        # removes all forward_hook handles added in the pre-hook
        for handle in self.register_forward_hook_handles.values():
            handle.remove()

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
        self.comm_module_operation_counts = {}
        self.comm_registry = set()
        for native_op, py_op in NATIVE_TO_PY_MAPPING.items():
            self.comm_registry.add(native_op)
            self.comm_registry.add(py_op)

        self.comm_registry.add(torch.ops._dtensor.shard_dim_alltoall)
        self.advanced_module_tracker = CommModeModuleTracker()

    def generate_json_dump(self, file_name="comm_mode_log.json", noise_level=3):
        """
        Creates json file used to build browser visual
        0. prints module-level collective counts
        1. prints dTensor operations not included in trivial operations
        2. prints operations not included in trivial operations
        3. prints all operations
        """

        (
            include_DTensor_ops,
            include_module_data,
            include_ops,
            include_trivial_ops,
        ) = self.set_noise_parameters(noise_level)

        # recursively builds json data
        def add_json_information(json_dict, fqn):
            json_dict["fqn"] = fqn
            json_dict["module_type"] = ""
            json_dict["parameters"] = []
            json_dict["children"] = []
            json_dict["collectives_forward"] = []
            json_dict["collectives_backward"] = []
            json_dict["operations_forward"] = []
            json_dict["operations_backward"] = []

            # adds module layer type and parameters, and their sharding
            if (
                "module_type" in self.advanced_module_tracker.module_helper_dict[fqn]
                and include_module_data
            ):
                json_dict[
                    "module_type"
                ] = self.advanced_module_tracker.module_helper_dict[fqn]["module_type"]

                if "parameters" in self.advanced_module_tracker.module_helper_dict[fqn]:
                    for (
                        param_name,
                        placement,
                    ) in self.advanced_module_tracker.module_helper_dict[fqn][
                        "parameters"
                    ].items():
                        json_dict["parameters"].append((param_name, placement))

            # adds module collective information
            if fqn in self.comm_module_counts:
                for collective, count in self.comm_module_counts[fqn][
                    "forward"
                ].items():
                    json_dict["collectives_forward"].append((str(collective), count))

                for collective, count in self.comm_module_counts[fqn][
                    "backward"
                ].items():
                    json_dict["collectives_backward"].append((str(collective), count))

            # adds module operation information
            forward_operations = []
            backward_operations = []
            checkpointing_operations = []

            # only get operations if the minimum operation noise level is set to true
            if include_DTensor_ops:
                if fqn in self.comm_module_operation_counts:
                    (
                        forward_operations,
                        backward_operations,
                        checkpointing_operations,
                    ) = self.get_operations_list(self.comm_module_operation_counts[fqn])

            # remove all operations who don't have DTensor inputs
            if not include_ops:
                forward_operations = [
                    op for op in forward_operations if len(op["input_sharding"])
                ]
                backward_operations = [
                    op for op in backward_operations if len(op["input_sharding"])
                ]
                checkpointing_operations = [
                    op for op in checkpointing_operations if len(op["input_sharding"])
                ]

            # remove all operations in trivial operations set
            if not include_trivial_ops:
                forward_operations = [
                    op
                    for op in forward_operations
                    if str(op["name"]) not in trivial_ops
                ]
                backward_operations = [
                    op
                    for op in backward_operations
                    if str(op["name"]) not in trivial_ops
                ]
                checkpointing_operations = [
                    op
                    for op in checkpointing_operations
                    if str(op["name"]) not in trivial_ops
                ]

            # converts operation information into string format for json.dumps()
            forward_operations = copy.deepcopy(forward_operations)
            for op in forward_operations:
                op["name"] = str(op["name"])

                for i in range(len(op["input_sharding"])):
                    op["input_sharding"][i] = str(op["input_sharding"][i])
                    op["input_shape"][i] = str(op["input_shape"][i])

            backward_operations = copy.deepcopy(backward_operations)
            for op in backward_operations:
                op["name"] = str(op["name"])

                for i in range(len(op["input_sharding"])):
                    op["input_sharding"][i] = str(op["input_sharding"][i])
                    op["input_shape"][i] = str(op["input_shape"][i])

            checkpointing_operations = copy.deepcopy(checkpointing_operations)
            for op in checkpointing_operations:
                op["name"] = str(op["name"])

                for i in range(len(op["input_sharding"])):
                    op["input_sharding"][i] = str(op["input_sharding"][i])
                    op["input_shape"][i] = str(op["input_shape"][i])

            json_dict["operations_forward"] = forward_operations
            json_dict["operations_backward"] = backward_operations
            json_dict["operations_checkpointing"] = checkpointing_operations

            if fqn not in self.advanced_module_tracker.parent_dict:
                return json_dict

            # recursively adds module's children
            for ele in self.advanced_module_tracker.parent_dict[fqn]:
                json_dict["children"].append(add_json_information({}, ele))

            return json_dict

        json_dict: Dict[str, Any] = {}
        add_json_information(json_dict, "Global")

        # converts dictonary into json file
        with open(file_name, "w") as json_file:
            json.dump(json_dict, json_file, indent=4)

    def generate_comm_debug_tracing_table(self, noise_level=3):
        """
        Generates detailed table displaying operations and collective tracing information
        on a module level. Amount of information is dependent on noise_level

        0. prints module-level collective counts
        1. prints dTensor operations not included in trivial operations, module information
        2. prints operations not included in trivial operations
        3. prints all operations
        """

        (
            include_DTensor_ops,
            include_module_data,
            include_ops,
            include_trivial_ops,
        ) = self.set_noise_parameters(noise_level)

        table = ""
        for fqn in self.advanced_module_tracker.module_helper_dict:
            # setting up indentations for table formatting
            indent = "  " * (
                2 * self.advanced_module_tracker.module_helper_dict[fqn]["depth"]
            )
            table += f"{indent}{fqn}\n"

            if include_module_data:
                if (
                    "module_type"
                    in self.advanced_module_tracker.module_helper_dict[fqn]
                ):
                    module_type = self.advanced_module_tracker.module_helper_dict[fqn][
                        "module_type"
                    ]
                    table += f"{indent}*module type: {module_type}\n"

                if "parameters" in self.advanced_module_tracker.module_helper_dict[fqn]:
                    table += f"{indent}*Parameter List\n"
                    for (
                        param_name,
                        placement,
                    ) in self.advanced_module_tracker.module_helper_dict[fqn][
                        "parameters"
                    ].items():
                        table += f"{indent} *{param_name}: {placement}\n"

            indent += "  "
            collective_indent = "  " * (
                2 * self.advanced_module_tracker.module_helper_dict[fqn]["depth"] + 2
            )
            operation_indent = "  " * (
                2 * self.advanced_module_tracker.module_helper_dict[fqn]["depth"] + 3
            )

            # separate the module's collective and operations by forward and backward
            forward_collectives = {}
            backward_collectives = {}
            if fqn in self.comm_module_counts:
                forward_collectives = self.comm_module_counts[fqn]["forward"]
                backward_collectives = self.comm_module_counts[fqn]["backward"]

            forward_operations = []
            backward_operations = []
            checkpointing_operations = []

            if include_DTensor_ops:
                if fqn in self.comm_module_operation_counts:
                    (
                        forward_operations,
                        backward_operations,
                        checkpointing_operations,
                    ) = self.get_operations_list(self.comm_module_operation_counts[fqn])

            def add_tracing_information(table, collectives_dict, operation_list):
                """
                adds tracing information for module's forward or backward
                """
                for collective, count in collectives_dict.items():
                    table += (
                        f"\033[1;33m{collective_indent}*{collective}: {count}\033[0m\n"
                    )

                def add_operations(
                    table, operation, collective_indent, operation_indent
                ):
                    """
                    adds operation information to the table
                    """
                    table += f"\033[1;33m{collective_indent}**{operation_name}\033[0m\n"

                    if len(operation["input_shape"]):
                        operation_shape = operation["input_shape"]
                        operation_sharding = operation["input_sharding"]
                        operation_device_mesh = operation["device_mesh"]

                        table += f"\033[1;31m{operation_indent}shape: {operation_shape}\033[0m\n"
                        table += f"\033[1;31m{operation_indent}sharding: {operation_sharding}\033[0m\n"
                        table += f"\033[1;31m{operation_indent}device mesh: {operation_device_mesh}\033[0m\n"

                    return table

                for operation in operation_list:
                    operation_name = str(operation["name"])

                    # include all operations
                    if include_trivial_ops:
                        table = add_operations(
                            table, operation, collective_indent, operation_indent
                        )

                    # include all operations not in trivial operations
                    elif include_ops and operation_name not in trivial_ops:
                        table = add_operations(
                            table, operation, collective_indent, operation_indent
                        )

                    # only include dTensor operations not in trivial set
                    elif (
                        include_DTensor_ops
                        and (operation_name not in trivial_ops)
                        and len(operation["input_shape"])
                    ):
                        table = add_operations(
                            table, operation, collective_indent, operation_indent
                        )

                return table

            if len(forward_collectives) or len(forward_operations):
                table += f"{indent}FORWARD PASS\n"
                table = add_tracing_information(
                    table, forward_collectives, forward_operations
                )

            if len(backward_collectives) or len(backward_operations):
                table += f"{indent}BACKWARD PASS\n"
                table = add_tracing_information(
                    table, backward_collectives, backward_operations
                )

            if len(checkpointing_operations):
                table += f"{indent}ACTIVATION CHECKPOINTING\n"
                table = add_tracing_information(table, {}, checkpointing_operations)

        return table

    def get_operations_list(self, module_operation_counts):
        forward_operations = [
            op for op in module_operation_counts["operations_list"] if not op["is_bw"]
        ]
        backward_operations = [
            op
            for op in module_operation_counts["operations_list"]
            if op["is_bw"] and not op["is_activation_checkpointing"]
        ]
        checkpointing_operations = [
            op
            for op in module_operation_counts["operations_list"]
            if op["is_activation_checkpointing"]
        ]

        return forward_operations, backward_operations, checkpointing_operations

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
        self.comm_module_counts["Global"] = {}
        self.comm_module_counts["Global"]["forward"] = defaultdict(int)
        self.comm_module_counts["Global"]["backward"] = defaultdict(int)

        self.comm_module_operation_counts.clear()

        super().__enter__()
        self.advanced_module_tracker.__enter__()
        return self

    def __exit__(self, *args):
        self.advanced_module_tracker.__exit__()
        super().__exit__(*args)

    def log_comm_debug_tracing_table_to_file(
        self, file_name="comm_mode_log.txt", noise_level=3
    ):
        """
        Alternative to console CommDebugMode output, writes to file specified by the user
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        table = ansi_escape.sub("", self.generate_comm_debug_tracing_table(noise_level))

        with open(file_name, "w") as log_file:
            log_file.write(table)

    def print_paramater_info(self):
        self.advanced_module_tracker.print_paramater_info()

    def print_sharding_info(self):
        self.advanced_module_tracker.print_sharding_info()

    def set_noise_parameters(self, noise_level):
        """
        sets variables controlling what information displays based on noise level
        """
        include_DTensor_ops = False
        include_module_data = False
        include_ops = False
        include_trivial_ops = False

        if noise_level > 0:
            include_DTensor_ops = True
            include_module_data = True

        if noise_level > 1:
            include_ops = True

        if noise_level > 2:
            include_trivial_ops = True

        return (
            include_DTensor_ops,
            include_module_data,
            include_ops,
            include_trivial_ops,
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into comms ops, before CommDebugMode sees them.

        # sets up operation-level collective count
        if self.advanced_module_tracker.name not in self.comm_module_operation_counts:
            # dictionary should hold module input and output shape, operations list and collective counter
            self.comm_module_operation_counts[self.advanced_module_tracker.name] = {
                "operations_list": []
            }
        operation_dict = {}
        operation_dict["name"] = func

        operation_dict["input_shape"] = []
        operation_dict["input_sharding"] = []
        operation_dict["device_mesh"] = ""

        # tracks if the operation is part of the backward pass
        operation_dict["is_bw"] = self.advanced_module_tracker.is_bw

        # tracks if the operation is part of activation checkpointing
        operation_dict[
            "is_activation_checkpointing"
        ] = self.advanced_module_tracker.activation_checkpointing

        if any(t == DTensor for t in types):
            for ele in args:
                if isinstance(ele, DTensor):
                    # saves shapes and placements of all DTensor args
                    operation_dict["input_shape"].append(ele.shape)
                    operation_dict["input_sharding"].append(ele.placements)
                    operation_dict["device_mesh"] = str(ele.device_mesh)

            self.comm_module_operation_counts[self.advanced_module_tracker.name][
                "operations_list"
            ].append(operation_dict)

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

            key = "forward"
            if self.advanced_module_tracker.is_bw:
                key = "backward"

            # adds collective count to current module
            if self.advanced_module_tracker.name not in self.comm_module_counts:
                self.comm_module_counts[self.advanced_module_tracker.name] = {}
                self.comm_module_counts[self.advanced_module_tracker.name][
                    "forward"
                ] = defaultdict(int)
                self.comm_module_counts[self.advanced_module_tracker.name][
                    "backward"
                ] = defaultdict(int)
            self.comm_module_counts[self.advanced_module_tracker.name][key][
                func_packet
            ] += 1

            # adds collective count to parent modules
            for par in self.advanced_module_tracker.module_parents_dict[
                self.advanced_module_tracker.name
            ]:
                # makes sure we aren't double counting when current sub-module hasn't been removed from parents
                if par != self.advanced_module_tracker.name:
                    if par not in self.comm_module_counts:
                        self.comm_module_counts[par] = {}
                        self.comm_module_counts[par]["forward"] = defaultdict(int)
                        self.comm_module_counts[par]["backward"] = defaultdict(int)
                    self.comm_module_counts[par][key][func_packet] += 1

        # if tensor op uses fake tensors, return
        if detect_fake_mode(args):
            return out

        # add tensor operation to module operation list
        self.comm_module_operation_counts[self.advanced_module_tracker.name][
            "operations_list"
        ].append(operation_dict)

        return out
