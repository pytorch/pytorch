from typing import Dict, Set, Optional, Tuple, List
import yaml

from dataclasses import dataclass

from tools.codegen.model import NativeFunction
from tools.codegen.selective_build.operator import *

# A SelectiveBuilder holds information extracted from the selective build
# YAML specification.
#
# It includes information about the build's selectivity, the debug_info
# associated with this selective build (opaque string), and the set of
# operators that should be included in the build.
#
@dataclass(frozen=True)
class SelectiveBuilder:

    # If true, then the build is not selective, and includes all
    # operators.
    include_all_operators: bool

    # Debug Information at the selective/custom build level.
    _debug_info: Optional[Tuple[str, ...]]

    # A dictionary of operator -> operator metadata.
    operators: Dict[str, SelectiveBuildOperator]

    # A dictionary of selected kernel tags and dtypes. Typically a
    # PyTorch Operator Kernel (function) may have many code paths
    # that are specialized for many many Tensor dtypes, so it's not
    # one per kernel function, but there could be many per kernel
    # function. The tag isn't a kernel function name, but some fragment
    # of the kernel function implementation itself.
    kernel_metadata: Dict[str, List[str]]

    # If true, then fragments for all dtypes for all kernel functions
    # are included. This is typically set when any one of the
    # operator lists is generated from a mechanism other than
    # tracing based selective build.
    include_all_kernel_dtypes: bool

    @staticmethod
    def get_nop_selector() -> 'SelectiveBuilder':
        return SelectiveBuilder.from_yaml_dict({'include_all_operators': True})

    @staticmethod
    def from_yaml_dict(data: Dict[str, object]) -> 'SelectiveBuilder':
        valid_top_level_keys = {
            'include_all_kernel_dtypes',
            'include_all_operators',
            'debug_info',
            'operators',
            'kernel_metadata',
        }
        top_level_keys = set(data.keys())
        if len(top_level_keys - valid_top_level_keys) > 0:
            raise Exception("Got unexpected top level keys: {}".format(
                ",".join(top_level_keys - valid_top_level_keys),
            ))
        include_all_operators = data.get('include_all_operators', False)
        assert isinstance(include_all_operators, bool)

        debug_info = None
        if 'debug_info' in data:
            di_list = data['debug_info']
            assert isinstance(di_list, list)

            debug_info = tuple(map(lambda x: str(x), di_list))

        operators = {}
        operators_dict = data.get('operators', {})
        assert isinstance(operators_dict, dict)

        for (k, v) in operators_dict.items():
            operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)

        kernel_metadata = {}
        kernel_metadata_dict = data.get('kernel_metadata', {})
        assert isinstance(kernel_metadata_dict, dict)

        for (k, v) in kernel_metadata_dict.items():
            kernel_metadata[str(k)] = list(map(lambda dtype: str(dtype), v))

        include_all_kernel_dtypes = data.get('include_all_kernel_dtypes', False)
        assert isinstance(include_all_kernel_dtypes, bool)

        return SelectiveBuilder(
            include_all_operators,
            debug_info,
            operators,
            kernel_metadata,
            include_all_kernel_dtypes,
        )

    @staticmethod
    def from_yaml_str(config_contents: str) -> 'SelectiveBuilder':
        contents = yaml.safe_load(config_contents)
        return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_yaml_path(config_path: str) -> 'SelectiveBuilder':
        with open(config_path, 'r') as f:
            contents = yaml.safe_load(f)
            return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_legacy_op_registration_allow_list(
            allow_list: Set[str],
            is_root_operator: bool,
            is_used_for_training: bool) -> 'SelectiveBuilder':
        operators = {}
        for op in allow_list:
            operators[op] = {
                'name': op,
                'is_root_operator': is_root_operator,
                'is_used_for_training': is_used_for_training,
                'include_all_overloads': True,
            }
        return SelectiveBuilder.from_yaml_dict({
            'operators': operators,
            'include_all_kernel_dtypes': True,
        })

    def is_operator_selected(self, name: str) -> bool:
        if self.include_all_operators:
            return True

        if name in self.operators:
            return True
        name = strip_operator_overload_name(name)
        return name in self.operators and self.operators[name].include_all_overloads

    def is_native_function_selected(self, func: NativeFunction) -> bool:
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected(op_name)

    def is_operator_selected_for_training(self, name: str) -> bool:
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        not_training_op = SelectiveBuildOperator(
            name='',
            is_root_operator=False,
            is_used_for_training=False,
            include_all_overloads=False,
            _debug_info=None,
        )
        op = not_training_op
        if name in self.operators:
            op = self.operators[name]

        name = strip_operator_overload_name(name)
        base_op = not_training_op
        if name in self.operators:
            base_op = self.operators[name]

        return (
            op.is_used_for_training or
            (base_op.include_all_overloads and base_op.is_used_for_training)
        )

    def is_native_function_selected_for_training(self, func: NativeFunction) -> bool:
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected_for_training(op_name)

    def is_root_operator(self, name: str) -> bool:
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        if name in self.operators:
            op: SelectiveBuildOperator = self.operators[name]
            return op.is_root_operator
        name = strip_operator_overload_name(name)
        if name not in self.operators:
            return False
        base_op: SelectiveBuildOperator = self.operators[name]
        return base_op.include_all_overloads and base_op.is_root_operator

    def is_kernel_dtype_selected(self, kernel_tag: str, dtype: str) -> bool:
        if self.include_all_operators or self.include_all_kernel_dtypes:
            return True

        return kernel_tag in self.kernel_metadata and dtype in self.kernel_metadata[kernel_tag]

    def to_dict(self) -> Dict[str, object]:
        ret: Dict[str, object] = {
            'include_all_kernel_dtypes': self.include_all_kernel_dtypes,
            'include_all_operators': self.include_all_operators,
        }
        operators = {}
        for (op_name, op) in self.operators.items():
            operators[op_name] = op.to_dict()
        ret['operators'] = operators

        if self._debug_info is not None:
            ret['debug_info'] = self._debug_info

        ret['kernel_metadata'] = {k: list(v) for (k, v) in self.kernel_metadata.items()}

        return ret


def merge_kernel_metadata(
        lhs: Dict[str, List[str]],
        rhs: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    kernel_metadata: Dict[str, List[str]] = {}
    for (tag_name, dtypes) in list(lhs.items()) + list(rhs.items()):
        dtypes_copy = set(dtypes)
        if tag_name in kernel_metadata:
            dtypes_copy |= set(kernel_metadata[tag_name])

        kernel_metadata[tag_name] = list(dtypes_copy)

    return kernel_metadata

def combine_selective_builders(lhs: SelectiveBuilder, rhs: SelectiveBuilder) -> SelectiveBuilder:
    include_all_operators = lhs.include_all_operators or rhs.include_all_operators
    debug_info = merge_debug_info(lhs._debug_info, rhs._debug_info)
    operators = merge_operator_dicts(lhs.operators, rhs.operators)
    kernel_metadata = merge_kernel_metadata(lhs.kernel_metadata, rhs.kernel_metadata)
    include_all_kernel_dtypes = lhs.include_all_kernel_dtypes or rhs.include_all_kernel_dtypes
    return SelectiveBuilder(
        include_all_operators,
        debug_info,
        operators,
        kernel_metadata,
        include_all_kernel_dtypes,
    )


def op_name_from_native_function(f: NativeFunction) -> str:
    # This was originally read from the 'operator_name_with_overload' field in the
    # declaration dict, which was the part before the first '(' in 'schema_string'.
    return f'aten::{f.func.name}'
