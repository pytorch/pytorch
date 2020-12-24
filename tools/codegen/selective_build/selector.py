from typing import Dict, Set, Optional, Tuple
import yaml

from dataclasses import dataclass

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

    @staticmethod
    def get_nop_selector() -> 'SelectiveBuilder':
        return SelectiveBuilder.from_yaml_dict({'include_all_operators': True})

    @staticmethod
    def from_yaml_dict(data: Dict[str, object]) -> 'SelectiveBuilder':
        valid_top_level_keys = {
            'include_all_operators',
            'debug_info',
            'operators',
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
        return SelectiveBuilder(include_all_operators, debug_info, operators)

    @staticmethod
    def from_yaml_str(config_contents: str) -> 'SelectiveBuilder':
        contents = yaml.load(config_contents)
        return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_yaml_path(config_path: str) -> 'SelectiveBuilder':
        with open(config_path, 'r') as f:
            contents = yaml.load(f)
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
        })

    def is_operator_selected(self, name: str) -> bool:
        if self.include_all_operators:
            return True

        if name in self.operators:
            return True
        name = strip_operator_overload_name(name)
        return name in self.operators and self.operators[name].include_all_overloads

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

    def to_dict(self) -> Dict[str, object]:
        ret: Dict[str, object] = {
            'include_all_operators': self.include_all_operators,
        }
        operators = {}
        for (op_name, op) in self.operators.items():
            operators[op_name] = op.to_dict()
        ret['operators'] = operators

        if self._debug_info is not None:
            ret['debug_info'] = self._debug_info

        return ret


def combine_selective_builders(lhs: SelectiveBuilder, rhs: SelectiveBuilder) -> SelectiveBuilder:
    include_all_operators = lhs.include_all_operators or rhs.include_all_operators
    debug_info = merge_debug_info(lhs._debug_info, rhs._debug_info)
    operators = merge_operator_dicts(lhs.operators, rhs.operators)
    return SelectiveBuilder(include_all_operators, debug_info, operators)
