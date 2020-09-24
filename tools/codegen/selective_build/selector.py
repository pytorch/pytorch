from typing import Dict, Optional, Any, Set
import yaml

from dataclasses import dataclass
from enum import Enum

from tools.codegen.selective_build.operator import *

# The SelectiveBuildOperators class encapsulates information about
# a specific version of selective build and includes information about
#
# [1] The models that contribute to the set of operators, and
# [2] The set of operators themselves.
#
@dataclass(frozen=True)
class SelectiveBuildOperators:

    # If true, then the build is not selective, and includes all
    # operators.
    include_all_operators: bool

    # The set of models that participate in this selective build.
    # Used only for debugging in we want to know if a specific model's
    # operators were included in thie build.
    model_names: Set[str]

    # A dictionary of operator -> operator metadata.
    operators: Dict[str, SelectiveBuildOperator]

    @staticmethod
    def from_yaml(data: Dict[str, object]) -> 'SelectiveBuildOperators':
        model_names = set(data.get('model_names', []))
        operators = {}
        for (k, v) in data.get('operators', {}).items():
            operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)
        return SelectiveBuildOperators(model_names, operators)

# A SelectiveBuilder holds information extracted from the selective build
# YAML specification.
#
# It includes information about the build's selectivity, the models that
# participated in the selective build, and the set of operators that should
# be included in the build.
#
@dataclass(frozen=True)
class SelectiveBuilder:

    # If true, then the build is not selective, and includes all
    # operators.
    include_all_operators: bool

    # The set of models that participate in this selective build.
    # Used only for debugging in we want to know if a specific model's
    # operators were included in thie build.
    model_names: Set[str]

    # A dictionary of operator -> operator metadata.
    operators: Dict[str, SelectiveBuildOperator]

    @staticmethod
    def get_nop_selector() -> 'SelectiveBuilder':
        return SelectiveBuilder.from_yaml_dict({'include_all_operators': True})

    @staticmethod
    def from_yaml_dict(data: Dict[str, object]) -> 'SelectiveBuilder':
        include_all_operators = data.get('include_all_operators', False)
        model_names = set(data.get('model_names', []))
        operators = {}
        for (k, v) in data.get('operators', {}).items():
            operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)
        return SelectiveBuilder(include_all_operators, model_names, operators)

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
    def from_legacy_op_registration_allow_list(allow_list: Set[str]) -> 'SelectiveBuilder':
        operators = {}
        for op in allow_list:
            operators[op] = SelectiveBuildOperator.from_legacy_operator_name_without_overload(
                op,
            )
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

        if name in self.operators:
            op: SelectiveBuildOperator = self.operators[name]
            return op.is_used_for_training
        name = strip_operator_overload_name(name)
        if name not in self.operators:
            return False
        op: SelectiveBuildOperator = self.operators[name]
        return op.include_all_overloads and op.is_used_for_training

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
        op: SelectiveBuildOperator = self.operators[name]
        return op.include_all_overloads and op.is_root_operator
