from typing import Dict, Optional, Any, Set
import yaml
from abc import ABC, abstractmethod

from tools.codegen.model import OperatorName
from tools.codegen.selective_build.operator import SelectiveBuildOperator

class SelectiveBuildBase(ABC):
    @abstractmethod
    def is_operator_selected(self, name: OperatorName) -> bool:
        pass

    @abstractmethod
    def is_operator_selected_for_training(self, name: OperatorName) -> bool:
        pass

    @abstractmethod
    def is_root_operator(self, name: OperatorName) -> bool:
        pass


class SelectiveBuildV1(SelectiveBuildBase):
    def __init__(self, operators: Dict[str, SelectiveBuildOperator]):
        self.operators = operators

    def is_operator_selected(self, name: OperatorName) -> bool:
        # For V1, check just the base operator name since the overload names are missing.
        return str(name.name) in self.operators

    def is_operator_selected_for_training(self, name: OperatorName) -> bool:
        if not self.is_operator_selected(name):
            return False
        op: SelectiveBuildOperator = self.operators[name.name]
        return op.is_used_for_training

    def is_root_operator(self, name: OperatorName) -> bool:
        if not self.is_operator_selected(name):
            return False
        op: SelectiveBuildOperator = self.operators[name.name]
        return op.is_root_operator

class SelectiveBuildV2(SelectiveBuildBase):
    def __init__(self, operators: Dict[str, SelectiveBuildOperator]):
        self.operators = operators

    def is_operator_selected(self, name: OperatorName) -> bool:
        # Check w/o stripping the overload name
        return str(name) in self.operators

    def is_operator_selected_for_training(self, name: OperatorName) -> bool:
        if not self.is_operator_selected(name):
            return False
        op = self.operators[name.name]
        return op.is_used_for_training

    def is_root_operator(self, name: OperatorName) -> bool:
        if not self.is_operator_selected(name):
            return False
        op: SelectiveBuildOperator = self.operators[name.name]
        return op.is_root_operator

class SelectiveBuildNotSelective(SelectiveBuildBase):
    def __init__(self):
        pass

    def is_operator_selected(self, name: OperatorName) -> bool:
        return True

    def is_operator_selected_for_training(self, name: OperatorName) -> bool:
        return True

    def is_root_operator(self, name: OperatorName) -> bool:
        return True

class SelectiveBuildSelectorFactory(object):
    @staticmethod
    def create_from_path(config_path: str, version: str) -> SelectiveBuildBase:
        with open(config_path, 'r') as f:
            contents = yaml.load(f)
            return SelectiveBuildSelectorFactory.create_from_contents(contents, version)

    @staticmethod
    def create_from_contents(config_contents: str, version: str) -> SelectiveBuildBase:
        contents = yaml.load(config_contents)
        return SelectiveBuildSelectorFactory.create_from_dict(contents, version)

    @staticmethod
    def create_from_dict(config_dict: Optional[Any], version: str) -> SelectiveBuildBase:
        if version not in ['all', 'v1', 'v2']:
            raise Exception("Version must be one of ['all', 'v1', 'v2']")

        if version == 'all':
            return SelectiveBuildNotSelective()

        if version not in config_dict:
            raise Exception(f'Provided version {version} not in provided config dict')

        operators = config_dict[version]

        if version == 'v1':
            return SelectiveBuildV1(operators)
        else:
            return SelectiveBuildV2(operators)

    @staticmethod
    def create_from_legacy_op_registration_allow_list(allow_list: Set[str]) -> SelectiveBuildBase:
        operators = {}
        for op in allow_list:
            operators[op] = SelectiveBuildOperator(OperatorName.parse(op), True, True)
        return SelectiveBuildSelectorFactory.create_from_dict(
            {'v1': operators}, 'v1',
        )
