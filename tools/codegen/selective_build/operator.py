from typing import Dict, List, Optional
from dataclasses import dataclass

# This class holds information about model metadata. Used to track
# information about where the selective build operator list comes
# from, and potentially which model each operator is used in.
@dataclass
class PyTorchModelMetadata:
    name: str
    version: str

    @staticmethod
    def from_yaml(data: Dict[str, str]) -> 'PyTorchModelMetadata':
        name = data['name']
        version = data['version']
        return PyTorchModelMetadata(name, version)


@dataclass(frozen=True)
class SelectiveBuildOperator():
    # The name of the operator. This includes the aten::, etc... prefix
    # The operator name may or may not have the overload name. If this
    # operator name does not specify an overload name, the way to determine
    # if this entry refers to the family of operators with this base name
    # or just the operator with this name is to look at the value of the
    # 'include_all_overloads' flag in this class.
    name: str

    # True if this is a root operator (i.e. called directly from TorchScript, etc...)
    is_root_operator: bool

    # Is this operator used for on-device training? If True, then we need to
    # use the information to generate code in VariableType_N.cpp for registration
    # of training related operators
    is_used_for_training: bool

    # If True, it indicates that this operator instance (object) refers to an
    # operator without the overload name and should apply to all overloads
    # which have this operator name as the base name. This flag is applicable
    # only for objects that have operator names without a DOT (period) character
    # in them.
    include_all_overloads: bool

    # The list of models that use this operator.
    models: Optional[List[PyTorchModelMetadata]]

    @staticmethod
    def from_yaml_dict(op_name: str, op_info: Dict[str, object]) -> 'SelectiveBuildOperator':
        is_root_operator = op_info.get('is_root_operator', True)
        is_used_for_training = op_info.get('is_used_for_training', True)
        include_all_overloads = op_info.get('include_all_overloads', True)
        models = None
        if 'models' in op_info:
            models = map(
                lambda x: PyTorchModelMetadata.from_yaml(x),
                op_info['models'],
            )

        return SelectiveBuildOperator(
            op_name,
            is_root_operator,
            is_used_for_training,
            include_all_overloads,
            models,
        )

    @staticmethod
    def from_legacy_operator_name_without_overload(name: str) -> 'SelectiveBuildOperator':
        return {
            'name': name,
            'is_root_operator': True,
            'is_used_for_training': True,
            'include_all_overloads': True,
        }

def combine_operators(
        lhs: 'SelectiveBuildOperator',
        rhs: 'SelectiveBuildOperator') -> 'SelectiveBuildOperator':
    if str(lhs.name) != str(rhs.name):
        raise Exception(
            "Expected both arguments to have the same name, but got '{}' and '{}' instead".format(
                str(lhs.name),
                str(rhs.name),
            )
        )

    models = None
    if lhs.models is not None:
        models = lhs.models[:]
    if rhs.models is not None:
        models.extend(rhs.models)

    return SelectiveBuildOperator(
        lhs.name,
        lhs.is_root_operator or rhs.is_root_operator,
        lhs.is_used_for_training or rhs.is_used_for_training,
        lhs.include_all_overloads or rhs.include_all_overloads,
        models,
    )

def strip_operator_overload_name(op_name: str) -> str:
    return op_name.split(".")[0]
