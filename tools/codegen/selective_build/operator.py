from typing import Dict
from dataclasses import dataclass

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

    @staticmethod
    def from_yaml_dict(op_name: str, op_info: Dict[str, object]) -> 'SelectiveBuildOperator':
        is_root_operator = op_info.get('is_root_operator', True)
        is_used_for_training = op_info.get('is_used_for_training', True)
        include_all_overloads = op_info.get('include_all_overloads', True)

        return SelectiveBuildOperator(
            op_name,
            is_root_operator,
            is_used_for_training,
            include_all_overloads,
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

    return SelectiveBuildOperator(
        lhs.name,
        lhs.is_root_operator or rhs.is_root_operator,
        lhs.is_used_for_training or rhs.is_used_for_training,
        lhs.include_all_overloads or rhs.include_all_overloads,
    )

def strip_operator_overload_name(op_name: str) -> str:
    return op_name.split(".")[0]
