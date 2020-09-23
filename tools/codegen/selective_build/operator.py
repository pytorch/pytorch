from dataclasses import dataclass

from tools.codegen.model import OperatorName

@dataclass(frozen=True)
class SelectiveBuildOperator():
    # The name of the operator. This includes the aten::, etc... prefix
    name: OperatorName

    # True if this is a root operator (i.e. called directly from TorchScript, etc...)
    is_root_operator: bool

    # Is this operator used for on-device training? If True, then we need to
    # use the information to generate code in VariableType_N.cpp for registration
    # of training related operators
    is_used_for_training: bool

    @staticmethod
    def combine(
        lhs: 'SelectiveBuildOperator',
        rhs: 'SelectiveBuildOperator',
    ) -> 'SelectiveBuildOperator':
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
        )
