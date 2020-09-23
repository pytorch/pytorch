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
