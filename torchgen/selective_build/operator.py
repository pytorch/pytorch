from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# This class holds information about a single operator used to determine
# the outcome of a selective/custom PyTorch build that doesn't include
# registration code for all the supported operators. This is done to
# reduce the size of the generated binary so that it can be deployed in
# situations where binary size comes at a premium.
#
@dataclass(frozen=True)
class SelectiveBuildOperator:
    # The name of the operator. This includes the aten::, etc... prefix
    # The operator name may or may not have the overload name. If this
    # operator name does not specify an overload name, the way to determine
    # if this entry refers to the family of operators with this base name
    # or just the operator with this name is to look at the value of the
    # 'include_all_overloads' flag in this class.
    name: str

    # True if this is a root operator (i.e. called directly from a
    # TorchScript model, etc...). An operator is considered to be a
    # root operator if it is called directly from any one of the models
    # that this instance of the pytorch library was built for. Hence, it
    # may not be a root operator in all of the models that are used in
    # this instance of the pytorch library.
    is_root_operator: bool

    # Is this operator used for on-device training? If True, then we need to
    # use the information to generate code in VariableType_N.cpp for registration
    # of training related operators. Again, this is True if this operator
    # is used for training in one or more models used by this instance of the
    # pytorch library.
    is_used_for_training: bool

    # If True, it indicates that this operator instance (object) refers to an
    # operator without the overload name and should apply to all overloads
    # which have this operator name as the base name. This flag is applicable
    # only for objects that have operator names without a DOT (period) character
    # in them.
    #
    # Note: This flag is a temporary workaround to grandfather in the current
    # static selective (custom) build mechanism, which largely ignores overload
    # names when determining whether to select operators for registration
    # purposes.
    include_all_overloads: bool

    # Debug Information at the operator level
    _debug_info: Optional[Tuple[str, ...]]

    @staticmethod
    def from_yaml_dict(
        op_name: str, op_info: Dict[str, object]
    ) -> "SelectiveBuildOperator":
        allowed_keys = {
            "name",
            "is_root_operator",
            "is_used_for_training",
            "include_all_overloads",
            "debug_info",
        }

        if len(set(op_info.keys()) - allowed_keys) > 0:
            raise Exception(
                "Got unexpected top level keys: {}".format(
                    ",".join(set(op_info.keys()) - allowed_keys),
                )
            )

        if "name" in op_info:
            assert op_name == op_info["name"]

        is_root_operator = op_info.get("is_root_operator", True)
        assert isinstance(is_root_operator, bool)

        is_used_for_training = op_info.get("is_used_for_training", True)
        assert isinstance(is_used_for_training, bool)

        include_all_overloads = op_info.get("include_all_overloads", True)
        assert isinstance(include_all_overloads, bool)

        debug_info: Optional[Tuple[str, ...]] = None
        if "debug_info" in op_info:
            di_list = op_info["debug_info"]
            assert isinstance(di_list, list)
            debug_info = tuple(str(x) for x in di_list)

        return SelectiveBuildOperator(
            name=op_name,
            is_root_operator=is_root_operator,
            is_used_for_training=is_used_for_training,
            include_all_overloads=include_all_overloads,
            _debug_info=debug_info,
        )

    @staticmethod
    def from_legacy_operator_name_without_overload(
        name: str,
    ) -> "SelectiveBuildOperator":
        return SelectiveBuildOperator(
            name=name,
            is_root_operator=True,
            is_used_for_training=True,
            include_all_overloads=True,
            _debug_info=None,
        )

    def to_dict(self) -> Dict[str, object]:
        ret: Dict[str, object] = {
            "is_root_operator": self.is_root_operator,
            "is_used_for_training": self.is_used_for_training,
            "include_all_overloads": self.include_all_overloads,
        }
        if self._debug_info is not None:
            ret["debug_info"] = self._debug_info

        return ret


def merge_debug_info(
    lhs: Optional[Tuple[str, ...]],
    rhs: Optional[Tuple[str, ...]],
) -> Optional[Tuple[str, ...]]:
    # Ensure that when merging, each entry shows up just once.
    if lhs is None and rhs is None:
        return None

    return tuple(set((lhs or ()) + (rhs or ())))


def combine_operators(
    lhs: "SelectiveBuildOperator", rhs: "SelectiveBuildOperator"
) -> "SelectiveBuildOperator":
    if str(lhs.name) != str(rhs.name):
        raise Exception(
            "Expected both arguments to have the same name, but got '{}' and '{}' instead".format(
                str(lhs.name),
                str(rhs.name),
            )
        )

    return SelectiveBuildOperator(
        name=lhs.name,
        # Consider this operator to be a root operator if it is a
        # root operator in any of the models used in this instance of
        # the pytorch library.
        is_root_operator=lhs.is_root_operator or rhs.is_root_operator,
        # Consider this operator to be a training operator if it is
        # an operator used for training in any of the models used
        # in this instance of the pytorch library.
        is_used_for_training=lhs.is_used_for_training or rhs.is_used_for_training,
        include_all_overloads=lhs.include_all_overloads or rhs.include_all_overloads,
        _debug_info=merge_debug_info(lhs._debug_info, rhs._debug_info),
    )


def merge_operator_dicts(
    lhs: Dict[str, SelectiveBuildOperator],
    rhs: Dict[str, SelectiveBuildOperator],
) -> Dict[str, SelectiveBuildOperator]:
    operators: Dict[str, SelectiveBuildOperator] = {}
    for op_name, op in list(lhs.items()) + list(rhs.items()):
        new_op = op
        if op_name in operators:
            new_op = combine_operators(operators[op_name], op)

        operators[op_name] = new_op

    return operators


def strip_operator_overload_name(op_name: str) -> str:
    return op_name.split(".")[0]
