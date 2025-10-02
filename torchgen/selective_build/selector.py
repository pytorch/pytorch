from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from torchgen.selective_build.operator import (
    merge_debug_info,
    merge_operator_dicts,
    SelectiveBuildOperator,
    strip_operator_overload_name,
)

if TYPE_CHECKING:
    from torchgen.model import NativeFunction


# A SelectiveBuilder holds information extracted from the selective build
# YAML specification.
#
# It includes information about the build's selectivity, the debug_info
# associated with this selective build (opaque string), and the set of
# operators that should be included in the build.
#
@dataclass(frozen=True)
class SelectiveBuilder:
    # If true, then the build is not selective, and includes all operators.
    include_all_operators: bool

    # Debug Information at the selective/custom build level.
    _debug_info: tuple[str, ...] | None

    # A dictionary of operator -> operator metadata.
    operators: dict[str, SelectiveBuildOperator]

    # A dictionary of selected kernel tags and dtypes.
    kernel_metadata: dict[str, list[str]]

    # ExecuTorch only. A dictionary of kernel tag -> list of input dtypes.
    et_kernel_metadata: dict[str, list[str]]

    # A set of all the custom torch bind classes used by the selected models.
    custom_classes: set[str]

    # A set of all the build features used by the selected models.
    build_features: set[str]

    # If true, then fragments for all dtypes for all kernel functions
    # are included as well as all custom classes.
    include_all_non_op_selectives: bool

    @staticmethod
    def get_nop_selector() -> SelectiveBuilder:
        return SelectiveBuilder.from_yaml_dict({"include_all_operators": True})

    @staticmethod
    def from_yaml_dict(data: dict[str, object]) -> SelectiveBuilder:
        valid_top_level_keys = {
            "include_all_non_op_selectives",
            "include_all_operators",
            "debug_info",
            "operators",
            "kernel_metadata",
            "et_kernel_metadata",
            "custom_classes",
            "build_features",
        }
        extra_keys = set(data) - valid_top_level_keys
        if extra_keys:
            raise Exception(  # noqa: TRY002
                f"Got unexpected top level keys: {','.join(extra_keys)}"
            )

        include_all_operators = data.get("include_all_operators", False)
        assert isinstance(include_all_operators, bool)

        debug_info = None
        if "debug_info" in data:
            di_list = data["debug_info"]
            assert isinstance(di_list, list)
            debug_info = tuple(str(x) for x in di_list)

        operators_dict = data.get("operators", {})
        assert isinstance(operators_dict, dict)
        operators = {
            k: SelectiveBuildOperator.from_yaml_dict(k, v)
            for k, v in operators_dict.items()
        }

        kernel_metadata_dict = data.get("kernel_metadata", {})
        assert isinstance(kernel_metadata_dict, dict)
        kernel_metadata = {
            str(k): [str(dtype) for dtype in v]
            for k, v in kernel_metadata_dict.items()
        }

        et_kernel_metadata = data.get("et_kernel_metadata", {})
        assert isinstance(et_kernel_metadata, dict)

        custom_classes = set(data.get("custom_classes", []))
        assert isinstance(custom_classes, Iterable)

        build_features = set(data.get("build_features", []))
        assert isinstance(build_features, Iterable)

        include_all_non_op_selectives = data.get("include_all_non_op_selectives", False)
        assert isinstance(include_all_non_op_selectives, bool)

        return SelectiveBuilder(
            include_all_operators,
            debug_info,
            operators,
            kernel_metadata,
            et_kernel_metadata,
            custom_classes,  # type: ignore[arg-type]
            build_features,  # type: ignore[arg-type]
            include_all_non_op_selectives,
        )

    @staticmethod
    def from_yaml_str(config_contents: str) -> SelectiveBuilder:
        contents = yaml.safe_load(config_contents)
        return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_yaml_path(config_path: str) -> SelectiveBuilder:
        with open(config_path) as f:
            contents = yaml.safe_load(f)
            return SelectiveBuilder.from_yaml_dict(contents)

    @staticmethod
    def from_legacy_op_registration_allow_list(
        allow_list: set[str], is_root_operator: bool, is_used_for_training: bool
    ) -> SelectiveBuilder:
        operators = {
            op: {
                "name": op,
                "is_root_operator": is_root_operator,
                "is_used_for_training": is_used_for_training,
                "include_all_overloads": True,
            }
            for op in allow_list
        }
        return SelectiveBuilder.from_yaml_dict(
            {
                "operators": operators,
                "include_all_non_op_selectives": True,
            }
        )

    def is_operator_selected(self, name: str) -> bool:
        if self.include_all_operators or name in self.operators:
            return True
        base_name = strip_operator_overload_name(name)
        return base_name in self.operators and self.operators[base_name].include_all_overloads

    def is_native_function_selected(self, func: NativeFunction) -> bool:
        return self.is_operator_selected(op_name_from_native_function(func))

    def is_operator_selected_for_training(self, name: str) -> bool:
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        not_training_op = SelectiveBuildOperator(
            name="",
            is_root_operator=False,
            is_used_for_training=False,
            include_all_overloads=False,
            _debug_info=None,
        )

        op = self.operators.get(name, not_training_op)
        base_op = self.operators.get(strip_operator_overload_name(name), not_training_op)

        return op.is_used_for_training or (
            base_op.include_all_overloads and base_op.is_used_for_training
        )

    def is_native_function_selected_for_training(self, func: NativeFunction) -> bool:
        return self.is_operator_selected_for_training(op_name_from_native_function(func))

    def is_root_operator(self, name: str) -> bool:
        if not self.is_operator_selected(name):
            return False
        if self.include_all_operators:
            return True

        op = self.operators.get(name)
        if op:
            return op.is_root_operator

        base_op = self.operators.get(strip_operator_overload_name(name))
        return bool(base_op and base_op.include_all_overloads and base_op.is_root_operator)

    def is_kernel_dtype_selected(self, kernel_tag: str, dtype: str) -> bool:
        if self.include_all_operators or self.include_all_non_op_selectives:
            return True
        return dtype in self.kernel_metadata.get(kernel_tag, [])

    def et_get_selected_kernels(self, op_name: str, kernel_key: list[str]) -> list[str]:
        """
        Return a list of kernel keys that cover the used ops
        """
        if op_name not in self.et_kernel_metadata:
            return kernel_key if self.include_all_operators else []

        result_set = set()
        for model_kernel_keys in self.et_kernel_metadata[op_name]:
            key_found = False
            for key in kernel_key:
                if key != "default" and key.split("/")[1] == model_kernel_keys.split("/")[1]:
                    result_set.add(key)
                    key_found = True
                    break
            if not key_found:
                if "default" not in kernel_key:
                    raise Exception("Missing kernel for the model")  # noqa: TRY002
                result_set.add("default")

        return list(result_set)

    def to_dict(self) -> dict[str, object]:
        ret: dict[str, object] = {
            "include_all_non_op_selectives": self.include_all_non_op_selectives,
            "include_all_operators": self.include_all_operators,
            "operators": {op_name: op.to_dict() for op_name, op in self.operators.items()},
            "kernel_metadata": {k: sorted(v) for k, v in self.kernel_metadata.items()},
            "et_kernel_metadata": self.et_kernel_metadata,
            "custom_classes": sorted(self.custom_classes),
            "build_features": sorted(self.build_features),
        }
        if self._debug_info is not None:
            ret["debug_info"] = sorted(self._debug_info)
        return ret


def merge_kernel_metadata(
    lhs: dict[str, list[str]], rhs: dict[str, list[str]]
) -> dict[str, list[str]]:
    kernel_metadata: dict[str, list[str]] = {}
    for tag_name, dtypes in {**lhs, **rhs}.items():
        kernel_metadata[tag_name] = list(set(lhs.get(tag_name, [])) | set(rhs.get(tag_name, [])))
    return kernel_metadata


def merge_et_kernel_metadata(
    lhs: dict[str, list[str]], rhs: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for op in set(lhs) | set(rhs):
        merged[op].update(lhs.get(op, []))
        merged[op].update(rhs.get(op, []))
    return {op: sorted(vals) for op, vals in merged.items()}


def combine_selective_builders(lhs: SelectiveBuilder, rhs: SelectiveBuilder) -> SelectiveBuilder:
    return SelectiveBuilder(
        lhs.include_all_operators or rhs.include_all_operators,
        merge_debug_info(lhs._debug_info, rhs._debug_info),
        merge_operator_dicts(lhs.operators, rhs.operators),
        merge_kernel_metadata(lhs.kernel_metadata, rhs.kernel_metadata),
        merge_et_kernel_metadata(lhs.et_kernel_metadata, rhs.et_kernel_metadata),
        lhs.custom_classes | rhs.custom_classes,
        lhs.build_features | rhs.build_features,
        lhs.include_all_non_op_selectives or rhs.include_all_non_op_selectives,
    )


def op_name_from_native_function(f: NativeFunction) -> str:
    # This was originally read from the 'operator_name_with_overload' field in the
    # declaration dict, which was the part before the first '(' in 'schema_string'.
    return f"{f.namespace}::{f.func.name}"
