#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml
from gen_op_registration_allowlist import (
    canonical_name,
    gen_transitive_closure,
    load_op_dep_graph,
)

from torchgen.selective_build.operator import (
    merge_operator_dicts,
    SelectiveBuildOperator,
)
from torchgen.selective_build.selector import merge_kernel_metadata


# Generate YAML file containing the operators used for a specific PyTorch model.
# ------------------------------------------------------------------------------
#
# This binary is responsible for generating the model_operators.yaml file for
# each model from a pt_operator_library() BUCK macro invocation.
#
# Output YAML file format:
# ------------------------
#
# <BEGIN FILE CONTENTS>
# include_all_non_op_selectives: False
# include_all_operators: False
# debug_info:
#   - model1@v100
#   - model2@v50
# operators:
#   aten::add:
#     is_root_operator: Yes
#     is_used_for_training: Yes
#     include_all_overloads: No
#     debug_info:
#       - model1@v100
#       - model2@v50
#   aten::add.int:
#     is_root_operator: No
#     is_used_for_training: No
#     include_all_overloads: Yes
# kernel_metadata:
#   add_kernel:
#     - Int8
#     - UInt32
#   sub_kernel:
#     - Int16
#     - Float
# <END FILE CONTENTS>
#
# There are a few main inputs to this application
# -----------------------------------------------
#
# 1. Inference Root Operators (--root-ops): Root operators (called directly
#    from TorchScript) used by inference use-cases.
#
# 2. Training Root Operators (--training-root-ops): Root operators used
#    by training use-cases. Currently, this list is the list of all operators
#    used by training, and not just the root operators. All Training ops are
#    also considered for inference, so these are merged into inference ops.
#
# 3. Operator Depencency Graph (--dep-graph-yaml-path): A path to the
#    operator dependency graph used to determine which operators depend on
#    which other operators for correct functioning. This is used for
#    generating the transitive closure of all the operators used by the
#    model based on the root operators when static selective build is used.
#    For tracing based selective build, we don't need to perform this
#    transitive cloure.
#
# 4. Model Metadata (--model-name, --model-versions, --model-assets,
#    --model-backends): Self-descriptive. These are used to tell this
#    script which model operator lists to fetch from the Model
#    Build Metadata YAML files.
#
# 5. Model YAML files (--models-yaml-path): These yaml files contains
#    (for each model/version/asset/backend) the set of used root and traced
#    operators. This is used to extract the actual set of operators
#    needed to be included in the build.
#


def canonical_opnames(opnames: list[str]) -> list[str]:
    return [canonical_name(opname) for opname in opnames]


def make_filter_from_options(
    model_name: str,
    model_versions: list[str],
    model_assets: list[str] | None,
    model_backends: list[str] | None,
):
    def is_model_included(model_info) -> bool:
        model = model_info["model"]
        if model["name"] != model_name:
            return False
        if str(model["version"]) not in model_versions:
            return False
        if model_assets is not None and model["asset"] not in model_assets:
            return False
        # TODO: Handle backend later
        return True

    return is_model_included


# Returns if a the specified rule is a new or old style pt_operator_library
def is_new_style_rule(model_name: str, model_versions: list[str] | None):
    return model_name is not None and model_versions is not None


# Verifies that specified model_name, and all specified versions and assets
# appear in at least one model yaml. Throws if verification is failed,
# returns None on success
def verify_all_specified_present(
    model_assets: list[str] | None,
    model_versions: list[str],
    selected_models_yaml: list[dict[str, Any]],
    rule_name: str,
    model_name: str,
    new_style_rule: bool,
) -> None:
    def find_missing_items(model_items, key, selected_models_yaml):
        missing_items = []
        if not new_style_rule or not model_items:
            return missing_items
        for item in model_items:
            found = False
            for model in selected_models_yaml:
                if str(model["model"][key]) == item:
                    found = True
            if not found:
                missing_items.append(item)
        return missing_items

    missing_assets = find_missing_items(model_assets, "asset", selected_models_yaml)
    missing_versions = find_missing_items(
        model_versions, "version", selected_models_yaml
    )

    if len(missing_versions) > 0 or len(missing_assets) > 0:  # at least one is missing
        name_warning = ""
        if len(selected_models_yaml) == 0:
            name_warning = (
                "WARNING: 0 yaml's were found for target rule. This could be because the "
                + "provided model name: {name} is incorrect. Please check that field as well as "
                + "the assets and versions."
            ).format(name=model_name)
        raise RuntimeError(
            (
                "Error: From the pt_operator_library rule for Rule: {name}, at least one entry for the "
                + "following fields was expected -- Model: {model_name} Expected Assets: {expected_assets}, Expected Versions: "
                + "{expected_versions}. {name_warning} In all_mobile_models.yaml either no assets were on one of the "
                + "specified versions, one of the specified assets was not present on any of the specified "
                + "versions, or both. Assets not found: {missing_assets}, Versions not found: {missing_versions} "
                + "For questions please ask in https://fb.workplace.com/groups/2148543255442743/"
            ).format(
                name=rule_name,
                model_name=model_name,
                expected_versions=model_versions,
                expected_assets=model_assets
                if model_assets
                else "<All model assets present on specified versions>",
                name_warning=name_warning,
                missing_versions=missing_versions
                if len(missing_versions) > 0
                else "<All specified versions had at least one asset>",
                missing_assets=missing_assets
                if len(missing_assets) > 0
                else "<All specified assets are present on at least 1 version>",
            )
        )


# Uses the selected models configs and then combines them into one dictionary,
# formats them as a string, and places the string into output as a top level debug_info
def create_debug_info_from_selected_models(
    output: dict[str, object],
    selected_models: list[dict],
    new_style_rule: bool,
) -> None:
    model_dict = {
        "asset_info": {},  # maps asset name -> dict of asset metadata like hashes
        "is_new_style_rule": new_style_rule,
    }

    for model in selected_models:
        model_info = model["model"]
        asset = model_info["asset"]
        hash = model_info["md5_hash"]

        asset_info = model_dict["asset_info"].setdefault(asset, {})

        asset_info.setdefault("md5_hash", []).append(hash)

    # Will later be used in gen_oplist to generate the model/version/asset checking
    output["debug_info"] = [json.dumps(model_dict)]


def fill_output(output: dict[str, object], options: object) -> None:
    """Populate the output dict with the information required to serialize
    the YAML file used for selective build.
    """
    dept_graph = load_op_dep_graph(options.dep_graph_yaml_path)

    model_versions = (
        options.model_versions.split(",") if options.model_versions is not None else []
    )
    model_assets = (
        options.model_assets.split(",") if options.model_assets is not None else None
    )

    all_models_yaml = []
    if options.models_yaml_path:
        for yaml_path in options.models_yaml_path:
            with open(yaml_path, "rb") as f:
                all_models_yaml.append(yaml.safe_load(f))

    model_filter_func = make_filter_from_options(
        options.model_name, model_versions, model_assets, options.model_backends
    )

    selected_models_yaml = list(filter(model_filter_func, all_models_yaml))

    verify_all_specified_present(
        model_assets=model_assets,
        model_versions=model_versions,
        selected_models_yaml=selected_models_yaml,
        rule_name=options.rule_name,
        model_name=options.model_name,
        new_style_rule=is_new_style_rule(options.model_name, options.model_versions),
    )

    create_debug_info_from_selected_models(
        output,
        selected_models_yaml,
        is_new_style_rule(options.model_name, options.model_versions),
    )

    # initialize variables for static build from the pt_operator_library rule
    if options.root_ops is not None:
        static_root_ops = set(filter(lambda x: len(x) > 0, options.root_ops.split(",")))
    else:
        static_root_ops = set()

    static_training_root_ops = set(
        filter(
            lambda x: len(x) > 0,
            (options.training_root_ops or "").split(","),
        )
    )
    if len(static_training_root_ops) > 0:
        static_root_ops = static_root_ops | static_training_root_ops
    # end if

    root_ops_unexpand = set()
    traced_ops = set()
    training_root_ops_unexpand = set()
    traced_training_ops = set()
    all_kernel_metadata = []
    all_custom_classes = set()
    all_build_features = set()

    # Go through each yaml file and retrieve operator information.
    for model_info in selected_models_yaml:
        if "traced_operators" not in model_info:
            # If this YAML file doesn't specify any traced operators, then it is using
            # the static analysis selective build approach of finding transitively
            # used operators, and we should update root_ops with the set of root
            # operators, all of whose overloads must be included. In addition, these
            # root_ops will be further expanded using the transitive closure of
            # operator dependencies.
            static_root_ops = static_root_ops | set(model_info["root_operators"])
        else:
            # If this YAML file specifies traced operators, then it is using
            # the tracing based selective build approach of finding used
            # operators, and we should update root_ops_unexpand with the set of root
            # operators whose overloads don't need to be included. In addition, these
            # root_ops_unexpand will NOT be further expanded. If the train flag is
            # set then the ops will be used for training, so we put them in a separate
            # set
            if model_info["train"]:
                training_root_ops_unexpand = training_root_ops_unexpand | set(
                    model_info["root_operators"]
                )
                traced_training_ops = traced_training_ops | set(
                    model_info["traced_operators"]
                )
            else:
                root_ops_unexpand = root_ops_unexpand | set(
                    model_info["root_operators"]
                )
                traced_ops = traced_ops | set(model_info["traced_operators"])

        if "kernel_metadata" in model_info:
            all_kernel_metadata.append(model_info["kernel_metadata"])

        if "custom_classes" in model_info:
            all_custom_classes = all_custom_classes | set(model_info["custom_classes"])

        if "build_features" in model_info:
            all_build_features = all_build_features | set(model_info["build_features"])

    # This following section on transitive closure is relevant to static build only
    canonical_root_ops = canonical_opnames(static_root_ops)
    # If no canonical_root_ops exist, don't compute the transitive closure
    # otherwise, we will include __BASE__ and __ROOT__ ops and mark them as required
    # for inference.
    if len(canonical_root_ops) > 0:
        closure_op_list = gen_transitive_closure(dept_graph, canonical_root_ops)
    else:
        closure_op_list = set()

    canonical_training_root_ops = canonical_opnames(static_training_root_ops)
    # If no canonical_training_root_ops exist, don't compute the transitive closure
    # otherwise, we will include __BASE__ and __ROOT__ ops and mark them as required
    # for training.
    if len(canonical_training_root_ops) > 0:
        closure_training_op_list = gen_transitive_closure(
            dept_graph, canonical_training_root_ops, train=True
        )
    else:
        closure_training_op_list = set()

    # bucketed_ops holds sets of operators that correspond to specific semantic buckets. For
    # example:
    #
    # 1. Root Operators not used for training w/o full overload inclusion
    # 2. Root Operators not used for training w/ full overload inclusion
    # 3. Root Operators used for training w/o full overload inclusion
    # 4. Root Operators used for training w/ full overload inclusion
    # 5. Non-root Operators not used for training w/o full overload inclusion
    # etc...
    #
    # Basically for each of the 3 boolean conditional, there are 2
    # options (True/False).
    #
    bucketed_ops = []

    # START STATIC BUILD OPS
    static_root_ops_bucket = {}
    for op_name in static_root_ops:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": False,
                "include_all_overloads": not options.not_include_all_overloads_static_root_ops,
                "debug_info": [options.model_name],
            },
        )
        static_root_ops_bucket[op_name] = op
    bucketed_ops.append(static_root_ops_bucket)

    closure_ops_bucket = {}
    for op_name in closure_op_list:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": False,
                "is_used_for_training": False,
                "include_all_overloads": not options.not_include_all_overloads_closure_ops,
                "debug_info": [options.model_name],
            },
        )
        closure_ops_bucket[op_name] = op
    bucketed_ops.append(closure_ops_bucket)

    static_training_root_ops_bucket = {}
    for op_name in static_training_root_ops:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": True,
                "include_all_overloads": True,
                "debug_info": [options.model_name],
            },
        )
        static_training_root_ops_bucket[op_name] = op
    bucketed_ops.append(static_training_root_ops_bucket)

    closure_training_ops_bucket = {}
    for op_name in closure_training_op_list:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": False,
                "is_used_for_training": True,
                "include_all_overloads": True,
                "debug_info": [options.model_name],
            },
        )
        closure_training_ops_bucket[op_name] = op
    bucketed_ops.append(closure_training_ops_bucket)
    # END STATIC BUILD OPS

    # START TRACING BASED BUILD OPS
    root_ops_unexpand_bucket = {}
    for op_name in root_ops_unexpand:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": False,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        root_ops_unexpand_bucket[op_name] = op
    bucketed_ops.append(root_ops_unexpand_bucket)

    traced_ops_bucket = {}
    for op_name in traced_ops:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": False,
                "is_used_for_training": False,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        traced_ops_bucket[op_name] = op
    bucketed_ops.append(traced_ops_bucket)

    training_root_ops_unexpand_bucket = {}
    for op_name in training_root_ops_unexpand:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": True,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        training_root_ops_unexpand_bucket[op_name] = op
    bucketed_ops.append(training_root_ops_unexpand_bucket)

    traced_training_ops_bucket = {}
    for op_name in traced_training_ops:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": False,
                "is_used_for_training": True,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        traced_training_ops_bucket[op_name] = op
    bucketed_ops.append(traced_training_ops_bucket)
    # END TRACING BASED BUILD OPS

    # Merge dictionaries together to remove op duplication
    operators: dict[str, SelectiveBuildOperator] = {}
    for ops_dict in bucketed_ops:
        operators = merge_operator_dicts(operators, ops_dict)

    # Loop over all operators, and if any of the them specifies that
    # all overloads need to be included, then set include_all_non_op_selectives
    # to True, since it indicates that this operator list came from something
    # other than a traced operator list.
    include_all_non_op_selectives = False
    for op_name, op_info in operators.items():
        include_all_non_op_selectives = (
            include_all_non_op_selectives or op_info.include_all_overloads
        )

    operators_as_dict = {}
    for k, v in operators.items():
        operators_as_dict[k] = v.to_dict()

    output["operators"] = operators_as_dict

    output["custom_classes"] = all_custom_classes

    output["build_features"] = all_build_features

    output["include_all_non_op_selectives"] = include_all_non_op_selectives
    if len(all_kernel_metadata) > 0:
        kernel_metadata = {}
        for kt in all_kernel_metadata:
            kernel_metadata = merge_kernel_metadata(kernel_metadata, kt)
        output["kernel_metadata"] = kernel_metadata


def add_arguments_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--root-ops",
        "--root_ops",
        help="A comma separated list of root operators used by the model",
        required=False,
    )
    parser.add_argument(
        "--training-root-ops",
        "--training_root_ops",
        help="A comma separated list of root operators used for training",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        help="The location of the output yaml file.",
        required=True,
    )
    parser.add_argument(
        "--dep-graph-yaml-path",
        "--dep_graph_yaml_path",
        type=str,
        help="A path to the Operator Dependency Graph YAML file.",
        required=True,
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        type=str,
        help="The name of the model that uses the specified root operators.",
        required=True,
    )
    parser.add_argument(
        "--model-versions",
        "--model_versions",
        type=str,
        help="A comma separated list of model versions.",
        required=False,
    )
    parser.add_argument(
        "--model-assets",
        "--model_assets",
        type=str,
        help="A comma separate list of model asset names (if absent, defaults to all assets for this model).",
        required=False,
    )
    parser.add_argument(
        "--model-backends",
        "--model_backends",
        type=str,
        default="CPU",
        help="A comma separated list of model backends.",
        required=False,
    )
    parser.add_argument(
        "--models-yaml-path",
        "--models_yaml_path",
        type=str,
        help="The paths to the mobile model config YAML files.",
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "--include-all-operators",
        "--include_all_operators",
        action="store_true",
        default=False,
        help="Set this flag to request inclusion of all operators (i.e. build is not selective).",
        required=False,
    )
    parser.add_argument(
        "--rule-name",
        "--rule_name",
        type=str,
        help="The name of pt_operator_library rule resulting in this generation",
        required=True,
    )
    parser.add_argument(
        "--not-include-all-overloads-static-root-ops",
        "--not_include_all_overloads_static_root_ops",
        action="store_true",
        default=False,
        help="Set this flag to not include all overloaded operators for static root ops bucket in fill_output() subroutine",
        required=False,
    )
    parser.add_argument(
        "--not-include-all-overloads-closure-ops",
        "--not_include_all_overloads_closure_ops",
        action="store_true",
        default=False,
        help="Set this flag to not include all overloaded operators for closure ops bucket in fill_output() subroutine",
        required=False,
    )
    return parser


def parse_options(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


def get_parser_options(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser = add_arguments_parser(parser)
    return parse_options(parser)


def main(argv) -> None:
    parser = argparse.ArgumentParser(description="Generate used operators YAML")
    options = get_parser_options(parser)

    model_dict = {
        "model_name": options.model_name,
        "asset_info": {},
        "is_new_style_rule": False,
    }
    output = {
        "debug_info": [json.dumps(model_dict)],
    }

    if options.include_all_operators:
        output["include_all_operators"] = True
        output["operators"] = {}
        output["kernel_metadata"] = {}
    else:
        fill_output(output, options)

    with open(options.output_path, "wb") as out_file:
        out_file.write(
            yaml.safe_dump(
                output,
                default_flow_style=False,
            ).encode("utf-8")
        )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
