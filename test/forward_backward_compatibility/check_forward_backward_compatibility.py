import argparse
import datetime
import re
import sys
from collections import defaultdict

import torch
from torch._C import parse_schema

# Maximum days we will allow to introduce BC
# breaking change into operators.
MAX_ALLOWED_PERIOD = datetime.timedelta(days=30)

# [Bypassing BC/FC tests]
# The date specifies how long the allowlist exclusion should apply to.
# You should pick a date in the future that you believe you can land your diff before then.
# But note that this date should be less than a month of when you are including this BC
# breaking change. In general, we don't recommend adding entry to this list.
# Please review following docs:
#
# 1. https://github.com/pytorch/pytorch/wiki/%5BDraft%5D-PyTorch's-Python-Frontend-Backward-and-Forward-Compatibility-Policy
# 2. torch/csrc/jit/operator_upgraders/README.md
#
# Allowlist entries can be removed after the date listed on them passes.
#
# Allowlist item format:
# [
#   0: function name regex
#   1: date until which the allowlist entry is valid
# ]
#
# NB: function name DOES NOT include overload name!
TEMPORARY_BC_ALLOW_LIST = [
    ("aten::_svd_helper", datetime.date(2022, 3, 1)),
    ("aten::scatter_reduce.two", datetime.date(2022, 3, 15)),
]

# Same things as TEMPORARY__BC_ALLOW_LIST but for FC changes
TEMPORARY_FC_ALLOW_LIST = [
    ("aten::_svd_helper", datetime.date(2022, 3, 1)),
    ("aten::scatter_reduce.two", datetime.date(2022, 3, 15)),
]

# WARNING: Operators included in this list indefinitely bypass all BC schema checks.
# This is almost certainly NOT what you want to do. See note above. ([Bypassing BC/FC tests])
INDEFINITE_BC_ALLOW_LIST = [
    "c10_experimental",
    # Internal
    "static",
    "prim::ModuleDictIndex",
    "prim::MKLDNNRelu6",
    "prim::MKLDNNRelu6_",
    "prim::Concat",
    # Internal, profiler-specific ops
    "profiler::_call_end_callbacks_on_jit_fut*",
    "profiler::_record_function_enter",
    "aten::_cholesky_helper",
    "aten::_lstsq_helper",
    "aten::_syevd_helper",
    "aten::_linalg_solve_out_helper_",
    "aten::select_backward",
    "aten::slice_backward",
    "aten::diagonal_backward",
    "aten::rowwise_prune",
    "aten::adaptive_avg_pool3d_backward",
    "aten::_embedding_bag_dense_backward",
    "aten::randperm",
    "aten::_convolution_nogroup",
    "aten::miopen_convolution_backward",
    "aten::miopen_convolution_backward_bias",
    "aten::miopen_convolution_backward_input",
    "aten::miopen_convolution_backward_weight",
    "aten::miopen_convolution_transpose_backward",
    "aten::miopen_convolution_transpose_backward_input",
    "aten::miopen_convolution_transpose_backward_weight",
    "aten::miopen_depthwise_convolution_backward",
    "aten::miopen_depthwise_convolution_backward_input",
    "aten::miopen_depthwise_convolution_backward_weight",
    "prepacked::unpack_prepacked_sizes_conv2d",
    "prepacked::unpack_prepacked_sizes_linear",
    "aten::native_multi_head_self_attention",
    "aten::_native_multi_head_self_attention",
    "aten::grid_sampler_3d_backward",
    "aten::_transform_bias_rescale_qkv",
    "aten::_scatter_reduce.two",
]

# Same thing as INDEFINITE_BC_ALLOW_LIST but for FC changes.
# In general, we don't recommend adding entry to this list.
# Please review following docs:
#
# 1. https://github.com/pytorch/pytorch/wiki/%5BDraft%5D-PyTorch's-Python-Frontend-Backward-and-Forward-Compatibility-Policy
# 2. torch/csrc/jit/operator_upgraders/README.md
INDEFINITE_FC_ALLOW_LIST = [
    "c10_experimental",
    # Internal
    "static",
    "prim::ModuleDictIndex",
    "prim::MKLDNNRelu6",
    "prim::MKLDNNRelu6_",
    "prim::Concat",
    # Internal, profiler-specific ops
    "profiler::_call_end_callbacks_on_jit_fut*",
    "profiler::_record_function_enter",
    "aten::_cholesky_helper",
    "aten::_lstsq_helper",
    "aten::_syevd_helper",
    "aten::_linalg_solve_out_helper_",
    "aten::select_backward",
    "aten::slice_backward",
    "aten::diagonal_backward",
    "aten::rowwise_prune",
    "aten::adaptive_avg_pool3d_backward",
    "aten::_embedding_bag_dense_backward",
    "aten::randperm",
    "aten::_convolution_nogroup",
    "aten::miopen_convolution_backward",
    "aten::miopen_convolution_backward_bias",
    "aten::miopen_convolution_backward_input",
    "aten::miopen_convolution_backward_weight",
    "aten::miopen_convolution_transpose_backward",
    "aten::miopen_convolution_transpose_backward_input",
    "aten::miopen_convolution_transpose_backward_weight",
    "aten::miopen_depthwise_convolution_backward",
    "aten::miopen_depthwise_convolution_backward_input",
    "aten::miopen_depthwise_convolution_backward_weight",
    "prepacked::unpack_prepacked_sizes_conv2d",
    "prepacked::unpack_prepacked_sizes_linear",
    "aten::native_multi_head_self_attention",
    "aten::_native_multi_head_self_attention",
    "aten::grid_sampler_3d_backward",
    "aten::_transform_bias_rescale_qkv",
    "aten::_scatter_reduce.two",
]

def compile_temp_allow_list(temp_allow_list):
    output = []
    for item in temp_allow_list:
        deadline = item[1]
        today = datetime.date.today()
        interval = deadline - today
        if deadline > today and deadline - today < MAX_ALLOWED_PERIOD:
            output.append((re.compile(item[0]), deadline))
        if interval >= MAX_ALLOWED_PERIOD:
            print("Operator foo will skip BC and FC schema checks for {} days, but only "
                  "{} days of skipping the checks are permitted. It's recommended that the skip date be "
                  "chosen only to permit the PR to be merged. Once the PR is merged for a couple days "
                  "the operator no longer needs to skip these checks.".format(interval.days, MAX_ALLOWED_PERIOD.days))
            sys.exit(1)

    return output

TEMPORARY_BC_ALLOW_LIST_COMPILED = compile_temp_allow_list(TEMPORARY_BC_ALLOW_LIST)
TEMPORARY_FC_ALLOW_LIST_COMPILED = compile_temp_allow_list(TEMPORARY_FC_ALLOW_LIST)

INDEFINITE_BC_ALLOW_LIST_COMPILED = [re.compile(item) for item in INDEFINITE_BC_ALLOW_LIST]
INDEFINITE_FC_ALLOW_LIST_COMPILED = [re.compile(item) for item in INDEFINITE_FC_ALLOW_LIST]

def temp_allow_listed(schema, compiled_allow_list):
    for item in compiled_allow_list:
        if item[0].search(str(schema)):
            return True
    return False

def indefinite_allow_listed(schema, compiled_allow_list):
    for item in compiled_allow_list:
        if item.search(str(schema)):
            return True
    return False

# The nightly will fail to parse newly added syntax to schema declarations
# Add new schemas that will fail the nightly here
dont_parse_list = [
    ("_TorchScriptTesting.*", datetime.date(2099, 9, 17)),
    ("test_backend", datetime.date(2099, 9, 17)),
    ("dist_c10d", datetime.date(2099, 9, 17)),
]

def has_valid_upgraders(schema, version_map):
    # we want to parse through the map to find if
    # the schema has valid upgraders. Since the
    # version map has entry for each overload
    # we need to do some ugly parsing.

    # the name of the operator
    schema_name = schema.name

    if schema_name not in version_map:
        return False

    entries = version_map[schema_name]

    possible_overloads = []
    possible_schemas = []
    for key, upgrader_schema_entries in entries.items():
        possible_overloads.append(key)
        possible_schemas.extend(upgrader_schema_entries)

    # let's make sure this existing schema is part of possible
    # schemas
    for old_schema in possible_schemas:
        if old_schema == schema:
            return True

    return False

def dont_parse(schema_line):
    for item in dont_parse_list:
        if item[1] < datetime.date.today():
            continue
        regexp = re.compile(item[0])
        if regexp.search(schema_line):
            return True
    return False

def load_schemas_to_dict():
    new_schemas = torch._C._jit_get_all_schemas()
    new_schemas += torch._C._jit_get_custom_class_schemas()
    new_schema_dict = defaultdict(list)
    for s in new_schemas:
        new_schema_dict[s.name].append(s)
    return new_schema_dict

def process_version_map(version_map):
    # version map maps full schema name to
    # list of upgraders. Since we only have
    # the name of the schema (aka no overload)
    # we want to first process the map to make
    # the key lookup easier. After this it will be:
    # Dict[schema_name, Dict[overload, List[schema]]]

    output = defaultdict(dict)
    for (key, entries) in version_map.items():
        operator_name = key.split(".")[0]
        schema_entries = [parse_schema(entry.old_schema) for entry in entries]
        output[operator_name][key] = schema_entries
    return output

def check_bc(existing_schemas):
    new_schema_dict = load_schemas_to_dict()
    version_map = process_version_map(torch._C._get_operator_version_map())
    is_bc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if temp_allow_listed(existing_schema, TEMPORARY_BC_ALLOW_LIST_COMPILED):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        if has_valid_upgraders(existing_schema, version_map):
            print("schema: ", str(existing_schema), " has valid upgrader, skipping")
        if indefinite_allow_listed(existing_schema, INDEFINITE_BC_ALLOW_LIST_COMPILED):
            print("schema: {} is in allowlist for BC-breaking evolution without deadline."
                  "This is dangerous, do not use unless you are sure there will not be "
                  "downstream consequences".format(str(existing_schema)))
            continue
        matching_new_schemas = new_schema_dict.get(existing_schema.name, [])
        found = False
        for matching_new_schema in matching_new_schemas:
            if matching_new_schema.is_backward_compatible_with(existing_schema):
                found = True
                break
        if not found:
            print(
                "Can NOT find backward compatible schemas after changes "
                "for schema {} from the following candidates:\n[\n{}\n]".format(
                    str(existing_schema),
                    "\n\t".join(str(s) for s in matching_new_schemas),
                )
            )
            # TODO Print out more details about why candidates don't match.
            broken_ops.append(str(existing_schema))
            is_bc = False
    if is_bc:
        print("Found backward compatible schemas for all existing schemas")
    else:
        print(
            "The PR is introducing backward incompatible changes to the "
            "operator library. Please contact PyTorch team to confirm "
            "whether this change is wanted or not. \n\nBroken ops: "
            "[\n\t{}\n]".format("\n\t".join(broken_ops))
        )
    return is_bc

def check_fc(existing_schemas):
    new_schema_dict = load_schemas_to_dict()
    is_fc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if temp_allow_listed(existing_schema, TEMPORARY_FC_ALLOW_LIST_COMPILED):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        if indefinite_allow_listed(existing_schema, INDEFINITE_FC_ALLOW_LIST_COMPILED):
            print("schema: {} is in allowlist for FC-breaking evolution without deadline."
                  "This is dangerous, do not use unless you are sure there will not be "
                  "downstream consequences".format(str(existing_schema)))
            continue
        matching_new_schemas = new_schema_dict.get(existing_schema.name, [])
        found = False
        possible_failure_reasons = []
        for matching_new_schema in matching_new_schemas:
            is_compatible, reason = matching_new_schema.check_forward_compatible_with(existing_schema)
            if is_compatible:
                found = True
                break
            if reason != "":
                possible_failure_reasons.append(reason)
        if not found:
            print(
                "Can NOT find forward compatible schemas after changes "
                "for schema {} from the following candidates:\n[\n{}\n]".format(
                    str(existing_schema),
                    "\n\t".join(str(s) for s in matching_new_schemas),
                )
            )
            print(
                "Refer to following reasons for failure "
                "to find FC schema:\n[\n{}\n]".format(
                    "\n\t".join(str(r) for r in possible_failure_reasons)
                )
            )
            broken_ops.append(str(existing_schema))
            is_fc = False
    if is_fc:
        print("Found forward compatible schemas for all existing schemas")
    else:
        print(
            "The PR is introducing a potentially forward incompatible changes to the "
            "operator library. Please contact PyTorch team to confirm "
            "whether this change is wanted or not. \n\nBroken ops: "
            "[\n\t{}\n]".format("\n\t".join(broken_ops))
        )
    return is_fc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--existing-schemas",
        help="filename to load existing schemas",
        type=str,
        default="schemas.txt",
    )
    args = parser.parse_args()
    existing_schema_dict = dict()
    slist = []
    with open(args.existing_schemas, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if dont_parse(line.strip()):
                print("Not parsing schema line: ", line.strip())
                continue
            s = parse_schema(line.strip())
            slist.append(s)

    is_fc = check_fc(slist)
    is_bc = check_bc(slist)

    if not (is_fc and is_bc):
        sys.exit(1)
