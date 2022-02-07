import argparse
import datetime
import re
import sys
import warnings
from collections import defaultdict

import torch
from torch._C import parse_schema


# The date specifies how long the allowlist exclusion should apply to.
#
#   - If we NEVER give BC guarantee for an operator, you can put the
#     date arbitrarily far in the future.
#   - Otherwise, pick a date that is far enough in the future that you
#     believe you can land your diff before then.
#
# Allowlist entries can be removed after the date listed on them passes.
#
# Allowlist item format:
# [
#   0: function name regex
#   1: date until which the allowlist entry is valid
#   2: (optional) function argument regex
# ]
#
# NB: function name DOES NOT include overload name!
ALLOW_LIST_WITH_DEADLINE = [
    ("aten::_svd_helper", datetime.date(2022, 3, 31)),
    ("aten::linalg_svdvals", datetime.date(2022, 3, 31)),
    ("aten::linalg_svdvals_out", datetime.date(2022, 3, 31)),
    ("aten::linalg_svd", datetime.date(2022, 3, 31)),
    ("aten::linalg_svd_out", datetime.date(2022, 3, 31)),
    ("aten::_max_pool1d_cpu_forward", datetime.date(2022, 2, 8)),
    ("aten::linspace", datetime.date(2022, 3, 1)),  # TODO this will be removed soon
    ("aten::logspace", datetime.date(2022, 3, 1)),  # TODO this will be removed soon
    ("aten::quantile", datetime.date(2022, 9, 30)),
    ("aten::nanquantile", datetime.date(2022, 9, 30)),
    ("aten::_convolution_double_backward", datetime.date(2022, 3, 31)),
]

MAX_ALLOWED_PERIOD = datetime.timedelta(days=365)

# WARNING: The operators that are included in this list is extremely unsafe
# because they would bypass all BC/FC checks for indefinite amount of time.
# This could cause many severe production issues. Please proceed with extreme caution,
# if you must add an entry to this list.
ALLOW_LIST_WITHOUT_DEADLINE = [
    "c10_experimental",
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
]

def compile_allow_list_with_deadline():
    output = []
    for item in ALLOW_LIST_WITH_DEADLINE:
        deadline = item[1]
        today = datetime.date.today()
        if deadline > today and deadline - today < MAX_ALLOWED_PERIOD:
            output.append((re.compile(item[0]), deadline, re.compile(item[2]) if len(item) > 2 else None))
        if deadline - today >= MAX_ALLOWED_PERIOD:
            print("{} will be BC broken for too long. We only allow {} days"
                  " for the BC breaking window".format(item[0], MAX_ALLOWED_PERIOD.days))
            sys.exit(1)

    return output

ALLOW_LIST_COMPILED_WITH_DEADLINE = compile_allow_list_with_deadline()

def allow_listed_with_deadline(schema):
    for item in ALLOW_LIST_COMPILED_WITH_DEADLINE:
        if item[0].search(str(schema)):
            if len(item) > 2 and item[2] is not None:
                # if arguments regex is present, use it
                return bool(item[2].search(str(schema)))
            return True
    return False

def allow_listed_without_deadline(schema):
    for item in ALLOW_LIST_WITHOUT_DEADLINE:
        if re.compile(item).search(str(schema)):
            return True
    return False

# The nightly will fail to parse newly added syntax to schema declarations
# Add new schemas that will fail the nightly here
dont_parse_list = [
    ("_TorchScriptTesting.*", datetime.date(2099, 9, 17)),
    ("test_backend", datetime.date(2099, 9, 17)),
    ("dist_c10d", datetime.date(2099, 9, 17)),
]


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

def check_bc(existing_schemas):
    new_schema_dict = load_schemas_to_dict()
    is_bc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if allow_listed_with_deadline(existing_schema):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        if allow_listed_without_deadline(existing_schema):
            print("schema: {} is found in forever BC breaking list. This is very dangerous".format(str(existing_schema)))
            continue
        print("processing existing schema: ", str(existing_schema))
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
        if allow_listed_with_deadline(existing_schema):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        print("processing existing schema: ", str(existing_schema))
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
            "The PR is introducing a forward incompatible changes to the "
            "operator library. Please contact PyTorch team to confirm "
            "whether this change is wanted or not. \n\nBroken ops: "
            "[\n\t{}\n]".format("\n\t".join(broken_ops))
        )


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

    if not check_fc(slist):
        sys.exit(1)

    if not check_bc(slist):
        sys.exit(1)
