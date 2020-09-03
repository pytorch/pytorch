from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import re
import sys
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
allow_list = [
    ("c10_experimental", datetime.date(2222, 1, 1)),
    # We export some functions and classes for test_jit.py directly from libtorch.so,
    # it's not important to have BC for them
    ("_TorchScriptTesting.*", datetime.date(9999, 1, 1)),
    # Internal, profiler-specific ops
    ("profiler::_call_end_callbacks_on_jit_fut*", datetime.date(9999, 1, 1)),
    ("profiler::_record_function_enter", datetime.date(9999, 1, 1)),
    ("tensorexpr::Group", datetime.date(2020, 9, 9)),
    ("aten::append*", datetime.date(2020, 4, 15)),
    ("aten::_min", datetime.date(2020, 9, 9)),
    ("aten::_max", datetime.date(2020, 9, 9)),
    ("aten::amax", datetime.date(2020, 10, 9)),
    ("aten::amin", datetime.date(2020, 10, 9)),
    ("aten::min_values", datetime.date(2020, 10, 9)),
    ("aten::max_values", datetime.date(2020, 10, 9)),
    ("aten::split_with_sizes", datetime.date(2020, 7, 29)),
    ("aten::eq", datetime.date(2020, 7, 30)),
    ("aten::log", datetime.date(2020, 7, 30)),
    ("aten::__and__", datetime.date(2020, 7, 30)),
    ("aten::__or__", datetime.date(2020, 7, 30)),
    ("aten::__xor__", datetime.date(2020, 7, 30)),
    ("aten::add", datetime.date(2020, 7, 30)),
    ("aten::__upsample_bilinear", datetime.date(2020, 7, 30)),
    ("aten::hash", datetime.date(2020, 7, 30)),
    ("aten::divmod", datetime.date(2020, 7, 30)),
    ("aten::sorted", datetime.date(2020, 8, 30)),
    ("aten::__contains__", datetime.date(2020, 7, 30)),
    ("aten::ne", datetime.date(2020, 7, 30)),
    ("aten::index", datetime.date(2020, 7, 30)),
    ("aten::isnan", datetime.date(2020, 7, 30)),
    ("aten::pow", datetime.date(2020, 7, 30)),
    ("aten::atan2", datetime.date(2020, 7, 30)),
    ("aten::copy_", datetime.date(2020, 7, 30)),
    ("aten::sort", datetime.date(2020, 7, 30)),
    ('aten::_convolution', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution_transpose', datetime.date(2020, 10, 15)),
    ('aten::_convolution_double_backward', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution_backward_input', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution_backward', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution_backward_weight', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution_transpose_backward', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution_transpose_backward_input', datetime.date(2020, 10, 15)),
    ('aten::cudnn_convolution_transpose_backward_weight', datetime.date(2020, 10, 15)),
    ("aten::_cudnn_init_dropout_state", datetime.date(2020, 7, 30)),
    ("aten::sparse_coo_tensor", datetime.date(2020, 7, 30)),
    ("aten::_sparse_coo_tensor_with_dims", datetime.date(2020, 7, 30)),
    ("aten::_sparse_coo_tensor_with_dims_and_tensors", datetime.date(2020, 7, 30)),
    ("aten::__lshift__", datetime.date(2020, 7, 30)),
    ("aten::__rshift__", datetime.date(2020, 7, 30)),
    ("aten::__round_to_zero_floordiv", datetime.date(2020, 7, 30)),
    ("aten::gcd", datetime.date(2020, 7, 30)),
    ("aten::unflatten", datetime.date(2020, 8, 14)),
    ("aten::linalg_outer", datetime.date(2020, 8, 30)),
    # WARNING: overload name here doesn't do anything
    ("aten::linalg_outer.out", datetime.date(2020, 8, 30)),
    ("aten::_compute_linear_combination", datetime.date(2020, 9, 1)),
    ("__getstate__", datetime.date(2020, 9, 1), "Conv[23]dPackedParams"),
    ("aten::_foreach_add_", datetime.date(2020, 10, 1)),
]


def allow_listed(schema, allow_list):
    for item in allow_list:
        if item[1] < datetime.date.today():
            continue
        regexp = re.compile(item[0])
        if regexp.search(schema.name):
            if len(item) > 2:
                # if arguments regex is present, use it
                regexp_args = re.compile(item[2])
                return bool(regexp_args.search(str(schema)))
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


def check_bc(existing_schemas):
    new_schemas = torch._C._jit_get_all_schemas()
    new_schemas += torch._C._jit_get_custom_class_schemas()
    new_schema_dict = defaultdict(list)
    for s in new_schemas:
        new_schema_dict[s.name].append(s)

    is_bc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if allow_listed(existing_schema, allow_list):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
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
            s = parse_schema(line.strip())
            slist.append(s)

    if not check_bc(slist):
        sys.exit(1)
