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
    # Internal
    ("static", datetime.date(9999, 1, 1)),
    ("prim::ModuleDictIndex", datetime.date(9999, 1, 1)),
    # Internal, profiler-specific ops
    ("profiler::_call_end_callbacks_on_jit_fut*", datetime.date(9999, 1, 1)),
    ("profiler::_record_function_enter", datetime.date(9999, 1, 1)),
    ("aten::_qr_helper", datetime.date(2021, 1, 31)),
    ("aten::fft", datetime.date(2021, 1, 31)),
    ("aten::ifft", datetime.date(2021, 1, 31)),
    ("aten::irfft", datetime.date(2021, 1, 31)),
    ("aten::rfft", datetime.date(2021, 1, 31)),
    ("aten::_lstsq_helper", datetime.date(9999, 1, 1)),
    ("aten::_svd_helper", datetime.date(2021, 1, 31)),
    ("aten::_syevd_helper", datetime.date(9999, 1, 1)),
    ("aten::_cudnn_rnn_flatten_weight", datetime.date(2020, 12, 31)),
    ("aten::_cudnn_rnn", datetime.date(2020, 12, 31)),
    ("aten::_cudnn_rnn_backward", datetime.date(2020, 12, 31)),
    ("aten::quantile", datetime.date(2021, 1, 31)),
    ("aten::nanquantile", datetime.date(2021, 1, 31)),
    ("aten::make_dual", datetime.date(2021, 2, 20)),
    ("aten::unpack_dual", datetime.date(2021, 2, 20)),
    ("aten::_fft_with_size", datetime.date(2021, 1, 31)),
    ("aten::thnn_conv_depthwise2d_backward", datetime.date(2021, 1, 31)),
    ("aten::slow_conv3d_backward", datetime.date(2021, 1, 31)),
    ("aten::thnn_conv2d_backward", datetime.date(2021, 1, 31)),
    ("aten::slow_conv_transpose3d_backward", datetime.date(2021, 1, 31)),
    ("aten::slow_conv_transpose2d_backward", datetime.date(2021, 1, 31)),
    ("aten::set_", datetime.date(2021, 1, 31)),
    ("aten::native_layer_norm", datetime.date(2021, 1, 31)),
    ("aten::native_layer_norm_backward", datetime.date(2021, 1, 31)),
    ("aten::elu_backward", datetime.date(2021, 1, 31)),
    ("aten::_multinomial_alias_setup", datetime.date(2021, 1, 31)),
    ("aten::_multinomial_alias_draw", datetime.date(2021, 1, 31)),
    ("prim::profile_optional", datetime.date(2021, 1, 31)),
    ("aten::fake_quantize_per_tensor_affine_backward", datetime.date(2021, 2, 20)),
    ("aten::fake_quantize_per_channel_affine_backward", datetime.date(2021, 2, 20)),
    ("aten::rowwise_prune", datetime.date(9999, 1, 1)),
    ("aten::_foreach_mul_", datetime.date(2021, 4, 2)),
    ("aten::_foreach_addcdiv_", datetime.date(2021, 4, 2)),
    ("aten::_foreach_div", datetime.date(2021, 4, 2)),
    ("aten::_foreach_addcmul_", datetime.date(2021, 4, 2)),
    ("aten::_foreach_sub", datetime.date(2021, 4, 2)),
    ("aten::_foreach_add", datetime.date(2021, 4, 2)),
    ("aten::_foreach_sub_", datetime.date(2021, 4, 2)),
    ("aten::_foreach_add_", datetime.date(2021, 4, 2)),
    ("aten::_foreach_mul", datetime.date(2021, 4, 2)),
    ("aten::_foreach_div_", datetime.date(2021, 4, 2)),
    ("aten::_foreach_addcdiv", datetime.date(2021, 4, 2)),
    ("aten::_foreach_addcmul", datetime.date(2021, 4, 2)),
    ("aten::mkldnn_linear", datetime.date(2021, 3, 2)),
    ("aten::_mode*", datetime.date(2021, 5, 2)),
    ("aten::linalg_multi_dot", datetime.date(2021, 3, 25)),
    ("aten::coalesce", datetime.date(2021, 4, 15)),
    ("aten::empty_meta", datetime.date(2021, 4, 1)),
    ("aten::div", datetime.date(2021, 4, 28)),
    ("aten::divide", datetime.date(2021, 4, 28)),
    ("aten::batch_norm_backward_elemt", datetime.date(2021, 5, 1)),
    ("aten::assert_async", datetime.date(2021, 5, 1)),
    ("aten::cumprod_backward", datetime.date(2021, 5, 1)),
    ("aten::_triangular_solve_helper", datetime.date(9999, 1, 1)),
    ("aten::complex*", datetime.date(2021, 5, 1)),
    ("aten::take_backward", datetime.date(2021, 5, 1)),
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


# The nightly will fail to parse newly added syntax to schema declarations
# Add new schemas that will fail the nightly here
dont_parse_list = [
    ("_TorchScriptTesting.*", datetime.date(2099, 9, 17)),
    ("test_backend", datetime.date(2099, 9, 17)),
    ("dist_c10d", datetime.date(2021, 1, 30)),
]


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

            if dont_parse(line.strip()):
                print("Not parsing schema line: ", line.strip())
                continue
            s = parse_schema(line.strip())
            slist.append(s)

    if not check_bc(slist):
        sys.exit(1)
