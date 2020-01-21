from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import re
import sys
import torch
from torch._C import parse_schema


# The date specifies how long the whitelist exclusion should apply to.
#
#   - If we NEVER give BC guarantee for an operator, you can put the
#     date arbitrarily far in the future.
#   - Otherwise, pick a date that is far enough in the future that you
#     believe you can land your diff before then.
#
# Whitelist entries can be removed after the date listed on them passes.
white_list = [
    ('c10_experimental', datetime.date(2222, 1, 1)),
    ('cudnn_convolution', datetime.date(2020, 2, 1)),
    ('cudnn_convolution_backward', datetime.date(2020, 2, 1)),
    ('cudnn_convolution_backward_bias', datetime.date(2020, 2, 1)),
    ('cudnn_convolution_transpose', datetime.date(2020, 2, 1)),
    ('cudnn_convolution_transpose_backward', datetime.date(2020, 2, 1)),
    ('cudnn_convolution_transpose_backward_bias', datetime.date(2020, 2, 1)),
    ('prim::AutogradAnyNonZero', datetime.date(2020, 2, 1)),
    ('upsample_linear1d.out', datetime.date(9999, 1, 1)),
    ('upsample_linear1d', datetime.date(9999, 1, 1)),
    ('upsample_linear1d_backward.grad_input', datetime.date(9999, 1, 1)),
    ('upsample_linear1d_backward', datetime.date(9999, 1, 1)),
    ('upsample_bilinear2d.out', datetime.date(9999, 1, 1)),
    ('upsample_bilinear2d', datetime.date(9999, 1, 1)),
    ('upsample_bilinear2d_backward.grad_input', datetime.date(9999, 1, 1)),
    ('upsample_bilinear2d_backward', datetime.date(9999, 1, 1)),
    ('upsample_bicubic2d.out', datetime.date(9999, 1, 1)),
    ('upsample_bicubic2d', datetime.date(9999, 1, 1)),
    ('upsample_bicubic2d_backward', datetime.date(9999, 1, 1)),
    ('upsample_bicubic2d_backward', datetime.date(9999, 1, 1)),
    ('upsample_trilinear3d.out', datetime.date(9999, 1, 1)),
    ('upsample_trilinear3d', datetime.date(9999, 1, 1)),
    ('upsample_trilinear3d_backward.grad_input', datetime.date(9999, 1, 1)),
    ('upsample_trilinear3d_backward', datetime.date(9999, 1, 1)),
    ('upsample_nearest1d.out', datetime.date(9999, 1, 1)),
    ('upsample_nearest1d', datetime.date(9999, 1, 1)),
    ('upsample_nearest1d_backward.grad_input', datetime.date(9999, 1, 1)),
    ('upsample_nearest1d_backward', datetime.date(9999, 1, 1)),
    ('upsample_nearest2d.out', datetime.date(9999, 1, 1)),
    ('upsample_nearest2d', datetime.date(9999, 1, 1)),
    ('upsample_nearest2d_backward.grad_input', datetime.date(9999, 1, 1)),
    ('upsample_nearest2d_backward', datetime.date(9999, 1, 1)),
    ('upsample_nearest3d.out', datetime.date(9999, 1, 1)),
    ('upsample_nearest3d', datetime.date(9999, 1, 1)),
    ('upsample_nearest3d_backward.grad_input', datetime.date(9999, 1, 1)),
    ('upsample_nearest3d_backward', datetime.date(9999, 1, 1)),
    ('_test_optional_float', datetime.date(9999, 1, 1)),
    ('aten::Int', datetime.date(2020, 1, 30)),
]

jit_test_functions = [
    '_TorchScriptTesting_StackString::pop',
    '_TorchScriptTesting_StackString::push',
    '_TorchScriptTesting_StackString::__init__',
    '_TorchScriptTesting_Foo::combine',
    '_TorchScriptTesting_Foo::add',
    '_TorchScriptTesting_Foo::increment',
    '_TorchScriptTesting_Foo::info',
    '_TorchScriptTesting_Foo::__init__',
]
for fn in jit_test_functions:
    white_list.append((fn, datetime.date(2020, 3, 1)))


def white_listed(schema, white_list):
    for item in white_list:
        if item[1] < datetime.date.today():
            continue
        regexp = re.compile(item[0])
        if regexp.search(schema.name):
            return True
    return False


def check_bc(new_schema_dict):
    existing_schemas = torch._C._jit_get_all_schemas()
    is_bc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if white_listed(existing_schema, white_list):
            print("skipping schema: ", str(existing_schema))
            continue
        print("processing existing schema: ", str(existing_schema))
        new_schemas = new_schema_dict.get(existing_schema.name, [])
        found = False
        for new_schema in new_schemas:
            if new_schema.is_backward_compatible_with(existing_schema):
                found = True
                break
        if not found:
            print('Can NOT find backward compatible schemas after changes '
                  'for schema {} from the following candidates:\n[\n{}\n]'
                  .format(
                      str(existing_schema),
                      "\n\t".join(str(s) for s in new_schemas)))
            # TODO Print out more details about why candidates don't match.
            broken_ops.append(str(existing_schema))
            is_bc = False
    if is_bc:
        print('Found backward compatible schemas for all existing schemas')
    else:
        print('The PR is introducing backward incompatible changes to the '
              'operator library. Please contact PyTorch team to confirm '
              'whether this change is wanted or not. \n\nBroken ops: '
              '[\n\t{}\n]'.format("\n\t".join(broken_ops)))
    return is_bc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--new-schemas',
        help='filename to load new schemas',
        type=str,
        default='schemas.txt')
    args = parser.parse_args()
    new_schema_dict = dict()
    with open(args.new_schemas, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "torch.classes" in line:
                # TODO Fix type __torch__.torch.classes.xxx
                continue
            s = parse_schema(line.strip())
            slist = new_schema_dict.get(s.name, [])
            slist.append(s)
            new_schema_dict[s.name] = slist

    if not check_bc(new_schema_dict):
        sys.exit(1)
