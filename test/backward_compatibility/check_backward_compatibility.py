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
    ("aten::append", datetime.date(9999, 1, 1)),
    ("prim::AutogradAnyNonZero", datetime.date(9999, 1, 1)),
    ("aten::grad", datetime.date(9999, 1, 1)),
    ("_c10_experimental", datetime.date(9999, 1, 1)),
    ("aten::thnn_conv3d", datetime.date(9999, 1, 1)),
    ("aten::native_layer_norm_double_backward", datetime.date(9999, 1, 1)),
    ("aten::cudnn_batch_norm", datetime.date(9999, 1, 1)),
    ("aten::cudnn_batch_norm_backward", datetime.date(9999, 1, 1)),
    ("aten::_batch_norm_impl_index_backward", datetime.date(9999, 1, 1)),
    ("aten::empty_like", datetime.date(9999, 1, 1)),
    ("aten::_batch_norm_impl_index", datetime.date(9999, 1, 1)),
    ("aten::index_fill_", datetime.date(9999, 1, 1)),
    ("aten::index_fill", datetime.date(9999, 1, 1)),
    ("aten::log_softmax", datetime.date(9999, 1, 1)),
    ("aten::softmax", datetime.date(9999, 1, 1)),
    ("aten::thnn_conv3d_forward", datetime.date(9999, 1, 1)),
    ("aten::thnn_conv3d_backward.output_mask", datetime.date(9999, 1, 1)),
]


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
              'whether this change is wanted or not. \n Broken ops: [\n{}]'
              .format("\n".join(broken_ops)))
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
        line = f.readline()
        while line:
            s = parse_schema(line.strip())
            line = f.readline()
            slist = new_schema_dict.get(s.name, [])
            slist.append(s)
            new_schema_dict[s.name] = slist

    if not check_bc(new_schema_dict):
        sys.exit(1)
