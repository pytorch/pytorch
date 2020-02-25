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
    # We export some functions and classes for test_jit.py directly from libtorch.so,
    # it's not important to have BC for them
    ('_TorchScriptTesting.*', datetime.date(9999, 1, 1)),
    ('aten::tril_indices', datetime.date(2020, 3, 1)),
    ('aten::triu_indices', datetime.date(2020, 3, 1)),
    ('prim::Drop', datetime.date(2020, 3, 1)),
    ('prim::Store', datetime.date(2020, 3, 1)),
    ('aten::_ncf_view', datetime.date(2020, 3, 1)),
    ('aten::_ncf_unsqueeze', datetime.date(2020, 3, 1)),
    ('prim::Load', datetime.date(2020, 3, 1)),
    ('prim::ImplicitTensorToNum', datetime.date(2020, 3, 1)),
    ('aten::is_owner', datetime.date(2020, 3, 1)),
    ('aten::to_here', datetime.date(2020, 3, 1)),
    ('prim::isinstance', datetime.date(2020, 3, 1)),
    ('prim::CreateObject', datetime.date(2020, 3, 1)),
    ('prim::Uninitialized', datetime.date(2020, 3, 1)),
    ('prim::fork', datetime.date(2020, 3, 1)),
    ('prim::unchecked_cast', datetime.date(2020, 3, 1)),
    ('prim::DictConstruct', datetime.date(2020, 3, 1)),
    ('prim::ListConstruct', datetime.date(2020, 3, 1)),
    ('prim::ListUnpack', datetime.date(2020, 3, 1)),
    ('prim::TupleConstruct', datetime.date(2020, 3, 1)),
    ('prim::TupleIndex', datetime.date(2020, 3, 1)),
    ('prim::TupleSlice', datetime.date(2020, 3, 1)),
    ('prim::TupleUnpack', datetime.date(2020, 3, 1)),
    ('prim::AutogradAdd', datetime.date(2020, 3, 1)),
    ('prim::AutogradAnyNonZero', datetime.date(2020, 3, 1)),
    ('onnx::Shape', datetime.date(2020, 3, 1)),
    ('onnx::Reshape', datetime.date(2020, 3, 1)),
    ('prim::BroadcastSizes', datetime.date(2020, 3, 1)),
    ('prim::Print', datetime.date(2020, 3, 1)),
    ('prim::MMTreeReduce', datetime.date(2020, 3, 1)),
    ('prim::Constant', datetime.date(2020, 3, 1)),
    ('_prim::TupleUnpack', datetime.date(2020, 3, 1)),
    ('_aten::format', datetime.date(2020, 3, 1)),
    ('aten::random_', datetime.date(2020, 3, 1)),
    ('quantized::add_(scalar_)?(relu_)?out', datetime.date(2020, 3, 1)),
    ('quantized::cat_(relu_)?out', datetime.date(2020, 3, 1)),
    ('quantized::mul_(scalar_)?(relu_)?out', datetime.date(2020, 3, 1)),
    ('aten::index_put', datetime.date(2020, 3, 1)),
    ('aten::index', datetime.date(2020, 3, 1)),
    ('aten::_index_put_impl', datetime.date(2020, 3, 1)),
    ('aten::index_put_', datetime.date(2020, 3, 1)),
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
            if "torch.classes" in line or "RRef" in line or "Any" in line:
                # TODO Fix type __torch__.torch.classes.xxx
                # TODO Delete RRef special case after add the RRef type
                # TODO: wait until nightly knows how to parse Any
                continue

            s = parse_schema(line.strip())
            slist = new_schema_dict.get(s.name, [])
            slist.append(s)
            new_schema_dict[s.name] = slist

    if not check_bc(new_schema_dict):
        sys.exit(1)
