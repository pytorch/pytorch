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
    ('aten::pop*', datetime.date(2020, 4, 1)),
    ('aten::insert*', datetime.date(2020, 4, 1)),
    ('aten::Delete*', datetime.date(2020, 4, 1)),
    ('aten::clear*', datetime.date(2020, 4, 1)),
    ('aten::_set_item*', datetime.date(2020, 4, 1)),
    ('aten::copy*', datetime.date(2020, 4, 1)),
    ('aten::extend*', datetime.date(2020, 4, 1)),
    ('aten::reverse*', datetime.date(2020, 4, 1)),
    ('aten::append*', datetime.date(2020, 4, 1)),
    ('aten::list*', datetime.date(2020, 4, 1)),
    ('aten::__getitem__*', datetime.date(2020, 4, 1)),
    ('aten::len*', datetime.date(2020, 4, 1)),
    ('aten::mul_*', datetime.date(2020, 4, 1)),
    ('aten::slice*', datetime.date(2020, 4, 1)),
    ('aten::add*', datetime.date(2020, 4, 1)),
    ('aten::mul*', datetime.date(2020, 4, 1)),
    ('aten::select*', datetime.date(2020, 4, 1)),
    ('aten::add_*', datetime.date(2020, 4, 1)),
    # _like default change, see https://github.com/pytorch/pytorch/issues/33580
    ('aten::randn_like', datetime.date(2020, 3, 15)),
    ('aten::full_like', datetime.date(2020, 3, 15)),
    ('aten::empty_like', datetime.date(2020, 3, 15)),
    ('aten::rand_like', datetime.date(2020, 3, 15)),
    ('aten::ones_like', datetime.date(2020, 3, 15)),
    ('aten::randint_like', datetime.date(2020, 3, 15)),
    ('aten::zeros_like', datetime.date(2020, 3, 15)),
    ('aten::floor_divide', datetime.date(2020, 4, 1)),
    ('aten::Bool', datetime.date(2020, 4, 1)),
    ('aten::Float', datetime.date(2020, 4, 1)),
    ('aten::to', datetime.date(2020, 4, 1)),
    ('aten::backward', datetime.date(2020, 4, 1)),
    ('aten::len', datetime.date(2020, 4, 1)),
    ('aten::remove', datetime.date(2020, 4, 1)),
    ('aten::index', datetime.date(2020, 4, 1)),
    ('aten::count', datetime.date(2020, 4, 1)),
    ('aten::__contains__', datetime.date(2020, 4, 1)),
    ('aten::sort', datetime.date(2020, 4, 1)),
    ('aten::sorted', datetime.date(2020, 4, 1)),
    ('aten::eq', datetime.date(2020, 4, 1)),
    ('aten::ne', datetime.date(2020, 4, 1)),
    ('aten::lt', datetime.date(2020, 4, 1)),
    ('aten::gt', datetime.date(2020, 4, 1)),
    ('aten::le', datetime.date(2020, 4, 1)),
    ('aten::ge', datetime.date(2020, 4, 1)),
    ('aten::divmod', datetime.date(2020, 4, 1)),
    ('aten::__upsample_bilinear', datetime.date(2020, 4, 1)),
    ('aten::__upsample', datetime.date(2020, 4, 1)),
    ('aten::__upsample_nearest', datetime.date(2020, 4, 1)),
    ('aten::__interpolate', datetime.date(2020, 4, 1)),
    ('aten::fabs', datetime.date(2020, 4, 1)),
    ('aten::gamma', datetime.date(2020, 4, 1)),
    ('prim::abs', datetime.date(2020, 4, 1)),
    ('aten::factorial', datetime.date(2020, 4, 1)),
    ('aten::radians', datetime.date(2020, 4, 1)),
    ('aten::degrees', datetime.date(2020, 4, 1)),
    ('prim::acosh', datetime.date(2020, 4, 1)),
    ('prim::atanh', datetime.date(2020, 4, 1)),
    ('aten::asinh', datetime.date(2020, 4, 1)),
    ('aten::floordiv', datetime.date(2020, 4, 1)),
    ('prim::NumToTensor', datetime.date(2020, 4, 1)),
    ('aten::sin', datetime.date(2020, 4, 1)),
    ('aten::round', datetime.date(2020, 4, 1)),
    ('aten::remainder', datetime.date(2020, 4, 1)),
    ('aten::isfinite', datetime.date(2020, 4, 1)),
    ('aten::sub', datetime.date(2020, 4, 1)),
    ('aten::sqrt', datetime.date(2020, 4, 1)),
    ('aten::log1p', datetime.date(2020, 4, 1)),
    ('aten::acos', datetime.date(2020, 4, 1)),
    ('aten::floor', datetime.date(2020, 4, 1)),
    ('aten::exp', datetime.date(2020, 4, 1)),
    ('aten::tan', datetime.date(2020, 4, 1)),
    ('aten::sinh', datetime.date(2020, 4, 1)),
    ('aten::ceil', datetime.date(2020, 4, 1)),
    ('aten::atan', datetime.date(2020, 4, 1)),
    ('aten::erf', datetime.date(2020, 4, 1)),
    ('aten::erfc', datetime.date(2020, 4, 1)),
    ('aten::cosh', datetime.date(2020, 4, 1)),
    ('aten::expm1', datetime.date(2020, 4, 1)),
    ('aten::isinf', datetime.date(2020, 4, 1)),
    ('aten::lgamma', datetime.date(2020, 4, 1)),
    ('aten::asin', datetime.date(2020, 4, 1)),
    ('aten::log', datetime.date(2020, 4, 1)),
    ('aten::log10', datetime.date(2020, 4, 1)),
    ('aten::cos', datetime.date(2020, 4, 1)),
    ('aten::tanh', datetime.date(2020, 4, 1)),
    ('prim::min', datetime.date(2020, 4, 1)),
    ('prim::max', datetime.date(2020, 4, 1)),
    ('aten::_linear_packed', datetime.date(2020, 4, 1)),
    ('aten::_linear_prepack', datetime.date(2020, 4, 1)),
    ('aten::_conv2d_packed', datetime.date(2020, 4, 1)),
    ('aten::_conv2d_prepack', datetime.date(2020, 4, 1)),
    ('aten::dequantize', datetime.date(2020, 4, 1)),
    ('aten::confirmed_by_owner', datetime.date(2020, 3, 17)),
    ('aten::owner', datetime.date(2020, 3, 27)),
    ('aten::owner_name', datetime.date(2020, 3, 27)),
    ('_xnnpack::conv2d_packed', datetime.date(2020, 4, 2)),
    ('_xnnpack::conv2d_prepack', datetime.date(2020, 4, 2)),
    ('_xnnpack::linear_packed', datetime.date(2020, 4, 2)),
    ('_xnnpack::linear_prepack', datetime.date(2020, 4, 2)),
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
            if "torch.classes" in line:
                # TODO Fix type __torch__.torch.classes.xxx
                continue

            s = parse_schema(line.strip())
            slist = new_schema_dict.get(s.name, [])
            slist.append(s)
            new_schema_dict[s.name] = slist

    if not check_bc(new_schema_dict):
        sys.exit(1)
