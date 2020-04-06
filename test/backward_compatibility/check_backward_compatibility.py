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
    ('_caffe2', datetime.date(9999, 1, 1)),
    ('_aten', datetime.date(9999, 1, 1)),
    ('prim::', datetime.date(9999, 1, 1)),
    ('onnx::', datetime.date(9999, 1, 1)),
    ('aten::_set_item', datetime.date(9999, 1, 1)),
    ('aten::setdefault', datetime.date(9999, 1, 1)),
    ('aten::_test_optional_float', datetime.date(9999, 1, 1)),
    ('aten::__upsample', datetime.date(9999, 1, 1)),
    ('aten::__interpolate', datetime.date(9999, 1, 1)),
    ('aten::divmod', datetime.date(9999, 1, 1)),
    ('aten::fabs', datetime.date(9999, 1, 1)),
    ('aten::gamma', datetime.date(9999, 1, 1)),
    ('aten::abs', datetime.date(9999, 1, 1)),
    ('aten::isinf', datetime.date(9999, 1, 1)),
    ('aten::factorial', datetime.date(9999, 1, 1)),
    ('aten::radians', datetime.date(9999, 1, 1)),
    ('aten::degrees', datetime.date(9999, 1, 1)),
    ('aten::acosh', datetime.date(9999, 1, 1)),
    ('aten::atanh', datetime.date(9999, 1, 1)),
    ('aten::asinh', datetime.date(9999, 1, 1)),
    ('aten::floordiv', datetime.date(9999, 1, 1)),
    ('aten::sorted', datetime.date(9999, 1, 1)),
    ('aten::__contains__', datetime.date(9999, 1, 1)),
    ('aten::count', datetime.date(9999, 1, 1)),
    ('aten::remove', datetime.date(9999, 1, 1)),
    ('aten::pop', datetime.date(9999, 1, 1)),
    ('aten::insert', datetime.date(9999, 1, 1)),
    ('aten::clear', datetime.date(9999, 1, 1)),
    ('aten::copy', datetime.date(9999, 1, 1)),
    ('aten::extend', datetime.date(9999, 1, 1)),
    ('aten::reverse', datetime.date(9999, 1, 1)),
    ('aten::append', datetime.date(9999, 1, 1)),
    ('aten::list', datetime.date(9999, 1, 1)),
    ('aten::__getitem__', datetime.date(9999, 1, 1)),
    ('aten::len', datetime.date(9999, 1, 1)),
    ('aten::backward', datetime.date(9999, 1, 1)),
    ('aten::Float', datetime.date(9999, 1, 1)),
    ('aten::Int', datetime.date(9999, 1, 1)),
    ('aten::Bool', datetime.date(9999, 1, 1)),
    ('aten::_ncf_view', datetime.date(9999, 1, 1)),
    ('aten::_ncf_unsqueeze', datetime.date(9999, 1, 1)),
    ('quantized::mul_scalar_relu_out', datetime.date(9999, 1, 1)),
    ('quantized::mul_scalar_out', datetime.date(9999, 1, 1)),
    ('quantized::mul_relu_out', datetime.date(9999, 1, 1)),
    ('quantized::mul_out', datetime.date(9999, 1, 1)),
    ('aten::tan', datetime.date(9999, 1, 1)),
    ('aten::sub', datetime.date(9999, 1, 1)),
    ('aten::sqrt', datetime.date(9999, 1, 1)),
    ('aten::sort', datetime.date(9999, 1, 1)),
    ('aten::slice', datetime.date(9999, 1, 1)),
    ('aten::sinh', datetime.date(9999, 1, 1)),
    ('aten::sin', datetime.date(9999, 1, 1)),
    ('aten::round', datetime.date(9999, 1, 1)),
    ('aten::remainder', datetime.date(9999, 1, 1)),
    ('aten::full_like', datetime.date(9999, 1, 1)),
    ('aten::real', datetime.date(9999, 1, 1)),
    ('aten::randn_like', datetime.date(9999, 1, 1)),
    ('aten::pow', datetime.date(9999, 1, 1)),
    ('aten::floor', datetime.date(9999, 1, 1)),
    ('quantized::cat_relu_out', datetime.date(9999, 1, 1)),
    ('quantized::cat_out', datetime.date(9999, 1, 1)),
    ('aten::neg', datetime.date(9999, 1, 1)),
    ('quantized::add_out', datetime.date(9999, 1, 1)),
    ('aten::expm1', datetime.date(9999, 1, 1)),
    ('aten::ceil', datetime.date(9999, 1, 1)),
    ('aten::add', datetime.date(9999, 1, 1)),
    ('aten::acos', datetime.date(9999, 1, 1)),
    ('aten::cudnn_convolution', datetime.date(9999, 1, 1)),
    ('aten::cudnn_convolution_backward', datetime.date(9999, 1, 1)),
    ('aten::cudnn_convolution_transpose', datetime.date(9999, 1, 1)),
    ('aten::cudnn_convolution_transpose_backward', datetime.date(9999, 1, 1)),
    ('aten::cudnn_convolution_backward_bias', datetime.date(9999, 1, 1)),
    ('aten::cudnn_convolution_transpose_backward_bias', datetime.date(9999, 1, 1)),
    ('aten::atan', datetime.date(9999, 1, 1)),
    ('aten::log10', datetime.date(9999, 1, 1)),
    ('quantized::add_scalar_out', datetime.date(9999, 1, 1)),
    ('quantized::add_scalar_relu_out', datetime.date(9999, 1, 1)),
    ('quantized::add_relu_out', datetime.date(9999, 1, 1)),
    ('aten::exp', datetime.date(9999, 1, 1)),
    ('aten::cosh', datetime.date(9999, 1, 1)),
    ('aten::erf', datetime.date(9999, 1, 1)),
    ('aten::imag', datetime.date(9999, 1, 1)),
    ('aten::empty_like', datetime.date(9999, 1, 1)),
    ('aten::eq', datetime.date(9999, 1, 1)),
    ('aten::index', datetime.date(9999, 1, 1)),
    ('aten::isfinite', datetime.date(9999, 1, 1)),
    ('aten::leaky_relu_backward', datetime.date(9999, 1, 1)),
    ('aten::lgamma', datetime.date(9999, 1, 1)),
    ('aten::log1p', datetime.date(9999, 1, 1)),
    ('aten::asin', datetime.date(9999, 1, 1)),
    ('aten::cos', datetime.date(9999, 1, 1)),
    ('aten::log', datetime.date(9999, 1, 1)),
    ('aten::mul', datetime.date(9999, 1, 1)),
    ('aten::ne', datetime.date(9999, 1, 1)),
    ('aten::rand_like', datetime.date(9999, 1, 1)),
    ('aten::randint_like', datetime.date(9999, 1, 1)),
    ('aten::rrelu_with_noise_backward', datetime.date(9999, 1, 1)),
    ('aten::select', datetime.date(9999, 1, 1)),
    ('aten::tanh', datetime.date(9999, 1, 1)),
    ('aten::add_', datetime.date(9999, 1, 1)),
    ('aten::ones_like', datetime.date(9999, 1, 1)),
    ('aten::to', datetime.date(9999, 1, 1)),
    ('aten::zeros_like', datetime.date(9999, 1, 1)),
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


blacklist = [
    "torch.classes",
    "Any",
    "RRef",
    "aten::setdefault",
    "aten::_set_item",
]


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
            if any(w for w in blacklist if w in line):
                # TODO Fix type __torch__.torch.classes.xxx
                continue

            s = parse_schema(line.strip())
            slist = new_schema_dict.get(s.name, [])
            slist.append(s)
            new_schema_dict[s.name] = slist

    if not check_bc(new_schema_dict):
        sys.exit(1)
