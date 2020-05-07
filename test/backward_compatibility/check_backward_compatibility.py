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
    ('aten::append*', datetime.date(2020, 4, 15)),
    ('aten::real*', datetime.date(2020, 4, 15)),
    ('aten::imag*', datetime.date(2020, 4, 15)),
    ('aten::quantize_per_tensor', datetime.date(2020, 4, 15)),
    ('aten::index_put', datetime.date(2020, 4, 10)),
    ('aten::index', datetime.date(2020, 4, 10)),
    ('aten::_index_put_impl', datetime.date(2020, 4, 10)),
    ('aten::index_put_', datetime.date(2020, 4, 10)),
    ('aten::quantize_per_tensor', datetime.date(2020, 4, 15)),
    ('aten::requires_grad_', datetime.date(2020, 4, 30)),
    ('quantized::batch_norm', datetime.date(2020, 4, 20)),
    ('aten::sizes', datetime.date(2020, 4, 30)),
    ('aten::strides', datetime.date(2020, 4, 30)),
    ('aten::backward', datetime.date(2020, 4, 30)),
    ('quantized::conv_prepack', datetime.date(2020, 6, 1)),
    ('quantized::conv_unpack', datetime.date(2020, 6, 1)),
    ('quantized::conv', datetime.date(2020, 6, 1)),
    ('quantized::conv2d_prepack', datetime.date(2020, 6, 1)),
    ('quantized::conv2d_unpack', datetime.date(2020, 6, 1)),
    ('quantized::conv2d', datetime.date(2020, 6, 1)),
    ('quantized::conv2d_relu', datetime.date(2020, 6, 1)),
    ('quantized::conv3d_prepack', datetime.date(2020, 6, 1)),
    ('quantized::conv3d_unpack', datetime.date(2020, 6, 1)),
    ('quantized::conv3d', datetime.date(2020, 6, 1)),
    ('quantized::conv3d_relu', datetime.date(2020, 5, 1)),
    ('_quantized::conv2d_relu', datetime.date(2020, 6, 1)),
    ('_quantized::conv2d', datetime.date(2020, 6, 1)),
    ('aten::batch_norm_gather_stats_with_counts', datetime.date(2020, 6, 30)),
    ('aten::quantized_lstm', datetime.date(2020, 6, 1)),
    ('aten::quantized_gru', datetime.date(2020, 6, 1)),
    ('quantized::make_quantized_cell_params', datetime.date(2020, 6, 1)),
    ('quantized::make_quantized_cell_params_fp16', datetime.date(2020, 6, 1)),
    ('quantized::make_quantized_cell_params_dynamic', datetime.date(2020, 6, 1)),
    ('aten::len', datetime.date(2020, 6, 30)),
    ('aten::keys', datetime.date(2020, 6, 30)),
    ('aten::values', datetime.date(2020, 6, 30)),
    ('aten::__getitem__', datetime.date(2020, 6, 30)),
    ('aten::get', datetime.date(2020, 6, 30)),
    ('aten::setdefault', datetime.date(2020, 6, 30)),
    ('aten::Delete', datetime.date(2020, 6, 30)),
    ('aten::pop', datetime.date(2020, 6, 30)),
    ('aten::popitem', datetime.date(2020, 6, 30)),
    ('aten::clear', datetime.date(2020, 6, 30)),
    ('aten::update', datetime.date(2020, 6, 30)),
    ('aten::items', datetime.date(2020, 6, 30)),
    ('aten::copy', datetime.date(2020, 6, 30)),
    ('aten::__contains__', datetime.date(2020, 6, 30)),
    ('aten::_set_item', datetime.date(2020, 6, 30)),
    ('aten::dict', datetime.date(2020, 6, 30)),
    ('aten::tensor', datetime.date(2020, 6, 30)),
    ('aten::as_tensor', datetime.date(2020, 6, 30)),
]


# The nightly will fail to parse newly added syntax to schema declarations
# Add new schemas that will fail the nightly here
dont_parse_list = [
    ('aten::quantized_lstm', datetime.date(2020, 6, 1)),
    ('aten::quantized_gru', datetime.date(2020, 6, 1)),
    ('quantized::make_quantized_cell_params', datetime.date(2020, 6, 1)),
    ('quantized::make_quantized_cell_params_fp16', datetime.date(2020, 6, 1)),
    ('quantized::make_quantized_cell_params_dynamic', datetime.date(2020, 6, 1)),
    ('quantized::conv_prepack', datetime.date(2020, 6, 1)),
    ('quantized::conv_unpack', datetime.date(2020, 6, 1)),
    ('quantized::conv', datetime.date(2020, 6, 1)),
    ('quantized::conv2d_prepack', datetime.date(2020, 6, 1)),
    ('quantized::conv2d_unpack', datetime.date(2020, 6, 1)),
    ('quantized::conv2d', datetime.date(2020, 6, 1)),
    ('quantized::conv2d_relu', datetime.date(2020, 6, 1)),
    ('quantized::conv3d_prepack', datetime.date(2020, 6, 1)),
    ('quantized::conv3d_unpack', datetime.date(2020, 6, 1)),
    ('quantized::conv3d', datetime.date(2020, 6, 1)),
    ('quantized::conv3d_relu', datetime.date(2020, 6, 1)),
]


def white_listed(schema, white_list):
    for item in white_list:
        if item[1] < datetime.date.today():
            continue
        regexp = re.compile(item[0])
        if regexp.search(schema.name):
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


def check_bc(new_schema_dict):
    existing_schemas = torch._C._jit_get_all_schemas()
    is_bc = True
    broken_ops = []
    for existing_schema in existing_schemas:
        if white_listed(existing_schema, white_list):
            print("Black list, skipping schema: ", str(existing_schema))
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
            if dont_parse(line.strip()):
                print("Not parsing schema line: ", line.strip())
                continue
            s = parse_schema(line.strip())
            slist = new_schema_dict.get(s.name, [])
            slist.append(s)
            new_schema_dict[s.name] = slist

    if not check_bc(new_schema_dict):
        sys.exit(1)
