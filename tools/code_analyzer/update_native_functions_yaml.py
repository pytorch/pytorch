"""
This script updates native_functions.yaml using information produced by the static
code analyzer. We could use it to perform large refactor or modifcation to the yaml
file programmatically.

Install ruamel.yaml which supports round trip dump and preserves comments in yaml:
  pip install ruamel.yaml

Run the script to produce new native_functions.yaml to stdout:
  python tools/code_analyzer/update_native_functions_yaml.py
"""

import sys
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

import tools.codegen.gen as gen_v2
import tools.codegen.api.cpp as cpp

NATIVE_FUNCTIONS = 'aten/src/ATen/native/native_functions.yaml'

def load_classify_ops_output(fname):
    import csv
    return csv.DictReader(
        open(fname),
        delimiter='\t',
        skipinitialspace=True,
        quoting=csv.QUOTE_NONE)


def dispatch_stub_ops_filter(ops_table):
    res = {}
    for op in ops_table:
        if op['DispatchStub'] and not op['has_dispatch']:
            res[op['name']] = 'CPU, CUDA'
    return res


def append_before_newline(target_map, new_key, new_value):
    assert new_key not in target_map
    assert isinstance(new_value, CommentedMap)

    target_map[new_key] = new_value

    # Move trailing newline associated with the prev key to the new key.
    trailing_newline = target_map.ca.items.pop(list(target_map.keys())[-2])
    new_value.ca.items[list(new_value.keys())[-1]] = trailing_newline


def update_native_functions(dispatch_updates):
    parsed_nfs = gen_v2.parse_native_yaml(NATIVE_FUNCTIONS)
    parsed_nfs_dict = {str(nf.func) : nf for nf in parsed_nfs}

    yaml = ruamel.yaml.YAML()
    yaml.width = 1024  # Don't wrap line
    yaml.boolean_representation = ['False', 'True']  # Uppercase True/False

    raw_nfs = yaml.load(open(NATIVE_FUNCTIONS, 'r').read())
    for fun in raw_nfs:
        signature = fun['func']
        opname = 'aten::' + signature.split('(')[0]
        if opname not in dispatch_updates:
            continue
        dispatch_keys = dispatch_updates[opname]
        dispatch_dict = CommentedMap()
        dispatch_dict[dispatch_keys] = cpp.name(parsed_nfs_dict[signature].func)
        append_before_newline(fun, 'dispatch', dispatch_dict)

    yaml.dump(raw_nfs, sys.stdout)

if __name__ == '__main__':
    ops_table = load_classify_ops_output('/tmp/table')
    dispatch_updates = dispatch_stub_ops_filter(ops_table)
    update_native_functions(dispatch_updates)
