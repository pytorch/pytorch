"""
This util is invoked from cmake to produce the op registration allowlist param
for `ATen/gen.py` for custom mobile build.
For custom build with dynamic dispatch, it takes the op dependency graph of ATen
and the list of root ops, and outputs all transitive dependencies of the root
ops as the allowlist.
For custom build with static dispatch, the op dependency graph will be omitted,
and it will directly output root ops as the allowlist.
"""

import argparse
import yaml

from collections import defaultdict


def canonical_name(opname):
    # Skip the overload name part as it's not supported by code analyzer yet.
    return opname.split('.', 1)[0]


def load_op_dep_graph(fname):
    with open(fname, 'r') as stream:
        result = defaultdict(set)
        for op in yaml.safe_load(stream):
            op_name = canonical_name(op['name'])
            for dep in op.get('depends', []):
                dep_name = canonical_name(dep['name'])
                result[op_name].add(dep_name)
        return result


def load_root_ops(fname):
    result = []
    with open(fname, 'r') as stream:
        for op in yaml.safe_load(stream):
            result.append(canonical_name(op))
    return result


def gen_transitive_closure(dep_graph, root_ops, train=False):
    result = set(root_ops)
    queue = root_ops[:]

    # The dependency graph might contain a special entry with key = `__BASE__`
    # and value = (set of `base` ops to always include in custom build).
    queue.append('__BASE__')

    # The dependency graph might contain a special entry with key = `__ROOT__`
    # and value = (set of ops reachable from C++ functions). Insert the special
    # `__ROOT__` key to include ops which can be called from C++ code directly,
    # in addition to ops that are called from TorchScript model.
    # '__ROOT__' is only needed for full-jit. Keep it only for training.
    # TODO: when FL is migrated from full-jit to lite trainer, remove '__ROOT__'
    if train:
        queue.append('__ROOT__')

    while queue:
        cur = queue.pop()
        for dep in dep_graph.get(cur, []):
            if dep not in result:
                result.add(dep)
                queue.append(dep)

    return sorted(result)

def gen_transitive_closure_str(dep_graph, root_ops):
    return ' '.join(gen_transitive_closure(dep_graph, root_ops))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Util to produce transitive dependencies for custom build')
    parser.add_argument(
        '--op-dependency',
        help='input yaml file of op dependency graph '
             '- can be omitted for custom build with static dispatch')
    parser.add_argument(
        '--root-ops',
        required=True,
        help='input yaml file of root (directly used) operators')
    args = parser.parse_args()

    deps = load_op_dep_graph(args.op_dependency) if args.op_dependency else {}
    root_ops = load_root_ops(args.root_ops)
    print(gen_transitive_closure_str(deps, root_ops))
