"""
This util is invoked from cmake to produce the op registration whitelist param
for `ATen/gen.py` for custom mobile build.
For custom build with dynamic dispatch, it takes the op dependency graph of ATen
and the list of root ops, and outputs all transitive dependencies of the root
ops as the whitelist.
For custom build with static dispatch, the op dependency graph will be omitted,
and it will directly output root ops as the whitelist.
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


def gen_transitive_closure(dep_graph, root_ops):
    result = set(root_ops)
    queue = root_ops[:]

    while queue:
        cur = queue.pop()
        for dep in dep_graph.get(cur, []):
            if dep not in result:
                result.add(dep)
                queue.append(dep)

    return ' '.join(sorted(result))


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
    print(gen_transitive_closure(deps, root_ops))
