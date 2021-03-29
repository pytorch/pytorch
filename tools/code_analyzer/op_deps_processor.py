"""
This util is used to parse op_deps_pass output (in yaml) and convert it into
other formats for downstream use cases. It is not used by OSS cmake build.

To run this file by hand from the root of the PyTorch repository, run:

python -m tools.code_analyzer.op_deps_processor \
  --op-dependency build_code_analyzer/work/torch_result.yaml \
  --output pt_deps.bzl
"""

import argparse
import yaml

from tools.codegen.code_template import CodeTemplate

BAZEL_OUTPUT = CodeTemplate("""\
TORCH_DEPS = {
${ops}
}
""")

BAZEL_OP = CodeTemplate("""\
    "${op_name}": [
${op_deps}
    ],
""")

BAZEL_OP_DEP = CodeTemplate("""\
        "${dep_name}",
""")

DOT_OUTPUT = CodeTemplate("""\
digraph {
    layout="circo";
${ops}
}
""")

DOT_OP = CodeTemplate("""\
${op_deps}
""")

DOT_OP_DEP = CodeTemplate("""\
    "${op_name}" -> "${dep_name}";
""")


def load_op_deps(fname):
    with open(fname, 'r') as stream:
        return yaml.safe_load(stream)


def process_base_ops(graph, base_ops):
    # remove base ops from all `depends` lists to compress the output graph
    for op in graph:
        op['depends'] = [
            dep for dep in op.get('depends', []) if dep['name'] not in base_ops
        ]

    # add base ops section at the beginning
    graph.insert(0, {
        'name': '__BASE__',
        'depends': [{'name': name} for name in base_ops]})


def convert(fname, graph, output_template, op_template, op_dep_template):
    ops = []
    for op in graph:
        op_name = op['name']
        op_deps = []

        for dep in op.get('depends', []):
            dep_name = dep['name']
            if dep_name == op_name:
                # skip itself reference
                continue
            op_deps.append(
                op_dep_template.substitute(
                    op_name=op_name,
                    dep_name=dep_name))

        if not op_deps:
            # skip ops without any fanout
            continue

        ops.append(
            op_template.substitute(
                op_name=op_name,
                op_deps=op_deps))

    with open(fname, 'w') as out:
        out.write(output_template.substitute(ops=ops))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Util to parse & convert op_deps_pass output')
    parser.add_argument(
        '--op_dependency',
        required=True,
        help='input yaml file of op dependency graph produced by op_deps_pass')
    parser.add_argument(
        '--format',
        default='bazel',
        help='output file format [bazel, dot]')
    parser.add_argument(
        '--base_ops',
        nargs='*',
        help='optional list of `base` ops that should always be kept in '
             'custom build, to make the output stable from trivial changes; '
             'each item is `namespace`::`operator name` without overload; '
             'e.g.: aten::empty aten::size ...')
    parser.add_argument(
        '--output',
        required=True,
        help='output file')
    args = parser.parse_args()

    deps = load_op_deps(args.op_dependency)

    if args.base_ops:
        process_base_ops(deps, args.base_ops)

    if args.format == 'bazel':
        convert(args.output, deps, BAZEL_OUTPUT, BAZEL_OP, BAZEL_OP_DEP)
    elif args.format == 'dot':
        convert(args.output, deps, DOT_OUTPUT, DOT_OP, DOT_OP_DEP)
    else:
        raise Exception("Unknown output format: " + args.format)
