from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import json
from collections import defaultdict
from caffe2.python import utils

try:
    import pydot
except ImportError:
    print(
        'Cannot import pydot, which is required for drawing a network. This '
        'can usually be installed in python with "pip install pydot". Also, '
        'pydot requires graphviz to convert dot files to pdf: in ubuntu, this '
        'can usually be installed with "sudo apt-get install graphviz".'
    )
    print(
        'net_drawer will not run correctly. Please install the correct '
        'dependencies.'
    )
    pydot = None

from caffe2.proto import caffe2_pb2

OP_STYLE = {
    'shape': 'box',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}
BLOB_STYLE = {'shape': 'octagon'}


def _rectify_operator_and_name(operators_or_net, name):
    """Gets the operators and name for the pydot graph."""
    if isinstance(operators_or_net, caffe2_pb2.NetDef):
        operators = operators_or_net.op
        if name is None:
            name = operators_or_net.name
    elif hasattr(operators_or_net, 'Proto'):
        net = operators_or_net.Proto()
        if not isinstance(net, caffe2_pb2.NetDef):
            raise RuntimeError(
                "Expecting NetDef, but got {}".format(type(net)))
        operators = net.op
        if name is None:
            name = net.name
    else:
        operators = operators_or_net
        if name is None:
            name = "unnamed"
    return operators, name


def _escape_label(name):
    # json.dumps is poor man's escaping
    return json.dumps(name)


def GetPydotGraph(operators_or_net, name=None, rankdir='LR'):
    operators, name = _rectify_operator_and_name(operators_or_net, name)
    graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes = {}
    pydot_node_counts = defaultdict(int)
    for op_id, op in enumerate(operators):
        if op.name:
            op_node = pydot.Node(
                '%s/%s (op#%d)' % (op.name, op.type, op_id), **OP_STYLE
            )
        else:
            op_node = pydot.Node('%s (op#%d)' % (op.type, op_id), **OP_STYLE)
        graph.add_node(op_node)
        # print 'Op: %s' % op.name
        # print 'inputs: %s' % str(op.input)
        # print 'outputs: %s' % str(op.output)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(
                    input_name + str(pydot_node_counts[input_name]),
                    label=_escape_label(input_name),
                    **BLOB_STYLE
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            graph.add_node(input_node)
            graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                # we are overwriting an existing blob. need to updat the count.
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(
                output_name + str(pydot_node_counts[output_name]),
                label=_escape_label(output_name),
                **BLOB_STYLE
            )
            pydot_nodes[output_name] = output_node
            graph.add_node(output_node)
            graph.add_edge(pydot.Edge(op_node, output_node))
    return graph


def GetPydotGraphMinimal(
    operators_or_net,
    name,
    rankdir='LR',
    minimal_dependency=False
):
    """Different from GetPydotGraph, hide all blob nodes and only show op nodes.

    If minimal_dependency is set as well, for each op, we will only draw the
    edges to the minimal necessary ancestors. For example, if op c depends on
    op a and b, and op b depends on a, then only the edge b->c will be drawn
    because a->c will be implied.
    """
    operators, name = _rectify_operator_and_name(operators_or_net, name)
    graph = pydot.Dot(name, rankdir=rankdir)
    # blob_parents maps each blob name to its generating op.
    blob_parents = {}
    # op_ancestry records the ancestors of each op.
    op_ancestry = defaultdict(set)
    for op_id, op in enumerate(operators):
        if op.name:
            op_node = pydot.Node(
                '%s/%s (op#%d)' % (op.name, op.type, op_id), **OP_STYLE
            )
        else:
            op_node = pydot.Node('%s (op#%d)' % (op.type, op_id), **OP_STYLE)
        graph.add_node(op_node)
        # Get parents, and set up op ancestry.
        parents = [
            blob_parents[input_name] for input_name in op.input
            if input_name in blob_parents
        ]
        op_ancestry[op_node].update(parents)
        for node in parents:
            op_ancestry[op_node].update(op_ancestry[node])
        if minimal_dependency:
            # only add nodes that do not have transitive ancestry
            for node in parents:
                if all(
                    [node not in op_ancestry[other_node]
                     for other_node in parents]
                ):
                    graph.add_edge(pydot.Edge(node, op_node))
        else:
            # Add all parents to the graph.
            for node in parents:
                graph.add_edge(pydot.Edge(node, op_node))
        # Update blob_parents to reflect that this op created the blobs.
        for output_name in op.output:
            blob_parents[output_name] = op_node
    return graph


def GetOperatorMapForPlan(plan_def):
    operator_map = {}
    for net_id, net in enumerate(plan_def.network):
        if net.HasField('name'):
            operator_map[plan_def.name + "_" + net.name] = net.op
        else:
            operator_map[plan_def.name + "_network_%d" % net_id] = net.op
    return operator_map


def main():
    parser = argparse.ArgumentParser(description="Caffe2 net drawer.")
    parser.add_argument(
        "--input",
        type=str,
        help="The input protobuf file."
    )
    parser.add_argument(
        "--output_prefix",
        type=str, default="",
        help="The prefix to be added to the output filename."
    )
    parser.add_argument(
        "--minimal", action="store_true",
        help="If set, produce a minimal visualization."
    )
    parser.add_argument(
        "--minimal_dependency", action="store_true",
        help="If set, only draw minimal dependency."
    )
    parser.add_argument(
        "--rankdir", type=str, default="LR",
        help="The rank direction of the pydot graph."
    )
    args = parser.parse_args()
    with open(args.input, 'r') as fid:
        content = fid.read()
        graphs = utils.GetContentFromProtoString(
            content, {
                caffe2_pb2.PlanDef: lambda x: GetOperatorMapForPlan(x),
                caffe2_pb2.NetDef: lambda x: {x.name: x.op},
            }
        )
    for key, operators in graphs.items():
        if args.minimal:
            graph = GetPydotGraphMinimal(
                operators, key,
                rankdir=args.rankdir,
                minimal_dependency=args.minimal_dependency)
        else:
            graph = GetPydotGraph(
                operators, key,
                rankdir=args.rankdir)
        filename = args.output_prefix + graph.get_name() + '.dot'
        graph.write(filename, format='raw')
        pdf_filename = filename[:-3] + 'pdf'
        try:
            graph.write_pdf(pdf_filename)
        except Exception:
            print(
                'Error when writing out the pdf file. Pydot requires graphviz '
                'to convert dot files to pdf, and you may not have installed '
                'graphviz. On ubuntu this can usually be installed with "sudo '
                'apt-get install graphviz". We have generated the .dot file '
                'but will not be able to generate pdf file for now.'
            )

if __name__ == '__main__':
    main()
