## @package net_drawer
# Module caffe2.python.net_drawer




import argparse
import json
import logging
from collections import defaultdict
from caffe2.python import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import pydot
except ImportError:
    logger.info(
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


def GetOpNodeProducer(append_output, **kwargs):
    def ReallyGetOpNode(op, op_id):
        if op.name:
            node_name = '%s/%s (op#%d)' % (op.name, op.type, op_id)
        else:
            node_name = '%s (op#%d)' % (op.type, op_id)
        if append_output:
            for output_name in op.output:
                node_name += '\n' + output_name
        return pydot.Node(node_name, **kwargs)
    return ReallyGetOpNode


def GetBlobNodeProducer(**kwargs):
    def ReallyGetBlobNode(node_name, label):
        return pydot.Node(node_name, label=label, **kwargs)
    return ReallyGetBlobNode

def GetPydotGraph(
    operators_or_net,
    name=None,
    rankdir='LR',
    op_node_producer=None,
    blob_node_producer=None
):
    if op_node_producer is None:
        op_node_producer = GetOpNodeProducer(False, **OP_STYLE)
    if blob_node_producer is None:
        blob_node_producer = GetBlobNodeProducer(**BLOB_STYLE)
    operators, name = _rectify_operator_and_name(operators_or_net, name)
    graph = pydot.Dot(name, rankdir=rankdir)
    pydot_nodes = {}
    pydot_node_counts = defaultdict(int)
    for op_id, op in enumerate(operators):
        op_node = op_node_producer(op, op_id)
        graph.add_node(op_node)
        # print 'Op: %s' % op.name
        # print 'inputs: %s' % str(op.input)
        # print 'outputs: %s' % str(op.output)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = blob_node_producer(
                    _escape_label(
                        input_name + str(pydot_node_counts[input_name])),
                    label=_escape_label(input_name),
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            graph.add_node(input_node)
            graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                # we are overwriting an existing blob. need to update the count.
                pydot_node_counts[output_name] += 1
            output_node = blob_node_producer(
                _escape_label(
                    output_name + str(pydot_node_counts[output_name])),
                label=_escape_label(output_name),
            )
            pydot_nodes[output_name] = output_node
            graph.add_node(output_node)
            graph.add_edge(pydot.Edge(op_node, output_node))
    return graph


def GetPydotGraphMinimal(
    operators_or_net,
    name=None,
    rankdir='LR',
    minimal_dependency=False,
    op_node_producer=None,
):
    """Different from GetPydotGraph, hide all blob nodes and only show op nodes.

    If minimal_dependency is set as well, for each op, we will only draw the
    edges to the minimal necessary ancestors. For example, if op c depends on
    op a and b, and op b depends on a, then only the edge b->c will be drawn
    because a->c will be implied.
    """
    if op_node_producer is None:
        op_node_producer = GetOpNodeProducer(False, **OP_STYLE)
    operators, name = _rectify_operator_and_name(operators_or_net, name)
    graph = pydot.Dot(name, rankdir=rankdir)
    # blob_parents maps each blob name to its generating op.
    blob_parents = {}
    # op_ancestry records the ancestors of each op.
    op_ancestry = defaultdict(set)
    for op_id, op in enumerate(operators):
        op_node = op_node_producer(op, op_id)
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


def _draw_nets(nets, g):
    nodes = []
    for i, net in enumerate(nets):
        nodes.append(pydot.Node(_escape_label(net)))
        g.add_node(nodes[-1])
        if i > 0:
            g.add_edge(pydot.Edge(nodes[-2], nodes[-1]))
    return nodes


def _draw_steps(steps, g, skip_step_edges=False):  # noqa
    kMaxParallelSteps = 3

    def get_label():
        label = [step.name + '\n']
        if step.report_net:
            label.append('Reporter: {}'.format(step.report_net))
        if step.should_stop_blob:
            label.append('Stopper: {}'.format(step.should_stop_blob))
        if step.concurrent_substeps:
            label.append('Concurrent')
        if step.only_once:
            label.append('Once')
        return '\n'.join(label)

    def substep_edge(start, end):
        return pydot.Edge(start, end, arrowhead='dot', style='dashed')

    nodes = []
    for i, step in enumerate(steps):
        parallel = step.concurrent_substeps

        nodes.append(pydot.Node(_escape_label(get_label()), **OP_STYLE))
        g.add_node(nodes[-1])

        if i > 0 and not skip_step_edges:
            g.add_edge(pydot.Edge(nodes[-2], nodes[-1]))

        if step.network:
            sub_nodes = _draw_nets(step.network, g)
        elif step.substep:
            if parallel:
                sub_nodes = _draw_steps(
                    step.substep[:kMaxParallelSteps], g, skip_step_edges=True)
            else:
                sub_nodes = _draw_steps(step.substep, g)
        else:
            raise ValueError('invalid step')

        if parallel:
            for sn in sub_nodes:
                g.add_edge(substep_edge(nodes[-1], sn))
            if len(step.substep) > kMaxParallelSteps:
                ellipsis = pydot.Node('{} more steps'.format(
                    len(step.substep) - kMaxParallelSteps), **OP_STYLE)
                g.add_node(ellipsis)
                g.add_edge(substep_edge(nodes[-1], ellipsis))
        else:
            g.add_edge(substep_edge(nodes[-1], sub_nodes[0]))

    return nodes


def GetPlanGraph(plan_def, name=None, rankdir='TB'):
    graph = pydot.Dot(name, rankdir=rankdir)
    _draw_steps(plan_def.execution_step, graph)
    return graph


def GetGraphInJson(operators_or_net, output_filepath):
    operators, _ = _rectify_operator_and_name(operators_or_net, None)
    blob_strid_to_node_id = {}
    node_name_counts = defaultdict(int)
    nodes = []
    edges = []
    for op_id, op in enumerate(operators):
        op_label = op.name + '/' + op.type if op.name else op.type
        op_node_id = len(nodes)
        nodes.append({
            'id': op_node_id,
            'label': op_label,
            'op_id': op_id,
            'type': 'op'
        })
        for input_name in op.input:
            strid = _escape_label(
                input_name + str(node_name_counts[input_name]))
            if strid not in blob_strid_to_node_id:
                input_node = {
                    'id': len(nodes),
                    'label': input_name,
                    'type': 'blob'
                }
                blob_strid_to_node_id[strid] = len(nodes)
                nodes.append(input_node)
            else:
                input_node = nodes[blob_strid_to_node_id[strid]]
            edges.append({
                'source': blob_strid_to_node_id[strid],
                'target': op_node_id
            })
        for output_name in op.output:
            strid = _escape_label(
                output_name + str(node_name_counts[output_name]))
            if strid in blob_strid_to_node_id:
                # we are overwriting an existing blob. need to update the count.
                node_name_counts[output_name] += 1
                strid = _escape_label(
                    output_name + str(node_name_counts[output_name]))

            if strid not in blob_strid_to_node_id:
                output_node = {
                    'id': len(nodes),
                    'label': output_name,
                    'type': 'blob'
                }
                blob_strid_to_node_id[strid] = len(nodes)
                nodes.append(output_node)
            edges.append({
                'source': op_node_id,
                'target': blob_strid_to_node_id[strid]
            })

    with open(output_filepath, 'w') as f:
        json.dump({'nodes': nodes, 'edges': edges}, f)


# A dummy minimal PNG image used by GetGraphPngSafe as a
# placeholder when rendering fail to run.
_DummyPngImage = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00'
    b'\x01\x01\x00\x00\x00\x007n\xf9$\x00\x00\x00\nIDATx\x9cc`\x00\x00'
    b'\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')


def GetGraphPngSafe(func, *args, **kwargs):
    """
    Invokes `func` (e.g. GetPydotGraph) with args. If anything fails - returns
    and empty image instead of throwing Exception
    """
    try:
        graph = func(*args, **kwargs)
        if not isinstance(graph, pydot.Dot):
            raise ValueError("func is expected to return pydot.Dot")
        return graph.create_png()
    except Exception as e:
        logger.error("Failed to draw graph: {}".format(e))
        return _DummyPngImage


def main():
    parser = argparse.ArgumentParser(description="Caffe2 net drawer.")
    parser.add_argument(
        "--input",
        type=str, required=True,
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
        "--append_output", action="store_true",
        help="If set, append the output blobs to the operator names.")
    parser.add_argument(
        "--rankdir", type=str, default="LR",
        help="The rank direction of the pydot graph."
    )
    args = parser.parse_args()
    with open(args.input, 'r') as fid:
        content = fid.read()
        graphs = utils.GetContentFromProtoString(
            content, {
                caffe2_pb2.PlanDef: GetOperatorMapForPlan,
                caffe2_pb2.NetDef: lambda x: {x.name: x.op},
            }
        )
    for key, operators in graphs.items():
        if args.minimal:
            graph = GetPydotGraphMinimal(
                operators,
                name=key,
                rankdir=args.rankdir,
                node_producer=GetOpNodeProducer(args.append_output, **OP_STYLE),
                minimal_dependency=args.minimal_dependency)
        else:
            graph = GetPydotGraph(
                operators,
                name=key,
                rankdir=args.rankdir,
                node_producer=GetOpNodeProducer(args.append_output, **OP_STYLE))
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
