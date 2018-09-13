"""
Experimental. Tools for visualizing the torch.jit.Graph objects.
"""
import json
import os
import string


template_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "_graph_vis_template.html"
)


def write(graph, filename):
    """
    Write an html file that visualizes a torch.jit.Graph using vis.js
    Arguments:
        graph (torch.jit.Graph): the graph.
        filename (string): the output filename, an html-file.
    """

    nodes = []
    edges = []
    options = {}

    counts = {}
    ir_nodes = {}
    offset = 0
    colors = {
        "prim::Constant": "rgb(221, 212, 137)",
        "prim::Return": "rgb(244, 104, 66)",
    }

    def get_color(node):
        return {
            "background": colors.get(node.kind(), "rgb(140, 184, 255)")
        }

    id = 0
    ir_nodes[graph.return_node()] = id
    nodes.append(
        {
            "id": id,
            "label": graph.return_node().kind(),
            "shape": "box",
            "color": get_color(graph.return_node()),
        }
    )
    id += 1

    ir_nodes[graph.param_node()] = id
    nodes.append(
        {
            "id": id,
            "label": graph.param_node().kind(),
            "color": get_color(graph.param_node()),
        }
    )
    id += 1

    for node in graph.nodes():
        d = {"id": id, "label": node.kind(), "color": get_color(node)}
        ir_nodes[node] = id
        nodes.append(d)
        id += 1
        offset += 30

    for node in graph.nodes():
        for output_value in node.outputs():
            for use in output_value.uses():
                edge = {
                    "from": ir_nodes[node],
                    "to": ir_nodes[use.user],
                    "label": "%{}".format(output_value.uniqueName()),
                    "arrows": "to",
                }
                edges.append(edge)

    for output_value in graph.param_node().outputs():
        for use in output_value.uses():
            edge = {
                "from": ir_nodes[graph.param_node()],
                "to": ir_nodes[use.user],
                "label": "%{}".format(output_value.uniqueName()),
                "arrows": "to",
            }
            edges.append(edge)

    with open(template_file, "r") as content_file:
        _vis_template = string.Template(content_file.read())
        result = _vis_template.substitute(
            raw_graph="{}".format(graph),
            nodes=json.dumps(nodes),
            edges=json.dumps(edges),
            options=json.dumps(options),
            name=filename,
        )
        with open(filename, "w") as f:
            f.write(result)
