"""
This module encapsulates dependencies on pygraphviz
"""

import colorsys

import cimodel.lib.conf_tree as conf_tree


def rgb2hex(rgb_tuple):
    def to_hex(f):
        return "%02x" % int(f * 255)

    return "#" + "".join(map(to_hex, list(rgb_tuple)))


def handle_missing_graphviz(f):
    """
    If the user has not installed pygraphviz, this causes
    calls to the draw() method of the returned object to do nothing.
    """
    try:
        import pygraphviz  # noqa: F401
        return f

    except ModuleNotFoundError:

        class FakeGraph:
            def draw(self, *args, **kwargs):
                pass

        return lambda _: FakeGraph()


@handle_missing_graphviz
def generate_graph(toplevel_config_node):
    """
    Traverses the graph once first just to find the max depth
    """

    config_list = conf_tree.dfs(toplevel_config_node)

    max_depth = 0
    for config in config_list:
        max_depth = max(max_depth, config.get_depth())

    # color the nodes using the max depth

    from pygraphviz import AGraph
    dot = AGraph()

    def node_discovery_callback(node, sibling_index, sibling_count):
        depth = node.get_depth()

        sat_min, sat_max = 0.1, 0.6
        sat_range = sat_max - sat_min

        saturation_fraction = sibling_index / float(sibling_count - 1) if sibling_count > 1 else 1
        saturation = sat_min + sat_range * saturation_fraction

        # TODO Use a hash of the node label to determine the color
        hue = depth / float(max_depth + 1)

        rgb_tuple = colorsys.hsv_to_rgb(hue, saturation, 1)

        this_node_key = node.get_node_key()

        dot.add_node(
            this_node_key,
            label=node.get_label(),
            style="filled",
            # fillcolor=hex_color + ":orange",
            fillcolor=rgb2hex(rgb_tuple),
            penwidth=3,
            color=rgb2hex(colorsys.hsv_to_rgb(hue, saturation, 0.9))
        )

    def child_callback(node, child):
        this_node_key = node.get_node_key()
        child_node_key = child.get_node_key()
        dot.add_edge((this_node_key, child_node_key))

    conf_tree.dfs_recurse(toplevel_config_node, lambda x: None, node_discovery_callback, child_callback)
    return dot
