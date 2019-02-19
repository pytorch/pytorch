import colorsys
import sys

from pygraphviz import AGraph


class ConfigNode:
    def __init__(self, parent, node_name):
        self.parent = parent
        self.node_name = node_name
        self.props = {}

    def get_label(self):
        label = self.node_name
        if not label:
            # FIXME this shouldn't be necessary
            label = "<None>"
        return label

    def get_children(self):
        return []

    def get_parents(self):
        return (self.parent.get_parents() + [self.parent.get_label()]) if self.parent else []

    def get_depth(self):
        return len(self.get_parents())

    def get_node_key(self):
        return "%".join(self.get_parents() + [self.get_label()])

    def find_prop(self, propname, searched=None):
        """
        Checks if its own dictionary has
        the property, otherwise asks parent node.
        """

        if searched is None:
            searched = []

        searched.append(self.node_name)

        if propname in self.props:
            return self.props[propname]
        elif self.parent:
            return self.parent.find_prop(propname, searched)
        else:
            # raise Exception('Property "%s" does not exist anywhere in the tree! Searched: %s' % (propname, searched))
            return None


def rgb2hex(rgb_tuple):
    def toHex(f):
        return "%02x" % int(f * 255)

    return "#" + "".join(map(toHex, list(rgb_tuple)))


def dfs(toplevel_config_node):

    dot = AGraph()

    config_list = []

    MAX_DEPTH = 7  # FIXME traverse once beforehand to find max depth

    def dfs_recurse(node):

        this_node_key = node.get_node_key()

        depth = node.get_depth()

        rgb_tuple = colorsys.hsv_to_rgb(depth / float(MAX_DEPTH), 0.5, 1)
        hex_color = rgb2hex(rgb_tuple)

        dot.add_node(
            this_node_key,
            label=node.get_label(),
            style="filled",
            color="black",
            # fillcolor=hex_color + ":orange",
            fillcolor=hex_color,
        )

        node_children = node.get_children()

        if node_children:
            for child in node_children:

                child_node_key = child.get_node_key()
                dot.add_edge((this_node_key, child_node_key))

                dfs_recurse(child)
        else:
            config_list.append(node)

    dfs_recurse(toplevel_config_node)
    return config_list, dot
