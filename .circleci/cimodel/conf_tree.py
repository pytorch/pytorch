#!/usr/bin/env python3


class Ver(object):
    """
    Represents a product with a version number
    """
    def __init__(self, name, version=""):
        self.name = name
        self.version = version

    def __str__(self):
        return self.name + self.version


class ConfigNode(object):
    def __init__(self, parent, node_name):
        self.parent = parent
        self.node_name = node_name
        self.props = {}

    def get_label(self):
        return self.node_name

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


def dfs_recurse(
        node,
        leaf_callback=lambda x: None,
        discovery_callback=lambda x, y, z: None,
        child_callback=lambda x, y: None,
        sibling_index=0,
        sibling_count=1):

    discovery_callback(node, sibling_index, sibling_count)

    node_children = node.get_children()
    if node_children:
        for i, child in enumerate(node_children):
            child_callback(node, child)

            dfs_recurse(
                child,
                leaf_callback,
                discovery_callback,
                child_callback,
                i,
                len(node_children),
            )
    else:
        leaf_callback(node)


def dfs(toplevel_config_node):

    config_list = []

    def leaf_callback(node):
        config_list.append(node)

    dfs_recurse(toplevel_config_node, leaf_callback)

    return config_list
