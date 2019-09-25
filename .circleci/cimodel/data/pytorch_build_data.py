#!/usr/bin/env python3

from cimodel.lib.conf_tree import ConfigNode, X, XImportant


CONFIG_TREE_DATA = [
    ("xenial", [
        (None, [
            XImportant("2.7.9"),
            X("2.7"),
            X("3.5"),
            X("nightly"),
        ]),
        ("gcc", [
            ("5.4", [  # All this subtree rebases to master and then build
                XImportant("3.6"),
                ("3.6", [
                    ("namedtensor", [XImportant(True)]),
                ]),
            ]),
            ("7", [X("3.6")]),
        ]),
        ("clang", [
            ("5", [
                XImportant("3.6"),  # This is actually the ASAN build
                ("3.6", [
                    ("namedtensor", [XImportant(True)]),  # ASAN
                ]),
            ]),
            ("7", [
                ("3.6", [
                    ("xla", [XImportant(True)]),
                ]),
            ]),
        ]),
        ("cuda", [
            ("9", [
                # Note there are magic strings here
                # https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/build.sh#L21
                # and
                # https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/build.sh#L143
                # and
                # https://github.com/pytorch/pytorch/blob/master/.jenkins/pytorch/build.sh#L153
                # (from https://github.com/pytorch/pytorch/pull/17323#discussion_r259453144)
                X("2.7"),
                XImportant("3.6"),
                ("2.7", [
                    ("namedtensor", [XImportant(True)]),
                ]),
            ]),
            ("9.2", [X("3.6")]),
            ("10", [X("3.6")]),
            ("10.1", [X("3.6")]),
        ]),
        ("android", [
            ("r19c", [
                ("3.6", [
                    ("android_abi", [XImportant("x86_32")]),
                    ("android_abi", [X("x86_64")]),
                    ("android_abi", [X("arm-v7a")]),
                    ("android_abi", [X("arm-v8a")]),
                ])
            ]),
        ]),
    ]),
]


def get_major_pyver(dotted_version):
    parts = dotted_version.split(".")
    return "py" + parts[0]


class TreeConfigNode(ConfigNode):
    def __init__(self, parent, node_name, subtree):
        super(TreeConfigNode, self).__init__(parent, self.modify_label(node_name))
        self.subtree = subtree
        self.init2(node_name)

    def modify_label(self, label):
        return label

    def init2(self, node_name):
        pass

    def get_children(self):
        return [self.child_constructor()(self, k, v) for (k, v) in self.subtree]


class TopLevelNode(TreeConfigNode):
    def __init__(self, node_name, subtree):
        super(TopLevelNode, self).__init__(None, node_name, subtree)

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return DistroConfigNode


class DistroConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["distro_name"] = node_name

    def child_constructor(self):
        distro = self.find_prop("distro_name")

        next_nodes = {
            "xenial": XenialCompilerConfigNode,
        }
        return next_nodes[distro]


class PyVerConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["pyver"] = node_name
        self.props["abbreviated_pyver"] = get_major_pyver(node_name)

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return ExperimentalFeatureConfigNode


class ExperimentalFeatureConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["experimental_feature"] = node_name

    def child_constructor(self):
        experimental_feature = self.find_prop("experimental_feature")

        next_nodes = {
            "xla": XlaConfigNode,
            "namedtensor": NamedTensorConfigNode,
            "important": ImportantConfigNode,
            "android_abi": AndroidAbiConfigNode,
        }
        return next_nodes[experimental_feature]


class XlaConfigNode(TreeConfigNode):
    def modify_label(self, label):
        return "XLA=" + str(label)

    def init2(self, node_name):
        self.props["is_xla"] = node_name

    def child_constructor(self):
        return ImportantConfigNode


class NamedTensorConfigNode(TreeConfigNode):
    def modify_label(self, label):
        return "NAMEDTENSOR=" + str(label)

    def init2(self, node_name):
        self.props["is_namedtensor"] = node_name

    def child_constructor(self):
        return ImportantConfigNode

class AndroidAbiConfigNode(TreeConfigNode):

    def init2(self, node_name):
        self.props["android_abi"] = node_name

    def child_constructor(self):
        return ImportantConfigNode

class ImportantConfigNode(TreeConfigNode):
    def modify_label(self, label):
        return "IMPORTANT=" + str(label)

    def init2(self, node_name):
        self.props["is_important"] = node_name

    def get_children(self):
        return []


class XenialCompilerConfigNode(TreeConfigNode):

    def modify_label(self, label):
        return label or "<unspecified>"

    def init2(self, node_name):
        self.props["compiler_name"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):

        return XenialCompilerVersionConfigNode if self.props["compiler_name"] else PyVerConfigNode


class XenialCompilerVersionConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["compiler_version"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return PyVerConfigNode
