from cimodel.lib.conf_tree import ConfigNode, XImportant
from cimodel.lib.conf_tree import Ver


CONFIG_TREE_DATA = [
    (Ver("ubuntu", "16.04"), [
        ([Ver("gcc", "5")], [XImportant("onnx_py2")]),
        ([Ver("clang", "7")], [XImportant("onnx_py3.6")]),
    ]),
]


class TreeConfigNode(ConfigNode):
    def __init__(self, parent, node_name, subtree):
        super(TreeConfigNode, self).__init__(parent, self.modify_label(node_name))
        self.subtree = subtree
        self.init2(node_name)

    # noinspection PyMethodMayBeStatic
    def modify_label(self, label):
        return str(label)

    def init2(self, node_name):
        pass

    def get_children(self):
        return [self.child_constructor()(self, k, v) for (k, v) in self.subtree]

    def is_build_only(self):
        if str(self.find_prop("language_version")) == "onnx_py3.6":
            return False
        return set(str(c) for c in self.find_prop("compiler_version")).intersection({
            "clang3.8",
            "clang3.9",
            "clang7",
            "android",
        }) or self.find_prop("distro_version").name == "macos"


class TopLevelNode(TreeConfigNode):
    def __init__(self, node_name, subtree):
        super(TopLevelNode, self).__init__(None, node_name, subtree)

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return DistroConfigNode


class DistroConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["distro_version"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return CompilerConfigNode


class CompilerConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["compiler_version"] = node_name

    # noinspection PyMethodMayBeStatic
    def child_constructor(self):
        return LanguageConfigNode


class LanguageConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["language_version"] = node_name
        self.props["build_only"] = self.is_build_only()

    def child_constructor(self):
        return ImportantConfigNode


class ImportantConfigNode(TreeConfigNode):
    def init2(self, node_name):
        self.props["important"] = True

    def get_children(self):
        return []
