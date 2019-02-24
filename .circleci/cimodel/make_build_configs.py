#!/usr/bin/env python3

"""
This module models the tree of configuration variants
for "smoketest" builds.

Each subclass of ConfigNode represents a layer of the configuration hierarchy.
These tree nodes encapsulate the logic for whether a branch of the hierarchy
should be "pruned".

In addition to generating config.yml content, the tree is also traversed
to produce a visualization of config dimensions.
"""

from collections import OrderedDict

from cimodel.conf_tree import ConfigNode


LINKING_DIMENSIONS = [
    "shared",
    "static",
]

DEPS_INCLUSION_DIMENSIONS = [
    "with-deps",
    "without-deps",
]


def get_processor_arch_name(cuda_version):
    return "cpu" if not cuda_version else "cu" + cuda_version


CUDA_VERSIONS = [
    None,  # cpu build
    "80",
    "90",
    "100",
]

STANDARD_PYTHON_VERSIONS = [
    "2.7",
    "3.5",
    "3.6",
    "3.7",
]


CONFIG_TREE_DATA = OrderedDict(
    linux=(CUDA_VERSIONS, OrderedDict(
        manywheel=[
            "2.7m",
            "2.7mu",
            "3.5m",
            "3.6m",
            "3.7m",
        ],
        conda=STANDARD_PYTHON_VERSIONS,
        libtorch=[
            "2.7m",
        ]
    )),
    macos=([None], OrderedDict(
        wheel=STANDARD_PYTHON_VERSIONS,
        conda=STANDARD_PYTHON_VERSIONS,
        libtorch=[
            "2.7",
        ],
    )),
)


class TopLevelNode(ConfigNode):
    def __init__(self, node_name, config_tree_data, smoke):
        super(TopLevelNode, self).__init__(None, node_name)

        self.config_tree_data = config_tree_data
        self.props["smoke"] = smoke

    def get_children(self):
        return [OSConfigNode(self, x, c, p) for (x, (c, p)) in self.config_tree_data.items()]


class OSConfigNode(ConfigNode):
    def __init__(self, parent, os_name, cuda_versions, py_tree):
        super(OSConfigNode, self).__init__(parent, os_name)

        self.py_tree = py_tree
        self.props["os_name"] = os_name
        self.props["cuda_versions"] = cuda_versions

    def get_children(self):
        return [PackageFormatConfigNode(self, k, v) for k, v in self.py_tree.items()]


class PackageFormatConfigNode(ConfigNode):
    def __init__(self, parent, package_format, python_versions):
        super(PackageFormatConfigNode, self).__init__(parent, package_format)

        self.props["python_versions"] = python_versions
        self.props["package_format"] = package_format

    def get_children(self):
        return [ArchConfigNode(self, v) for v in self.find_prop("cuda_versions")]


class ArchConfigNode(ConfigNode):
    def __init__(self, parent, cu):
        super(ArchConfigNode, self).__init__(parent, get_processor_arch_name(cu))

        self.props["cu"] = cu

    def get_children(self):
        return [PyVersionConfigNode(self, v) for v in self.find_prop("python_versions")]


class PyVersionConfigNode(ConfigNode):
    def __init__(self, parent, pyver):
        super(PyVersionConfigNode, self).__init__(parent, pyver)

        self.props["pyver"] = pyver

    def get_children(self):

        smoke = self.find_prop("smoke")
        package_format = self.find_prop("package_format")
        os_name = self.find_prop("os_name")

        has_libtorch_variants = smoke and package_format == "libtorch" and os_name == "linux"
        linking_variants = LINKING_DIMENSIONS if has_libtorch_variants else []

        return [LinkingVariantConfigNode(self, v) for v in linking_variants]


class LinkingVariantConfigNode(ConfigNode):
    def __init__(self, parent, linking_variant):
        super(LinkingVariantConfigNode, self).__init__(parent, linking_variant)

    def get_children(self):
        return [DependencyInclusionConfigNode(self, v) for v in DEPS_INCLUSION_DIMENSIONS]


class DependencyInclusionConfigNode(ConfigNode):
    def __init__(self, parent, deps_variant):
        super(DependencyInclusionConfigNode, self).__init__(parent, deps_variant)

        self.props["libtorch_variant"] = "-".join([self.parent.get_label(), self.get_label()])
