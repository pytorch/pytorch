import unittest

from torchgen.utils import NamespaceHelper


class TestNamespaceHelper(unittest.TestCase):
    def test_create_from_namespaced_tuple(self) -> None:
        helper = NamespaceHelper.from_namespaced_entity("aten::add")
        self.assertEqual(helper.entity_name, "add")
        self.assertEqual(helper.get_cpp_namespace(), "aten")

    def test_default_namespace(self) -> None:
        helper = NamespaceHelper.from_namespaced_entity("add")
        self.assertEqual(helper.entity_name, "add")
        self.assertEqual(helper.get_cpp_namespace(), "")
        self.assertEqual(helper.get_cpp_namespace("default"), "default")

    def test_namespace_levels_more_than_max(self) -> None:
        with self.assertRaises(AssertionError):
            NamespaceHelper(
                namespace_str="custom_1::custom_2", entity_name="", max_level=1
            )
