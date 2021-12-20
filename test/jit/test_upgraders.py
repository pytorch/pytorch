# Owner(s): ["oncall: jit"]

import io
import os
import sys
import torch
from torch.testing import FileCheck
from typing import Union

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestUpgraders(JitTestCase):
    def test_populated_upgrader_graph(self):
        @torch.jit.script
        def f():
            return 0

        buffer = io.BytesIO()
        torch.jit.save(f, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
        upgraders_size = torch._C._get_upgraders_map_size()
        upgraders_dump = torch._C._dump_upgraders_map()
        # make sure we only populate the upgrader map only once
        # so we load it again and make sure the upgrader map has
        # same content
        buffer.seek(0)
        torch.jit.load(buffer)
        upgraders_size_second_time = torch._C._get_upgraders_map_size()
        upgraders_dump_second_time = torch._C._dump_upgraders_map()
        self.assertTrue(upgraders_size == upgraders_size_second_time)
        self.assertTrue(upgraders_dump == upgraders_dump_second_time)

    def test_add_value_to_version_map(self):
        map_before_test = torch._C._get_operator_version_map()

        upgrader_bumped_version = 3
        upgrader_name = "_test_serialization_subcmul_0_2"
        upgrader_schema = "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"
        dummy_entry = torch._C._UpgraderEntry(upgrader_bumped_version, upgrader_name, upgrader_schema)

        torch._C._test_only_add_entry_to_op_version_map("aten::_test_serialization_subcmul", dummy_entry)
        map_after_test = torch._C._get_operator_version_map()
        self.assertTrue("aten::_test_serialization_subcmul" in map_after_test)
        self.assertTrue(len(map_after_test) - len(map_before_test) == 1)
        torch._C._test_only_remove_entry_to_op_version_map("aten::_test_serialization_subcmul")
        map_after_remove_test = torch._C._get_operator_version_map()
        self.assertTrue("aten::_test_serialization_subcmul" not in map_after_remove_test)
        self.assertEqual(len(map_after_remove_test), len(map_before_test))

    def test_populated_test_upgrader_graph(self):
        @torch.jit.script
        def f():
            return 0

        buffer = io.BytesIO()
        torch.jit.save(f, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)

        # upgrader map should have populated now
        upgraders_size = torch._C._get_upgraders_map_size()

        test_map = {"a": "b", "c": "d"}
        torch._C._test_only_populate_upgraders(test_map)
        upgraders_size_after_test = torch._C._get_upgraders_map_size()
        self.assertEqual(upgraders_size_after_test - upgraders_size, 2)
        upgraders_dump = torch._C._dump_upgraders_map()
        self.assertTrue("a" in upgraders_dump)
        self.assertTrue("c" in upgraders_dump)

        torch._C._test_only_remove_upgraders(test_map)
        upgraders_size_after_remove_test = torch._C._get_upgraders_map_size()
        self.assertTrue(upgraders_size_after_remove_test == upgraders_size)
        upgraders_dump_after_remove_test = torch._C._dump_upgraders_map()
        self.assertTrue("a" not in upgraders_dump_after_remove_test)
        self.assertTrue("c" not in upgraders_dump_after_remove_test)

    def test_aten_div_tensor_at_3(self):
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_div_tensor_v3.pt"
        loaded_model = torch.jit.load(model_path)
        # there are 3 aten::div in this model
        # And the upgrader for aten::div uses two
        # div's because of if/else branch
        FileCheck().check("prim::If").run(loaded_model.graph)
        FileCheck().check_count("aten::div", 6).run(loaded_model.graph)

        buffer = io.BytesIO()
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        loaded_model_twice = torch.jit.load(buffer)
        # we check by its' code because graph variable names
        # can be different every time
        self.assertEqual(loaded_model.code, loaded_model_twice.code)

    def test_aten_test_serialization(self):
        model_path = pytorch_test_dir + "/jit/fixtures/_test_serialization_subcmul_v2.pt"

        # add test version entry to the version map
        upgrader_bumped_version = 3
        upgrader_name = "_test_serialization_subcmul_0_2"
        upgrader_schema = "aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=2) -> Tensor"
        dummy_entry = torch._C._UpgraderEntry(upgrader_bumped_version, upgrader_name, upgrader_schema)

        torch._C._test_only_add_entry_to_op_version_map("aten::_test_serialization_subcmul", dummy_entry)

        # add test upgrader in the upgraders map
        @torch.jit.script
        def _test_serialization_subcmul_0_2(self: torch.Tensor, other: torch.Tensor, alpha: Union[int, float] = 2) -> torch.Tensor:
            return other - (self * alpha)

        torch._C._test_only_populate_upgraders({"_test_serialization_subcmul_0_2": str(_test_serialization_subcmul_0_2.graph)})

        # test if the server is able to find the test upgraders and apply to IR
        loaded_model = torch.jit.load(model_path)
        FileCheck().check_count("aten::mul", 2).run(loaded_model.graph)
        FileCheck().check_count("aten::sub", 2).run(loaded_model.graph)

        buffer = io.BytesIO()
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        loaded_model_twice = torch.jit.load(buffer)
        # we check by its' code because graph variable names
        # can be different every time
        self.assertEqual(loaded_model.code, loaded_model_twice.code)
        torch._C._test_only_remove_entry_to_op_version_map("aten::_test_serialization_subcmul")
        torch._C._test_only_remove_upgraders({"_test_serialization_subcmul_0_2": str(_test_serialization_subcmul_0_2.graph)})

    def test_aten_div_scalar_at_3(self):
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_div_scalar_float_v3.pt"
        loaded_model = torch.jit.load(model_path)
        FileCheck().check("prim::If").run(loaded_model.graph)
        FileCheck().check_count("aten::div", 2).run(loaded_model.graph)

        buffer = io.BytesIO()
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        loaded_model_twice = torch.jit.load(buffer)

        self.assertEqual(loaded_model(torch.Tensor([5.0, 3.0]), 2.0),
                         loaded_model_twice(torch.Tensor([5.0, 3.0]), 2.0))

    def test_aten_div_tensor_out_at_3(self):
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_div_tensor_out_v3.pt"
        loaded_model = torch.jit.load(model_path)
        FileCheck().check("prim::If").run(loaded_model.graph)
        FileCheck().check_count("aten::div", 2).run(loaded_model.graph)

        buffer = io.BytesIO()
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        loaded_model_twice = torch.jit.load(buffer)
        # we check by its' code because graph variable names
        # can be different every time
        self.assertEqual(loaded_model.code, loaded_model_twice.code)

    def test_aten_full_at_4(self):
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_full_integer_value_v4.pt"
        loaded_model = torch.jit.load(model_path)
        FileCheck().check_count("aten::Float", 1).run(loaded_model.graph)
        FileCheck().check_count("aten::full", 2).run(loaded_model.graph)

        buffer = io.BytesIO()
        torch.jit.save(loaded_model, buffer)
        buffer.seek(0)
        loaded_model_twice = torch.jit.load(buffer)
        # we check by its' code because graph variable names
        # can be different every time
        self.assertEqual(loaded_model.code, loaded_model_twice.code)

    def test_aten_full_out_at_4(self):
        model_path = pytorch_test_dir + "/jit/fixtures/test_versioned_full_preserved_v4.pt"
        loaded_model = torch.jit.load(model_path)
        FileCheck().check_count("aten::full", 5).run(loaded_model.graph)
