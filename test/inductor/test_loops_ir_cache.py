import hashlib
import tempfile
from pathlib import Path

import torch

from torch._inductor import config
from torch._inductor.codecache import (
    _build_cpp_code_hash,
    _collect_cpp_ir_constant_metadata_for_hash,
    FxGraphCachePickler,
    FxGraphHashDetails,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLoopsIrNativeSourceHash(TestCase):
    def digest(self, root):
        hasher = hashlib.sha256()
        _build_cpp_code_hash(str(root), hasher)
        return hasher.digest()

    def test_nested_option_sources_and_headers_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = root / "op_options"
            options.mkdir()
            source = options / "gemm_options.cpp"
            header = options / "op_config.h"
            source.write_text("implementation v1", encoding="utf-8")
            header.write_text("config v1", encoding="utf-8")
            initial = self.digest(root)
            source.write_text("implementation v2", encoding="utf-8")
            changed_source = self.digest(root)
            self.assertNotEqual(initial, changed_source)
            header.write_text("config v2", encoding="utf-8")
            self.assertNotEqual(changed_source, self.digest(root))

    def test_hash_is_independent_of_root_and_creation_order(self):
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            left, right = Path(first), Path(second)
            for root, names in ((left, ("b.h", "a.cpp")), (right, ("a.cpp", "b.h"))):
                for name in names:
                    (root / name).write_text(name, encoding="utf-8")
            self.assertEqual(self.digest(left), self.digest(right))

    def test_relative_source_name_is_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "first.cpp"
            source.write_text("same contents", encoding="utf-8")
            initial = self.digest(root)
            source.rename(root / "second.cpp")
            self.assertNotEqual(initial, self.digest(root))

    def test_nonproduction_files_do_not_change_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ir.cpp").write_text("production", encoding="utf-8")
            initial = self.digest(root)
            tests = root / "op_options" / "tests"
            tests.mkdir(parents=True)
            (tests / "op_options_cpu_test.cpp").write_text(
                "test only", encoding="utf-8"
            )
            (root / "notes.md").write_text("documentation", encoding="utf-8")
            self.assertEqual(initial, self.digest(root))

    def test_missing_native_tree_leaves_hasher_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                hashlib.sha256().digest(), self.digest(Path(directory) / "missing")
            )


class TestLoopsIrConstantCache(TestCase):
    def graph(self, value, *, nested=False, read_padding=False):
        root = torch.nn.Module()
        owner = root
        target = "table"
        if nested:
            root.child = torch.nn.Module()
            owner = root.child
            target = "child.table"
        owner.register_buffer("table", value)
        graph = torch.fx.Graph()
        table = graph.get_attr(target)
        result = table
        if read_padding:
            result = graph.call_function(
                torch.ops.aten.as_strided.default, (table, [3], [1], 1)
            )
        graph.output((result,))
        return torch.fx.GraphModule(root, graph)

    def key(self, gm):
        details = FxGraphHashDetails(gm, [], {}, [])
        return FxGraphCachePickler(gm).dumps(details)

    def test_cpp_offset_changes_key_with_equal_logical_values(self):
        left = self.graph(torch.tensor([9, 0, 2, 9])[1:3])
        right = self.graph(torch.tensor([9, 9, 0, 2])[2:4])
        self.assertEqual(left.table, right.table)
        self.assertNotEqual(left.table.storage_offset(), right.table.storage_offset())
        with config.patch(experimental_cpp_ir=True):
            self.assertNotEqual(self.key(left), self.key(right))

    def test_cpp_constant_values_still_change_key(self):
        left = self.graph(torch.tensor([0, 2]))
        right = self.graph(torch.tensor([0, 3]))
        self.assertEqual(
            _collect_cpp_ir_constant_metadata_for_hash(left)[0][:2],
            _collect_cpp_ir_constant_metadata_for_hash(right)[0][:2],
        )
        with config.patch(experimental_cpp_ir=True):
            self.assertNotEqual(self.key(left), self.key(right))

    def test_cpp_storage_extent_is_preserved(self):
        left = self.graph(torch.tensor([9, 0, 2, 9])[1:3])
        right = self.graph(torch.tensor([9, 0, 2, 9, 9])[1:3])
        self.assertEqual(left.table, right.table)
        self.assertNotEqual(
            _collect_cpp_ir_constant_metadata_for_hash(left),
            _collect_cpp_ir_constant_metadata_for_hash(right),
        )

    def test_nested_attribute_metadata_uses_structured_resolver(self):
        gm = self.graph(torch.tensor([9, 0, 2])[1:], nested=True)
        ((target, metadata, _digest),) = _collect_cpp_ir_constant_metadata_for_hash(gm)
        self.assertEqual(target, "child.table")
        self.assertEqual(metadata.storage_offset, 1)
        self.assertEqual(metadata.stride, (1,))

    def test_stock_does_not_gain_cpp_constant_metadata_field(self):
        gm = self.graph(torch.tensor([0, 2]))
        with config.patch(experimental_cpp_ir=False):
            details = FxGraphHashDetails(gm, [], {}, [])
        self.assertFalse(hasattr(details, "cpp_ir_constant_metadata"))

    def test_absent_graph_has_no_constant_metadata(self):
        self.assertEqual(_collect_cpp_ir_constant_metadata_for_hash(None), ())

    def test_native_padding_reads_do_not_collide(self):
        left = self.graph(torch.tensor([9.0, 0.0, 111.0, 2.0])[1::2], read_padding=True)
        right = self.graph(
            torch.tensor([9.0, 0.0, 222.0, 2.0])[1::2], read_padding=True
        )
        self.assertEqual(left.table, right.table)
        self.assertFalse(torch.equal(left()[0], right()[0]))
        self.assertEqual(
            _collect_cpp_ir_constant_metadata_for_hash(left)[0][:2],
            _collect_cpp_ir_constant_metadata_for_hash(right)[0][:2],
        )
        with config.patch(experimental_cpp_ir=True):
            self.assertNotEqual(self.key(left), self.key(right))

    def test_equal_storage_content_has_stable_key_without_identity(self):
        left = self.graph(torch.tensor([9.0, 0.0, 111.0, 2.0])[1::2])
        right = self.graph(torch.tensor([9.0, 0.0, 111.0, 2.0])[1::2])
        with config.patch(experimental_cpp_ir=True):
            self.assertEqual(self.key(left), self.key(right))

    def test_storage_digest_preserves_nan_payload_bits(self):
        left = self.graph(
            torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
        )
        right = self.graph(
            torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
        )
        self.assertTrue(torch.isnan(left.table).all())
        self.assertTrue(torch.isnan(right.table).all())
        self.assertNotEqual(
            _collect_cpp_ir_constant_metadata_for_hash(left)[0][2],
            _collect_cpp_ir_constant_metadata_for_hash(right)[0][2],
        )

    def test_real_storage_hash_is_safe_inside_fake_tensor_mode(self):
        gm = self.graph(torch.tensor([0, 2]))
        expected = _collect_cpp_ir_constant_metadata_for_hash(gm)
        with FakeTensorMode(allow_non_fake_inputs=True):
            actual = _collect_cpp_ir_constant_metadata_for_hash(gm)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    run_tests()
