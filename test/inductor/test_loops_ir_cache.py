import hashlib
import tempfile
from pathlib import Path

from torch._inductor.codecache import _build_cpp_code_hash
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


if __name__ == "__main__":
    run_tests()
