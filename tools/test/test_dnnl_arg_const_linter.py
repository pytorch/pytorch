import tempfile
import unittest
from pathlib import Path

from tools.linter.adapters.dnnl_arg_const_linter import check_file


class TestDnnlArgConstLinter(unittest.TestCase):
    def _lint(self, text: str):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sample.cpp"
            path.write_text(text)
            return check_file(str(path))

    def test_input_requires_const_data_ptr(self):
        msgs = self._lint(
            """
void f() {
  auto src_m = make_onednn_memory(src_md, engine, src.data_ptr());
  args.insert({DNNL_ARG_SRC, src_m});
}
"""
        )
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "incorrect-pointer-kind")
        self.assertIn("const_data_ptr", msgs[0].replacement or "")

    def test_output_requires_mutable_data_ptr(self):
        msgs = self._lint(
            """
void f() {
  auto dst_m = make_onednn_memory(dst_md, engine, dst.data_ptr());
  args.insert({DNNL_ARG_DST, dst_m});
}
"""
        )
        self.assertEqual(len(msgs), 1)
        self.assertIn("mutable_data_ptr", msgs[0].replacement or "")

    def test_explicit_wrong_const_for_output_is_flagged(self):
        msgs = self._lint(
            """
void f() {
  auto dst_m = make_onednn_memory(dst_md, engine, dst.const_data_ptr());
  args.insert({DNNL_ARG_DST, dst_m});
}
"""
        )
        self.assertEqual(len(msgs), 1)
        self.assertIn("mutable_data_ptr", msgs[0].replacement or "")

    def test_explicit_wrong_mutable_for_input_is_flagged(self):
        msgs = self._lint(
            """
void f() {
  auto src_m = make_onednn_memory(src_md, engine, src.mutable_data_ptr());
  args.insert({DNNL_ARG_SRC, src_m});
}
"""
        )
        self.assertEqual(len(msgs), 1)
        self.assertIn("const_data_ptr", msgs[0].replacement or "")

    def test_correct_usage_is_not_flagged(self):
        msgs = self._lint(
            """
void f() {
  auto src_m = make_onednn_memory(src_md, engine, src.const_data_ptr());
  auto dst_m = make_onednn_memory(dst_md, engine, dst.mutable_data_ptr());
  args.insert({DNNL_ARG_SRC, src_m});
  args.insert({DNNL_ARG_DST, dst_m});
}
"""
        )
        self.assertEqual(msgs, [])

    def test_compound_attr_scales_dst_is_input(self):
        msgs = self._lint(
            """
void f() {
  auto dst_sc_m = make_onednn_memory(sc_md, engine, dst_sc.mutable_data_ptr());
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_sc_m});
}
"""
        )
        self.assertEqual(len(msgs), 1)
        self.assertIn("const_data_ptr", msgs[0].replacement or "")


if __name__ == "__main__":
    unittest.main()
