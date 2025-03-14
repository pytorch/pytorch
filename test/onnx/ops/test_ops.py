# Owner(s): ["module: onnx"]
"""Test torch.onnx.ops."""

from __future__ import annotations

import torch
from torch.testing._internal import common_utils
from torch.onnx.ops import _symbolic_impl


class SymbolicOpsTest(common_utils.TestCase):
    def test_symbolic_has_correct_schema(self):
        torch.library.opcheck(
            _symbolic_impl._symbolic,
            ([torch.tensor(1)], "CustomOp", 1, [torch.tensor(42)]),
            dict(
                shape=[
                    1,
                ],
                attr_keys=["key"],
                attr_types=["i"],
                attr_pos=[(0, 1)],
                attr_ints=[1],
                attr_floats=[1.0],
                attr_strs=["attr"],
                metadata_props_keys=["meta_key"],
                metadata_props_values=["meta_value"],
                domain="custom_domain",
                version=42,
            ),
        )

        # Empty inputs
        torch.library.opcheck(
            _symbolic_impl._symbolic,
            ([], "CustomOp", 1, []),
            dict(
                shape=[
                    1,
                ],
                attr_keys=[],
                attr_types=[],
                attr_pos=[],
                attr_ints=[],
                attr_floats=[],
                attr_strs=[],
                metadata_props_keys=[],
                metadata_props_values=[],
            ),
        )

    def test_symbolic_multi_out_has_correct_schema(self):
        torch.library.opcheck(
            _symbolic_impl._symbolic_multi_out,
            ([torch.tensor(1)], "CustomMultiOutOp", [1, 2, 10], [torch.tensor(42)]),
            dict(
                shapes=[[1, 2], [42], []],
                attr_keys=["key"],
                attr_types=["i"],
                attr_pos=[(0, 1)],
                attr_ints=[1],
                attr_floats=[1.0],
                attr_strs=["attr"],
                metadata_props_keys=["meta_key"],
                metadata_props_values=["meta_value"],
                domain="",
                version=1,
            ),
        )

        # Empty inputs
        torch.library.opcheck(
            _symbolic_impl._symbolic_multi_out,
            ([], "CustomMultiOutOp", [], []),
            dict(
                shapes=[],
                attr_keys=[],
                attr_types=[],
                attr_pos=[],
                attr_ints=[],
                attr_floats=[],
                attr_strs=[],
                metadata_props_keys=[],
                metadata_props_values=[],
            ),
        )


if __name__ == "__main__":
    common_utils.run_tests()