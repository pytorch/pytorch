# Owner(s): ["module: onnx"]
"""Test torch.onnx.ops."""

from __future__ import annotations

from onnxscript import ir

import torch
from torch.onnx.ops import _symbolic_impl
from torch.testing._internal import common_utils


class SchemaTest(common_utils.TestCase):
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


class SymbolicOpsTest(common_utils.TestCase):
    def test_symbolic_accepts_valid_inputs(self):
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dict(
                int_key=1,
                float_key=1.0,
                str_key="attr",
                bool_key=True,
                list_int_key=[1, 2],
                list_float_key=[1.0, 2.0],
                list_str_key=["attr1", "attr2"],
                list_bool_key=[True, False],
            ),
            dtype=torch.float32,
            shape=[1, 2, 3],
            version=1,
            metadata_props={"meta_key": "meta_value"},
        )
        self.assertEqual(output.shape, torch.Size([1, 2, 3]))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))

    def test_symbolic_accepts_valid_inputs_empty_shape(self):
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtype=torch.float32,
            shape=[],
        )
        self.assertEqual(output.shape, torch.Size([]))

    def test_symbolic_accepts_valid_inputs_integer_types(self):
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtype=1,  # 1 is float32 in ONNX
            shape=[42],
        )
        self.assertEqual(output.dtype, torch.float32)

    def test_symbolic_accepts_valid_inputs_int4_type(self):
        output = torch.onnx.ops.symbolic(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtype=22,  # 22 is INT4 in ONNX
            shape=[42],
        )
        # We use torch uint8 for int4
        self.assertEqual(output.dtype, torch.uint8)

    def test_symbolic_is_exportable(self):
        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.onnx.ops.symbolic(
                    "custom_domain::CustomOp",
                    (x,),
                    dict(
                        int_key=1,
                        float_key=1.0,
                        str_key="attr",
                        bool_key=True,
                        list_int_key=[1, 2],
                        list_float_key=[1.0, 2.0],
                        list_str_key=["attr1", "attr2"],
                        list_bool_key=[True, False],
                    ),
                    dtype=x.dtype,
                    shape=[1, 2, 3],
                    version=1,
                    metadata_props={"meta_key": "meta_value"},
                )

        onnx_program = torch.onnx.export(
            Model(), (torch.tensor(1),), dynamo=True, verbose=False
        )
        assert onnx_program is not None
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        attributes = node.attributes
        self.assertEqual(
            attributes,
            dict(
                int_key=1,
                float_key=1.0,
                str_key="attr",
                bool_key=True,
                list_int_key=[1, 2],
                list_float_key=[1.0, 2.0],
                list_str_key=["attr1", "attr2"],
                list_bool_key=[True, False],
            ),
        )
        self.assertEqual(node.metadata_props, {"meta_key": "meta_value"})
        outputs = node.outputs
        self.assertEqual(list(outputs[0].shape), [1, 2, 3])
        self.assertEqual(outputs[0].dtype, ir.DataType.INT64)

    def test_symbolic_preserves_dynamic_shapes(self):
        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.onnx.ops.symbolic(
                    "custom_domain::CustomOp",
                    (x, y),
                    dtype=x.dtype,
                    shape=[*x.shape, *y.shape],
                    version=1,
                )

        onnx_program = torch.onnx.export(
            Model(),
            (torch.zeros(2, 3), torch.zeros(1, 2)),
            dynamic_shapes=({0: "batch"}, {1: "something_else"}),
            dynamo=True,
            verbose=False,
        )
        assert onnx_program is not None
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        inputs = onnx_program.model.graph.inputs
        self.assertEqual(str(inputs[0][0]), "batch")
        self.assertEqual(inputs[0][1], 3)
        self.assertEqual(inputs[1][0], 1)
        self.assertEqual(str(inputs[1][1]), "something_else")
        outputs = node.outputs
        self.assertEqual(str(outputs[0][0]), "batch")
        self.assertEqual(outputs[0][1], 3)
        self.assertEqual(outputs[0][2], 1)
        self.assertEqual(str(outputs[0][3]), "something_else")
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT)

    def test_symbolic_multi_out_accepts_valid_inputs(self):
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomMultiOutOp",
            (torch.tensor(1),),
            dict(
                int_key=1,
                float_key=1.0,
                str_key="attr",
                bool_key=True,
                list_int_key=[1, 2],
                list_float_key=[1.0, 2.0],
                list_str_key=["attr1", "attr2"],
                list_bool_key=[True, False],
            ),
            dtypes=(
                1,  # 1 is float32 in ONNX
                torch.int32,
                torch.float8_e4m3fn,
            ),
            shapes=([1, 2], [42], []),
            version=1,
            metadata_props={"meta_key": "meta_value"},
        )
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].shape, torch.Size([1, 2]))
        self.assertEqual(outputs[0].dtype, torch.float32)
        self.assertEqual(outputs[1].shape, torch.Size([42]))
        self.assertEqual(outputs[1].dtype, torch.int32)
        self.assertEqual(outputs[2].shape, torch.Size([]))
        self.assertEqual(outputs[2].dtype, torch.float8_e4m3fn)
        self.assertEqual(outputs[0].device, torch.device("cpu"))
        self.assertEqual(outputs[1].device, torch.device("cpu"))
        self.assertEqual(outputs[2].device, torch.device("cpu"))

    def test_symbolic_multi_out_accepts_valid_inputs_empty_shape(self):
        output = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(torch.float32,),
            shapes=[[]],
        )
        self.assertEqual(output.shape, torch.Size([]))

    def test_symbolic_multi_out_accepts_valid_inputs_integer_types(self):
        output = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(1,),  # 1 is float32 in ONNX
            shapes=[[42]],
        )
        self.assertEqual(output.dtype, torch.float32)

    def test_symbolic_multi_out_accepts_valid_inputs_int4_type(self):
        output = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(22,),  # 22 is INT4 in ONNX
            shapes=[[42]],
        )
        # We use torch uint8 for int4
        self.assertEqual(output.dtype, torch.uint8)

    def test_symbolic_multi_out_is_exportable(self):
        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.onnx.ops.symbolic_multi_out(
                    "custom_domain::CustomOp",
                    (x,),
                    dict(
                        int_key=1,
                        float_key=1.0,
                        str_key="attr",
                        bool_key=True,
                        list_int_key=[1, 2],
                        list_float_key=[1.0, 2.0],
                        list_str_key=["attr1", "attr2"],
                        list_bool_key=[True, False],
                    ),
                    dtypes=(torch.float32, torch.int32, torch.float8_e4m3fn),
                    shapes=([1, 2], [42], []),
                    version=1,
                    metadata_props={"meta_key": "meta_value"},
                )

        onnx_program = torch.onnx.export(
            Model(), (torch.tensor(1),), dynamo=True, verbose=False
        )
        assert onnx_program is not None
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        attributes = node.attributes
        self.assertEqual(
            attributes,
            dict(
                int_key=1,
                float_key=1.0,
                str_key="attr",
                bool_key=True,
                list_int_key=[1, 2],
                list_float_key=[1.0, 2.0],
                list_str_key=["attr1", "attr2"],
                list_bool_key=[True, False],
            ),
        )
        self.assertEqual(node.metadata_props, {"meta_key": "meta_value"})
        outputs = node.outputs
        self.assertEqual(list(outputs[0].shape), [1, 2])
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(list(outputs[0].shape), [42])
        self.assertEqual(outputs[1].dtype, ir.DataType.INT32)
        self.assertEqual(list(outputs[1].shape), [])
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT8E4M3FN)

    def test_symbolic_multi_out_preserves_dynamic_shapes(self):
        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                return torch.onnx.ops.symbolic_multi_out(
                    "custom_domain::CustomOp",
                    (x, y),
                    dtypes=(x.dtype, 22),  # 22 is INT4
                    shapes=[[*x.shape, *y.shape], [42]],
                    version=1,
                )

        onnx_program = torch.onnx.export(
            Model(),
            (torch.zeros(2, 3), torch.zeros(1, 2)),
            dynamic_shapes=({0: "batch"}, {1: "something_else"}),
            dynamo=True,
            verbose=False,
        )
        assert onnx_program is not None
        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "CustomOp")
        self.assertEqual(node.domain, "custom_domain")
        inputs = onnx_program.model.graph.inputs
        self.assertEqual(str(inputs[0][0]), "batch")
        self.assertEqual(inputs[0][1], 3)
        self.assertEqual(inputs[1][0], 1)
        self.assertEqual(str(inputs[1][1]), "something_else")
        outputs = node.outputs
        self.assertEqual(str(outputs[0][0]), "batch")
        self.assertEqual(outputs[0][1], 3)
        self.assertEqual(outputs[0][2], 1)
        self.assertEqual(str(outputs[0][3]), "something_else")
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT32)
        self.assertEqual(list(outputs[1].shape), 42)
        self.assertEqual(outputs[1].dtype, ir.DataType.INT4)

    def test_symbolic_multi_out_raises_when_dtypes_and_shapes_differ(self):
        with self.assertRaises(RuntimeError):
            torch.onnx.ops.symbolic_multi_out(
                "custom_domain::CustomMultiOutOp",
                (torch.tensor(1),),
                dict(
                    int_key=1,
                    float_key=1.0,
                    str_key="attr",
                    bool_key=True,
                    list_int_key=[1, 2],
                    list_float_key=[1.0, 2.0],
                    list_str_key=["attr1", "attr2"],
                    list_bool_key=[True, False],
                ),
                dtypes=(torch.float32, torch.int32),
                shapes=([1, 2], [42], []),
                version=1,
                metadata_props={"meta_key": "meta_value"},
            )

        with self.assertRaises(RuntimeError):
            torch.onnx.ops.symbolic_multi_out(
                "custom_domain::CustomMultiOutOp",
                (torch.tensor(1),),
                dict(
                    int_key=1,
                    float_key=1.0,
                    str_key="attr",
                    bool_key=True,
                    list_int_key=[1, 2],
                    list_float_key=[1.0, 2.0],
                    list_str_key=["attr1", "attr2"],
                    list_bool_key=[True, False],
                ),
                dtypes=(torch.float32,),
                shapes=([1, 2], [42]),
                version=1,
                metadata_props={"meta_key": "meta_value"},
            )


if __name__ == "__main__":
    common_utils.run_tests()
