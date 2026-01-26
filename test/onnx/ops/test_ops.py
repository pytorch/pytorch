# Owner(s): ["module: onnx"]
"""Test torch.onnx.ops."""

from __future__ import annotations

import onnx_ir.passes.common as common_passes
from onnxscript import ir

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.onnx.ops import _impl, _symbolic_impl
from torch.testing._internal import common_utils


class SchemaTest(common_utils.TestCase):
    def test_symbolic_has_correct_schema(self):
        torch.library.opcheck(
            _symbolic_impl._symbolic,
            ([torch.tensor(1)], "CustomOp", 1),
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
            ([], "CustomOp", 1),
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
            ([torch.tensor(1)], "CustomMultiOutOp", [1, 2, 10]),
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
            ([], "CustomMultiOutOp", []),
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
                    (x, None),
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
                int_key=ir.AttrInt64("int_key", 1),
                float_key=ir.AttrFloat32("float_key", 1.0),
                str_key=ir.AttrString("str_key", "attr"),
                bool_key=ir.AttrInt64("bool_key", 1),
                list_int_key=ir.AttrInt64s("list_int_key", [1, 2]),
                list_float_key=ir.AttrFloat32s("list_float_key", [1.0, 2.0]),
                list_str_key=ir.AttrStrings("list_str_key", ["attr1", "attr2"]),
                list_bool_key=ir.AttrInt64s("list_bool_key", [1, 0]),
            ),
        )
        self.assertEqual(node.metadata_props["meta_key"], "meta_value")
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
        self.assertEqual(str(inputs[0].shape[0]), "batch")
        self.assertEqual(inputs[0].shape[1], 3)
        self.assertEqual(inputs[1].shape[0], 1)
        self.assertEqual(str(inputs[1].shape[1]), "something_else")
        outputs = node.outputs
        self.assertEqual(str(outputs[0].shape[0]), "batch")
        self.assertEqual(outputs[0].shape[1], 3)
        self.assertEqual(outputs[0].shape[2], 1)
        self.assertEqual(str(outputs[0].shape[3]), "something_else")
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
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(torch.float32,),
            shapes=[[]],
        )
        self.assertEqual(outputs[0].shape, torch.Size([]))

    def test_symbolic_multi_out_accepts_valid_inputs_integer_types(self):
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(1,),  # 1 is float32 in ONNX
            shapes=[[42]],
        )
        self.assertEqual(outputs[0].dtype, torch.float32)

    def test_symbolic_multi_out_accepts_valid_inputs_int4_type(self):
        outputs = torch.onnx.ops.symbolic_multi_out(
            "custom_domain::CustomOp",
            (torch.tensor(1),),
            dtypes=(22,),  # 22 is INT4 in ONNX
            shapes=[[42]],
        )
        # We use torch uint8 for int4
        self.assertEqual(outputs[0].dtype, torch.uint8)

    def test_symbolic_multi_out_is_exportable(self):
        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return torch.onnx.ops.symbolic_multi_out(
                    "custom_domain::CustomOp",
                    (x, None),
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
                int_key=ir.AttrInt64("int_key", 1),
                float_key=ir.AttrFloat32("float_key", 1.0),
                str_key=ir.AttrString("str_key", "attr"),
                bool_key=ir.AttrInt64("bool_key", 1),
                list_int_key=ir.AttrInt64s("list_int_key", [1, 2]),
                list_float_key=ir.AttrFloat32s("list_float_key", [1.0, 2.0]),
                list_str_key=ir.AttrStrings("list_str_key", ["attr1", "attr2"]),
                list_bool_key=ir.AttrInt64s("list_bool_key", [1, 0]),
            ),
        )
        self.assertEqual(node.metadata_props["meta_key"], "meta_value")
        outputs = node.outputs
        self.assertEqual(list(outputs[0].shape), [1, 2])
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(list(outputs[1].shape), [42])
        self.assertEqual(outputs[1].dtype, ir.DataType.INT32)
        self.assertEqual(list(outputs[2].shape), [])
        self.assertEqual(outputs[2].dtype, ir.DataType.FLOAT8E4M3FN)

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
        self.assertEqual(str(inputs[0].shape[0]), "batch")
        self.assertEqual(inputs[0].shape[1], 3)
        self.assertEqual(inputs[1].shape[0], 1)
        self.assertEqual(str(inputs[1].shape[1]), "something_else")
        outputs = node.outputs
        self.assertEqual(str(outputs[0].shape[0]), "batch")
        self.assertEqual(outputs[0].shape[1], 3)
        self.assertEqual(outputs[0].shape[2], 1)
        self.assertEqual(str(outputs[0].shape[3]), "something_else")
        self.assertEqual(outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(list(outputs[1].shape), [42])
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


@common_utils.instantiate_parametrized_tests
class NativeOnnxOpsTest(common_utils.TestCase):
    def export(self, model, args=(), kwargs=None, **options) -> torch.onnx.ONNXProgram:
        onnx_program = torch.onnx.export(
            model,
            args,
            kwargs=kwargs,
            dynamo=True,
            fallback=False,
            verbose=False,
            **options,
        )
        assert onnx_program is not None
        common_passes.CheckerPass()(onnx_program.model)
        return onnx_program

    def test_onnx_ops_can_be_decomposed_to_aten(self):
        input_data = torch.rand(2, 3, 4, 8)
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        class Model(torch.nn.Module):
            def forward(
                self, input_data, cos_cache_data, sin_cache_data, position_ids_data
            ):
                return torch.onnx.ops.rotary_embedding(
                    input_data,
                    cos_cache_data,
                    sin_cache_data,
                    position_ids_data,
                    interleaved=True,
                )

        model = Model()

        ep = torch.export.export(
            model,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
        )
        self.assertIn(
            "onnx.RotaryEmbedding.opset23",
            [str(node.target) for node in ep.graph.nodes],
        )
        # The program can be decomposed into aten ops so it is fully compatible with the PyTorch ecosystem
        aten_decomped = ep.run_decompositions(torch.onnx.ops.aten_decompositions())
        self.assertNotIn(
            "onnx.RotaryEmbedding.opset23",
            [str(node.target) for node in aten_decomped.graph.nodes],
        )
        torch.testing.assert_close(
            aten_decomped.module()(
                input_data, cos_cache_data, sin_cache_data, position_ids_data
            ),
            model(input_data, cos_cache_data, sin_cache_data, position_ids_data),
        )

    def test_rotary_embedding_opcheck(self):
        input_data = torch.rand(2, 3, 4, 8)
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        torch.library.opcheck(
            _impl.rotary_embedding_23,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
        )

    def test_rotary_embedding(self):
        input_data = torch.rand(2, 3, 4, 8)
        position_ids_data = torch.randint(0, 50, (2, 4)).long()
        sin_cache_data = torch.rand(50, 4)
        cos_cache_data = torch.rand(50, 4)

        # Eager mode is supported. Autograd is also supported so users can choose to use the op
        # in development and production
        result = torch.onnx.ops.rotary_embedding(
            input_data, cos_cache_data, sin_cache_data, position_ids_data
        )
        self.assertEqual(result.shape, input_data.shape)

        class Model(torch.nn.Module):
            def forward(
                self, input_data, cos_cache_data, sin_cache_data, position_ids_data
            ):
                return torch.onnx.ops.rotary_embedding(
                    input_data,
                    cos_cache_data,
                    sin_cache_data,
                    position_ids_data,
                    interleaved=True,
                )

        model = Model()

        # Dynamic shapes are supported
        dynamic_shapes = {
            "input_data": {0: torch.export.Dim.DYNAMIC},
            "cos_cache_data": None,
            "sin_cache_data": None,
            "position_ids_data": {0: torch.export.Dim.DYNAMIC},
        }

        onnx_program = self.export(
            model,
            (input_data, cos_cache_data, sin_cache_data, position_ids_data),
            dynamic_shapes=dynamic_shapes,
            opset_version=23,
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("RotaryEmbedding", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_rotary_embedding_3d(self):
        num_heads = 2
        input_data = torch.rand(2, 3, 8)
        sin_cache_data = torch.rand(2, 3, 2)
        cos_cache_data = torch.rand(2, 3, 2)

        class Model(torch.nn.Module):
            def forward(self, input_data, cos_cache_data, sin_cache_data):
                return torch.onnx.ops.rotary_embedding(
                    input_data,
                    cos_cache_data,
                    sin_cache_data,
                    num_heads=num_heads,
                )

        model = Model()

        # Dynamic shapes are supported
        dynamic_shapes = {
            "input_data": {0: torch.export.Dim.DYNAMIC},
            "cos_cache_data": {0: torch.export.Dim.DYNAMIC},
            "sin_cache_data": {0: torch.export.Dim.DYNAMIC},
        }

        onnx_program = self.export(
            model,
            (input_data, cos_cache_data, sin_cache_data),
            dynamic_shapes=dynamic_shapes,
            opset_version=23,
        )
        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("RotaryEmbedding", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_without_past_kv_caches(self):
        """Test basic attention functionality."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test eager mode
        torch.library.opcheck(_impl.attention_23, (Q, K, V))
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V
        )

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))
        self.assertEqual(present_key.shape, K.shape)
        self.assertEqual(present_value.shape, V.shape)
        self.assertEqual(
            qk_output.shape, (batch_size, q_num_heads, q_seq_len, kv_seq_len)
        )

    def test_attention_3d_inputs(self):
        """Test attention with 3D inputs (requires num_heads parameters)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_seq_len, q_num_heads * head_size)
        K = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)
        V = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)

        torch.library.opcheck(
            _impl.attention_23,
            (Q, K, V),
            dict(q_num_heads=q_num_heads, kv_num_heads=kv_num_heads),
        )
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
        )

        # Output should be reshaped back to 3D
        self.assertEqual(output.shape, (batch_size, q_seq_len, q_num_heads * head_size))
        self.assertEqual(
            present_key.shape, (batch_size, kv_num_heads, kv_seq_len, head_size)
        )
        self.assertEqual(
            present_value.shape, (batch_size, kv_num_heads, kv_seq_len, head_size)
        )

    @common_utils.parametrize(
        "name, kv_num_heads",
        [
            ("group_query_attention", 4),
            ("multi_query_attention", 1),
        ],
    )
    def test_attention_kv_num_heads(self, name: str, kv_num_heads: int):
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads = 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V))
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, None, enable_gqa=True
        )

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))
        self.assertEqual(present_key.shape, K.shape)
        self.assertEqual(present_value.shape, V.shape)
        torch.testing.assert_close(output, expected)

    def test_attention_mqa(self):
        """Test Multi-Query Attention (MQA)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 1  # MQA: kv_num_heads = 1
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V))
        output, present_key, present_value, qk_output = torch.onnx.ops.attention(
            Q, K, V
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, None, enable_gqa=True
        )

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))
        torch.testing.assert_close(output, expected)

    def test_attention_with_2d_mask(self):
        """Test attention with 2D attention mask (q_seq_len, kv_seq_len)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test with boolean mask
        bool_mask = torch.randint(0, 2, (q_seq_len, kv_seq_len), dtype=torch.bool)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=bool_mask))
        output_bool, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=bool_mask)

        # Test with float mask
        float_mask = torch.randn(q_seq_len, kv_seq_len)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=float_mask))
        output_float, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=float_mask)

        self.assertEqual(
            output_bool.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )
        self.assertEqual(
            output_float.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )

    def test_attention_with_4d_mask(self):
        """Test attention with 4D attention mask (batch_size, num_heads, q_seq_len, kv_seq_len)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test with boolean mask
        bool_mask = torch.randint(
            0, 2, (batch_size, q_num_heads, q_seq_len, kv_seq_len), dtype=torch.bool
        )
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=bool_mask))
        output_bool, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=bool_mask)

        # Test with float mask
        float_mask = torch.randn(batch_size, q_num_heads, q_seq_len, kv_seq_len)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=float_mask))
        output_float, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=float_mask)

        self.assertEqual(
            output_bool.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )
        self.assertEqual(
            output_float.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )

    def test_attention_with_zero_float_mask(self):
        """Test attention with zero float mask."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        zero_mask = torch.zeros(q_seq_len, kv_seq_len)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=zero_mask))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=zero_mask)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_with_causal_mask_pattern(self):
        """Test attention with lower triangular causal mask pattern."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 4  # Square for causal
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Create a lower triangular causal mask
        causal_mask = torch.tril(torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool))
        torch.library.opcheck(
            _impl.attention_23, (Q, K, V), dict(attn_mask=causal_mask)
        )
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=causal_mask)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_with_gqa_and_mask(self):
        """Test attention with GQA and different mask shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 4  # GQA
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test 2D mask with GQA
        mask_2d = torch.randint(0, 2, (q_seq_len, kv_seq_len), dtype=torch.bool)
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=mask_2d))
        output_2d, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask_2d)

        # Test 4D mask with GQA (note: using q_num_heads for mask heads)
        mask_4d = torch.randint(
            0, 2, (batch_size, q_num_heads, q_seq_len, kv_seq_len), dtype=torch.bool
        )
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(attn_mask=mask_4d))
        output_4d, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask_4d)

        self.assertEqual(
            output_2d.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )
        self.assertEqual(
            output_4d.shape, (batch_size, q_num_heads, q_seq_len, head_size)
        )

    def test_attention_causal(self):
        """Test causal attention."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 4  # Square for causal
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(is_causal=True))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, is_causal=True)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_with_past_kv(self):
        """Test attention with past key/value caches."""
        batch_size, q_seq_len, kv_seq_len, past_seq_len = 2, 4, 6, 3
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        past_key = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)
        past_value = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)

        torch.library.opcheck(
            _impl.attention_23,
            (Q, K, V),
            dict(past_key=past_key, past_value=past_value),
        )
        output, present_key, present_value, _ = torch.onnx.ops.attention(
            Q, K, V, past_key=past_key, past_value=past_value
        )

        # Present key/value should include past + current
        expected_total_seq_len = past_seq_len + kv_seq_len
        self.assertEqual(
            present_key.shape,
            (batch_size, kv_num_heads, expected_total_seq_len, head_size),
        )
        self.assertEqual(
            present_value.shape,
            (batch_size, kv_num_heads, expected_total_seq_len, head_size),
        )

    def test_attention_with_softcap(self):
        """Test attention with softcap."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(softcap=30.0))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, softcap=30.0)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_qk_output_modes(self):
        """Test different QK matmul output modes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        for mode in [0, 1, 2, 3]:
            torch.library.opcheck(
                _impl.attention_23,
                (Q, K, V),
                dict(qk_matmul_output_mode=mode),
            )
            output, _, _, qk_output = torch.onnx.ops.attention(
                Q, K, V, qk_matmul_output_mode=mode
            )

            self.assertEqual(
                output.shape, (batch_size, q_num_heads, q_seq_len, head_size)
            )
            self.assertEqual(
                qk_output.shape, (batch_size, q_num_heads, q_seq_len, kv_seq_len)
            )

    def test_attention_custom_scale(self):
        """Test attention with custom scale factor."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        custom_scale = 0.25
        torch.library.opcheck(_impl.attention_23, (Q, K, V), dict(scale=custom_scale))
        output, _, _, _ = torch.onnx.ops.attention(Q, K, V, scale=custom_scale)

        self.assertEqual(output.shape, (batch_size, q_num_heads, q_seq_len, head_size))

    def test_attention_export(self):
        """Test that attention can be exported to ONNX."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class AttentionModel(torch.nn.Module):
            def forward(self, Q, K, V):
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V)
                return output

        model = AttentionModel()

        onnx_program = self.export(
            model,
            (Q, K, V),
            opset_version=23,
        )

        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("Attention", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_dynamic_shapes(self):
        """Test attention export with dynamic shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        attn_mask = torch.randint(
            0, 2, (batch_size, 1, q_seq_len, kv_seq_len), dtype=torch.bool
        )

        class AttentionModel(torch.nn.Module):
            def forward(self, Q, K, V, attn_mask):
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=attn_mask)
                return output

        model = AttentionModel()

        dynamic_shapes = {
            "Q": {0: "batch", 2: "q_seq_len"},
            "K": {0: "batch", 2: "kv_seq_len"},
            "V": {0: "batch", 2: "kv_seq_len"},
            "attn_mask": {0: "batch", 2: "q_seq_len", 3: "kv_seq_len"},
        }

        onnx_program = self.export(
            model,
            (Q, K, V, attn_mask),
            dynamic_shapes=dynamic_shapes,
            opset_version=23,
        )

        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("Attention", onnx_program.model.graph.node(0).op_type)
        node = onnx_program.model.graph.node(0)
        # Verify inputs
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(
            node.inputs[0].shape, ["batch", q_num_heads, "q_seq_len", head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, ["batch", kv_num_heads, "kv_seq_len", head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, ["batch", kv_num_heads, "kv_seq_len", head_size]
        )

        # Verify default attributes (should be minimal)
        self.assertEqual(len(node.attributes), 0)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_3d_export(self):
        """Test attention export with 3D inputs."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_seq_len, q_num_heads * head_size)
        K = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)
        V = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)

        class AttentionModel(torch.nn.Module):
            def forward(self, Q, K, V):
                output, _, _, _ = torch.onnx.ops.attention(
                    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
                )
                return output

        model = AttentionModel()

        onnx_program = self.export(
            model,
            (Q, K, V),
            opset_version=23,
        )

        self.assertEqual(onnx_program.model.opset_imports[""], 23)
        self.assertEqual("Attention", onnx_program.model.graph.node(0).op_type)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_decomposition(self):
        """Test that attention can be decomposed to aten ops."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class AttentionModel(torch.nn.Module):
            def forward(self, Q, K, V):
                output, present_key, present_value, qk_output = (
                    torch.onnx.ops.attention(Q, K, V)
                )
                return output

        model = AttentionModel()

        ep = torch.export.export(model, (Q, K, V))
        self.assertIn(
            "onnx.Attention.opset23",
            [str(node.target) for node in ep.graph.nodes],
        )

        # The program can be decomposed into aten ops
        aten_decomped = ep.run_decompositions(torch.onnx.ops.aten_decompositions())
        self.assertNotIn(
            "onnx.Attention.opset23",
            [str(node.target) for node in aten_decomped.graph.nodes],
        )

        # Results should match
        torch.testing.assert_close(
            aten_decomped.module()(Q, K, V),
            model(Q, K, V),
        )

    def test_attention_export_with_past_key_value(self):
        """Test export with past_key, past_value to ensure the optional input order is correct."""
        batch_size, q_seq_len, kv_seq_len, past_seq_len = 2, 4, 6, 3
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        past_key = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)
        past_value = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)

        class Model(torch.nn.Module):
            def forward(self, Q, K, V, past_key, past_value):
                output, present_key, present_value, _ = torch.onnx.ops.attention(
                    Q,
                    K,
                    V,
                    past_key=past_key,
                    attn_mask=None,
                    # Switched argument order
                    past_value=past_value,
                )
                return output, present_key, present_value

        model = Model()
        onnx_program = self.export(
            model, (Q, K, V, past_key, past_value), opset_version=23
        )

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify all 6 inputs are present
        self.assertEqual(
            len(node.inputs), 6
        )  # Q, K, V, attn_mask, past_key, past_value
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_num_heads, q_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertIsNone(node.inputs[3])
        self.assertEqual(
            node.inputs[4].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[5].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_all_optional_inputs(self):
        """Test export with all optional inputs: mask, past_key, past_value."""
        batch_size, q_seq_len, kv_seq_len, past_seq_len = 2, 4, 6, 3
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        attn_mask = torch.randint(
            0, 2, (1, 1, q_seq_len, kv_seq_len + past_seq_len), dtype=torch.bool
        )
        past_key = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)
        past_value = torch.rand(batch_size, kv_num_heads, past_seq_len, head_size)

        class FullAttentionModel(torch.nn.Module):
            def forward(self, Q, K, V, attn_mask, past_key, past_value):
                output, present_key, present_value, qk_matmul = (
                    torch.onnx.ops.attention(
                        Q,
                        K,
                        V,
                        attn_mask=attn_mask,
                        past_key=past_key,
                        past_value=past_value,
                    )
                )
                return output, present_key, present_value, qk_matmul

        model = FullAttentionModel()
        onnx_program = self.export(
            model, (Q, K, V, attn_mask, past_key, past_value), opset_version=23
        )

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify all 6 inputs are present
        self.assertEqual(
            len(node.inputs), 6
        )  # Q, K, V, attn_mask, past_key, past_value
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_num_heads, q_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[3].shape, [1, 1, q_seq_len, kv_seq_len + past_seq_len]
        )
        self.assertEqual(
            node.inputs[4].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[5].shape, [batch_size, kv_num_heads, past_seq_len, head_size]
        )
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_3d_with_num_heads_attributes(self):
        """Test export with 3D inputs and explicit num_heads attributes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 4  # GQA
        head_size = 64

        Q = torch.rand(batch_size, q_seq_len, q_num_heads * head_size)
        K = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)
        V = torch.rand(batch_size, kv_seq_len, kv_num_heads * head_size)

        class Attention3DModel(torch.nn.Module):
            def forward(self, Q, K, V):
                output, _, _, _ = torch.onnx.ops.attention(
                    Q, K, V, q_num_heads=q_num_heads, kv_num_heads=kv_num_heads
                )
                return output

        model = Attention3DModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify 3D input shapes
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_seq_len, q_num_heads * head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_seq_len, kv_num_heads * head_size]
        )
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_seq_len, kv_num_heads * head_size]
        )

        # Verify num_heads attributes are set
        attrs = node.attributes
        self.assertIn("q_num_heads", attrs)
        self.assertIn("kv_num_heads", attrs)
        self.assertEqual(attrs["q_num_heads"].value, q_num_heads)
        self.assertEqual(attrs["kv_num_heads"].value, kv_num_heads)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_all_attributes(self):
        """Test export with all possible attributes set."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class FullAttributesModel(torch.nn.Module):
            def forward(self, Q, K, V):
                output, _, _, _ = torch.onnx.ops.attention(
                    Q,
                    K,
                    V,
                    is_causal=True,
                    qk_matmul_output_mode=2,
                    scale=0.25,
                    softcap=30.0,
                    softmax_precision=1,  # FLOAT
                )
                return output

        model = FullAttributesModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify all attributes are set correctly
        attrs = node.attributes
        self.assertIn("is_causal", attrs)
        self.assertIn("qk_matmul_output_mode", attrs)
        self.assertIn("scale", attrs)
        self.assertIn("softcap", attrs)
        self.assertIn("softmax_precision", attrs)

        self.assertEqual(attrs["is_causal"].value, 1)  # True as int
        self.assertEqual(attrs["qk_matmul_output_mode"].value, 2)
        self.assertAlmostEqual(attrs["scale"].value, 0.25, places=6)
        self.assertAlmostEqual(attrs["softcap"].value, 30.0, places=6)
        self.assertEqual(attrs["softmax_precision"].value, 1)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_with_different_mask_shapes(self):
        """Test export with different attention mask shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        # Test 2D mask
        mask_2d = torch.randint(0, 2, (q_seq_len, kv_seq_len), dtype=torch.bool)

        class Mask2DModel(torch.nn.Module):
            def forward(self, Q, K, V, mask):
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model_2d = Mask2DModel()
        onnx_program_2d = self.export(model_2d, (Q, K, V, mask_2d), opset_version=23)

        node_2d = onnx_program_2d.model.graph.node(0)
        self.assertEqual(node_2d.inputs[3].shape, [q_seq_len, kv_seq_len])
        onnx_testing.assert_onnx_program(onnx_program_2d)

        # Test 3D mask
        mask_3d = torch.randint(
            0, 2, (batch_size, 1, q_seq_len, kv_seq_len), dtype=torch.bool
        )

        class Mask3DModel(torch.nn.Module):
            def forward(self, Q, K, V, mask):
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model_3d = Mask3DModel()
        onnx_program_3d = self.export(model_3d, (Q, K, V, mask_3d), opset_version=23)

        node_3d = onnx_program_3d.model.graph.node(0)
        self.assertEqual(
            node_3d.inputs[3].shape, [batch_size, 1, q_seq_len, kv_seq_len]
        )
        onnx_testing.assert_onnx_program(onnx_program_3d)

        # Test 4D mask
        mask_4d = torch.randint(
            0, 2, (batch_size, q_num_heads, q_seq_len, kv_seq_len), dtype=torch.bool
        )

        class Mask4DModel(torch.nn.Module):
            def forward(self, Q, K, V, mask):
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model_4d = Mask4DModel()
        onnx_program_4d = self.export(model_4d, (Q, K, V, mask_4d), opset_version=23)

        node_4d = onnx_program_4d.model.graph.node(0)
        self.assertEqual(
            node_4d.inputs[3].shape, [batch_size, q_num_heads, q_seq_len, kv_seq_len]
        )
        onnx_testing.assert_onnx_program(onnx_program_4d)

    def test_attention_export_with_float_mask(self):
        """Test export with float attention mask."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        float_mask = torch.randn(q_seq_len, kv_seq_len)

        class FloatMaskModel(torch.nn.Module):
            def forward(self, Q, K, V, mask):
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V, attn_mask=mask)
                return output

        model = FloatMaskModel()
        onnx_program = self.export(model, (Q, K, V, float_mask), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")
        self.assertEqual(node.inputs[3].shape, [q_seq_len, kv_seq_len])
        # Verify the mask input has float dtype in the ONNX model
        self.assertEqual(node.inputs[3].dtype, ir.DataType.FLOAT)
        onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_qk_output_modes(self):
        """Test export with different QK output modes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        for mode in [0, 1, 2, 3]:

            class QKOutputModel(torch.nn.Module):
                def __init__(self, qk_mode):
                    super().__init__()
                    self.qk_mode = qk_mode

                def forward(self, Q, K, V):
                    output, _, _, qk_output = torch.onnx.ops.attention(
                        Q, K, V, qk_matmul_output_mode=self.qk_mode
                    )
                    return output, qk_output

            model = QKOutputModel(mode)
            onnx_program = self.export(model, (Q, K, V), opset_version=23)

            node = onnx_program.model.graph.node(0)
            self.assertEqual(node.op_type, "Attention")

            # Verify qk_matmul_output_mode attribute
            attrs = node.attributes
            if mode != 0:
                self.assertIn("qk_matmul_output_mode", attrs)
                self.assertEqual(attrs["qk_matmul_output_mode"].value, mode)

            # Verify 4 outputs (output, present_key, present_value, qk_output)
            self.assertEqual(len(node.outputs), 4)
            onnx_testing.assert_onnx_program(onnx_program)

    def test_attention_export_mqa(self):
        """Test export with Multi-Query Attention (MQA)."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 1  # MQA
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class MQAModel(torch.nn.Module):
            def forward(self, Q, K, V):
                output, _, _, _ = torch.onnx.ops.attention(Q, K, V)
                return output

        model = MQAModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify MQA tensor shapes
        self.assertEqual(
            node.inputs[0].shape, [batch_size, q_num_heads, q_seq_len, head_size]
        )
        self.assertEqual(
            node.inputs[1].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )  # kv_num_heads = 1
        self.assertEqual(
            node.inputs[2].shape, [batch_size, kv_num_heads, kv_seq_len, head_size]
        )
        onnx_testing.assert_onnx_program(onnx_program)

    @common_utils.parametrize(
        "precision_enum, precision_name",
        [
            (1, "FLOAT"),
            (10, "FLOAT16"),
            (11, "DOUBLE"),
            (16, "BFLOAT16"),
        ],
    )
    def test_attention_export_with_softmax_precision(
        self, precision_enum, precision_name: str
    ):
        """Test export with different softmax precision values."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 8
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class SoftmaxPrecisionModel(torch.nn.Module):
            def __init__(self, precision):
                super().__init__()
                self.precision = precision

            def forward(self, Q, K, V):
                output, _, _, _ = torch.onnx.ops.attention(
                    Q, K, V, softmax_precision=self.precision
                )
                return output

        model = SoftmaxPrecisionModel(precision_enum)
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        # Verify softmax_precision attribute
        attrs = node.attributes
        self.assertIn("softmax_precision", attrs)
        self.assertEqual(attrs["softmax_precision"].value, precision_enum)
        onnx_testing.assert_onnx_program(onnx_program, atol=2e-3, rtol=6e-3)

    def test_attention_export_gqa(self):
        """Test export and verify output tensor shapes."""
        batch_size, q_seq_len, kv_seq_len = 2, 4, 6
        q_num_heads, kv_num_heads = 8, 4  # GQA
        head_size = 64

        Q = torch.rand(batch_size, q_num_heads, q_seq_len, head_size)
        K = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)
        V = torch.rand(batch_size, kv_num_heads, kv_seq_len, head_size)

        class AttentionOutputsModel(torch.nn.Module):
            def forward(self, Q, K, V):
                result, _, _, _ = torch.onnx.ops.attention(Q, K, V)
                return result

        model = AttentionOutputsModel()
        onnx_program = self.export(model, (Q, K, V), opset_version=23)

        node = onnx_program.model.graph.node(0)
        self.assertEqual(node.op_type, "Attention")

        graph_outputs = onnx_program.model.graph.outputs
        # output: (batch_size, q_num_heads, q_seq_len, head_size)
        self.assertEqual(
            list(graph_outputs[0].shape),
            [batch_size, q_num_heads, q_seq_len, head_size],
        )

        onnx_testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    common_utils.run_tests()
