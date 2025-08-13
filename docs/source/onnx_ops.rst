torch.onnx.ops
==============

.. automodule:: torch.onnx.ops

Symbolic Operators
------------------

Operators that can be used to create any ONNX ops in the FX graph symbolically.
These operators do not do actual computation. It's recommended that you used them
inside an ``if torch.onnx.is_in_onnx_export`` block.

.. autofunction:: torch.onnx.ops.symbolic
.. autofunction:: torch.onnx.ops.symbolic_multi_out


ONNX Operators
--------------

The following operators are implemented as native PyTorch ops and can be exported as
ONNX operators. They can be used natively in an ``nn.Module``.

For example, you can define a module::

    class Model(torch.nn.Module):
        def forward(
            self, input_data, cos_cache_data, sin_cache_data, position_ids_data
        ):
            return torch.onnx.ops.rotary_embedding(
                input_data,
                cos_cache_data,
                sin_cache_data,
                position_ids_data,
            )

and export it to ONNX using::

    input_data = torch.rand(2, 3, 4, 8)
    position_ids_data = torch.randint(0, 50, (2, 3)).long()
    sin_cache_data = torch.rand(50, 4)
    cos_cache_data = torch.rand(50, 4)
    dynamic_shapes = {
        "input_data": {0: torch.export.Dim.DYNAMIC},
        "cos_cache_data": None,
        "sin_cache_data": None,
        "position_ids_data": {0: torch.export.Dim.DYNAMIC},
    }
    onnx_program = torch.onnx.export(
        model,
        (input_data, cos_cache_data, sin_cache_data, position_ids_data),
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        opset_version=23,
    )

Printing the ONNX program will show the ONNX operators used in the graph::

    <...>
    graph(
        name=main_graph,
        inputs=(
            %"input_data"<FLOAT,[s0,3,4,8]>,
            %"cos_cache_data"<FLOAT,[50,4]>,
            %"sin_cache_data"<FLOAT,[50,4]>,
            %"position_ids_data"<INT64,[s0,3]>
        ),
        outputs=(
            %"rotary_embedding"<FLOAT,[s0,3,4,8]>
        ),
    ) {
        0 |  # rotary_embedding
            %"rotary_embedding"<FLOAT,[s0,3,4,8]> ⬅️ ::RotaryEmbedding(%"input_data", %"cos_cache_data", %"sin_cache_data", %"position_ids_data")
        return %"rotary_embedding"<FLOAT,[s0,3,4,8]>
    }

with the corresponding ``ExportedProgram``::

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, input_data: "f32[s0, 3, 4, 8]", cos_cache_data: "f32[50, 4]", sin_cache_data: "f32[50, 4]", position_ids_data: "i64[s0, 3]"):
                rotary_embedding: "f32[s0, 3, 4, 8]" = torch.ops.onnx.RotaryEmbedding.opset23(input_data, cos_cache_data, sin_cache_data, position_ids_data);  input_data = cos_cache_data = sin_cache_data = position_ids_data = None
                return (rotary_embedding,)


.. autofunction:: torch.onnx.ops.rotary_embedding

ONNX to ATen Decomposition Table
--------------------------------

You can use :func:`torch.onnx.ops.aten_decompositions` to obtain a decomposition table
to decompose ONNX operators defined above to ATen operators.

::

    class Model(torch.nn.Module):
        def forward(
            self, input_data, cos_cache_data, sin_cache_data, position_ids_data
        ):
            return torch.onnx.ops.rotary_embedding(
                input_data,
                cos_cache_data,
                sin_cache_data,
                position_ids_data,
            )

    model = Model()

    ep = torch.export.export(
        model,
        (input_data, cos_cache_data, sin_cache_data, position_ids_data),
    )
    # The program can be decomposed into aten ops
    ep_decomposed = ep.run_decompositions(torch.onnx.ops.aten_decompositions())

.. autofunction:: torch.onnx.ops.aten_decompositions
