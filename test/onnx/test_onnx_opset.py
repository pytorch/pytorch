# Owner(s): ["module: onnx"]

import io
import itertools

import onnx

import pytorch_test_common

import torch
import torch.onnx
from torch.nn import Module
from torch.onnx import producer_name, producer_version
from torch.onnx._globals import GLOBALS
from torch.testing._internal import common_utils


def check_onnx_opset_operator(
    model, ops, opset_version=GLOBALS.export_onnx_opset_version
):
    # check_onnx_components
    assert (
        model.producer_name == producer_name
        and model.producer_version == producer_version
        and model.opset_import[0].version == opset_version
    )

    # check the schema with the onnx checker
    onnx.checker.check_model(model)

    # check target type and attributes
    graph = model.graph
    # ops should contain an object for each node
    # in graph.node, in the right order.
    # At least the op_name should be specified,
    # but the op's attributes can optionally be
    # specified as well
    assert len(ops) == len(graph.node)
    for i in range(0, len(ops)):
        assert graph.node[i].op_type == ops[i]["op_name"]
        if "attributes" in ops[i]:
            attributes = ops[i]["attributes"]
            assert len(attributes) == len(graph.node[i].attribute)
            for j in range(0, len(attributes)):
                for attribute_field in attributes[j].keys():
                    assert attributes[j][attribute_field] == getattr(
                        graph.node[i].attribute[j], attribute_field
                    )


def check_onnx_opsets_operator(
    module,
    x,
    ops,
    opset_versions,
    training=torch.onnx.TrainingMode.EVAL,
    input_names=None,
    dynamic_axes=None,
):
    for opset_version in opset_versions:
        f = io.BytesIO()
        torch.onnx.export(
            module,
            x,
            f,
            opset_version=opset_version,
            training=training,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
        )
        model = onnx.load(io.BytesIO(f.getvalue()))
        check_onnx_opset_operator(model, ops[opset_version], opset_version)


class TestONNXOpset(pytorch_test_common.ExportTestCase):
    def test_opset_fallback(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.isnan(x)

        ops = [{"op_name": "IsNaN"}]
        ops = {9: ops, 10: ops}
        x = torch.tensor([1.0, float("nan"), 2.0])
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_topk(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.topk(x, 3)

        ops_9 = [
            {
                "op_name": "TopK",
                "attributes": [
                    {"name": "axis", "i": -1, "type": 2},
                    {"name": "k", "i": 3, "type": 2},
                ],
            }
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]},
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.arange(1.0, 6.0, requires_grad=True)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        # test with dynamic k
        class MyModuleDynamic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input, k):
                return torch.topk(input, k)

        ops_10 = [
            {"op_name": "Constant", "attributes": [{"name": "value", "type": 4}]},
            {"op_name": "Reshape"},
            {"op_name": "TopK", "attributes": [{"name": "axis", "i": -1, "type": 2}]},
        ]
        ops = {10: ops_10}
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        module = MyModuleDynamic()
        check_onnx_opsets_operator(module, (x, k), ops, opset_versions=[10])

    def test_maxpool(self):
        module = torch.nn.MaxPool1d(2, stride=1)

        ops_9 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        ops_10 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [1], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[9, 10])

        # add test with dilations
        module = torch.nn.MaxPool1d(2, stride=1, dilation=2)

        ops_10 = [
            {
                "op_name": "MaxPool",
                "attributes": [
                    {"name": "ceil_mode", "i": 0, "type": 2},
                    {"name": "dilations", "ints": [2], "type": 7},
                    {"name": "kernel_shape", "ints": [2], "type": 7},
                    {"name": "pads", "ints": [0, 0], "type": 7},
                    {"name": "strides", "ints": [1], "type": 7},
                ],
            }
        ]
        ops = {10: ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

    def test_upsample(self):
        class MyModule(Module):
            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                size = [int(i) for i in size]
                return torch.nn.functional.interpolate(x, size=size, mode="nearest")

        module = MyModule()
        ops8 = [
            {
                "op_name": "Upsample",
                "attributes": [
                    {"name": "mode", "s": (b"nearest"), "type": 3},
                    {"name": "scales", "floats": [1.0, 1.0, 2.0, 2.0], "type": 6},
                ],
            }
        ]
        ops9 = [
            {"op_name": "Constant"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops = {8: ops8, 9: ops9}
        x = torch.randn(2, 2, 2, 2)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])

    def test_cast_constant(self):
        class MyModule(Module):
            def forward(self, x):
                return x - 1

        module = MyModule()
        ops_8 = [
            {"op_name": "Constant"},
            {"op_name": "Cast", "attributes": [{"name": "to", "i": 7, "type": 2}]},
            {"op_name": "Sub"},
        ]
        ops_9 = [{"op_name": "Constant"}, {"op_name": "Sub"}]
        ops = {8: ops_8, 9: ops_9}
        x = torch.ones(5, 6, dtype=torch.long)
        check_onnx_opsets_operator(module, x, ops, opset_versions=[8, 9])

    def test_slice(self):
        class MyModule(Module):
            def forward(self, x):
                return x[0:1]

        ops_9 = [
            {
                "op_name": "Slice",
                "attributes": [
                    {"name": "axes", "ints": [0], "type": 7},
                    {"name": "ends", "ints": [1], "type": 7},
                    {"name": "starts", "ints": [0], "type": 7},
                ],
            }
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(3)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1 : x.size(0)]

        module = DynamicSliceModel()
        x = torch.rand(1, 2)
        ops_10 = [
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather", "attributes": [{"name": "axis", "i": 0, "type": 2}]},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {
                "op_name": "Unsqueeze",
                "attributes": [{"name": "axes", "i": 0, "type": 7}],
            },
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {10: ops_10}
        check_onnx_opsets_operator(
            module,
            x,
            ops,
            opset_versions=[10],
            input_names=["x"],
            dynamic_axes={"x": [0, 1]},
        )

        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {10: ops_10}
        check_onnx_opsets_operator(module, x, ops, opset_versions=[10])

    def test_flip(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice", "attributes": []},
        ]
        ops = {10: ops_10}
        import numpy

        x = torch.tensor(numpy.arange(6.0).reshape(2, 3))
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[10])

    def test_dropout(self):
        class MyModule(Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        x = torch.randn(1, 2, 3)

        # we should only export the onnx Dropout op in training mode; test both modes

        # test training mode
        ops = [
            {
                "op_name": "Dropout",
                "attributes": [{"name": "ratio", "f": 0.5, "type": 1}],
            }
        ]
        ops = {9: ops, 10: ops}
        check_onnx_opsets_operator(
            MyModule(),
            x,
            ops,
            opset_versions=[9, 10],
            training=torch.onnx.TrainingMode.TRAINING,
        )

        # test eval mode
        ops = [{"op_name": "Identity"}]
        ops = {9: ops, 10: ops}
        check_onnx_opsets_operator(
            MyModule(),
            x,
            ops,
            opset_versions=[9, 10],
            training=torch.onnx.TrainingMode.EVAL,
        )

    def test_full(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.full((3, 4), x)

        ops = [
            {"op_name": "Constant"},
            {"op_name": "ConstantOfShape"},
            {"op_name": "Add"},
        ]
        ops = {9: ops, 10: ops}
        x = torch.tensor(12.0)
        check_onnx_opsets_operator(MyModule(), x, ops, opset_versions=[9, 10])

    def test_interpolate(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                return torch.nn.functional.interpolate(x, size=size, mode="nearest")

        ops_9 = [
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Concat"},
            {"op_name": "Cast"},
            {"op_name": "Shape"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops_10 = [
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Gather"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Constant"},
            {"op_name": "Mul"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Unsqueeze"},
            {"op_name": "Concat"},
            {"op_name": "Cast"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {
                "op_name": "Resize",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]

        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        check_onnx_opsets_operator(
            MyModel(),
            x,
            ops,
            opset_versions=[9, 10],
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2, 3]},
        )

        ops_9 = [
            {"op_name": "Constant"},
            {"op_name": "Shape"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {"op_name": "Shape"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Constant"},
            {"op_name": "Slice"},
            {"op_name": "Cast"},
            {"op_name": "Div"},
            {"op_name": "Constant"},
            {"op_name": "Concat"},
            {"op_name": "Resize"},
        ]

        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        check_onnx_opsets_operator(MyModel(), x, ops, opset_versions=[9, 10])

        class MyDynamicModel(torch.nn.Module):
            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                # work around for now: turn the dynamic sizes into constant
                size = [int(i) for i in size]
                return torch.nn.functional.interpolate(x, size=size, mode="nearest")

        ops_9 = [
            {"op_name": "Constant"},
            {
                "op_name": "Upsample",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops_10 = [
            {"op_name": "Constant"},
            {
                "op_name": "Resize",
                "attributes": [{"name": "mode", "s": (b"nearest"), "type": 3}],
            },
        ]
        ops = {9: ops_9, 10: ops_10}
        x = torch.randn(20, 16, 50)
        check_onnx_opsets_operator(MyDynamicModel(), x, ops, opset_versions=[9, 10])

    def test_affine_grid(self):
        class MyModule(Module):
            def __init__(self, align_corners):
                super().__init__()
                self.align_corners = align_corners

            def forward(self, theta, size):
                return torch.nn.functional.affine_grid(
                    theta, size, align_corners=self.align_corners
                )

        opset_version = 20
        ops_2d = {
            opset_version: [
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Concat"},
                {"op_name": "AffineGrid"},
            ]
        }

        ops_3d = {
            opset_version: [
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Constant"},
                {"op_name": "Unsqueeze"},
                {"op_name": "Concat"},
                {"op_name": "AffineGrid"},
            ]
        }
        # 2D affine
        theta_2d = torch.empty(1, 2, 3, dtype=torch.double)
        size_2d = torch.Size([1, 1, 2, 2])
        # 3D affine
        theta_3d = torch.empty(1, 3, 4, dtype=torch.double)
        size_3d = torch.Size([1, 1, 2, 2, 2])

        for inputs, align_corners in itertools.product(
            ((theta_2d, size_2d, ops_2d), (theta_3d, size_3d, ops_3d)),
            (True, False),
        ):
            theta, size, ops = inputs
            args = (
                theta,
                size,
            )
            check_onnx_opsets_operator(
                MyModule(align_corners=align_corners),
                args,
                ops,
                opset_versions=[opset_version],
                training=torch.onnx.TrainingMode.TRAINING,
            )
            check_onnx_opsets_operator(
                MyModule(align_corners=align_corners),
                args,
                ops,
                opset_versions=[opset_version],
                training=torch.onnx.TrainingMode.EVAL,
            )

    def test_grid_sample(self):
        class MyModule(torch.nn.Module):
            def __init__(self, mode, padding_mode, align_corners):
                super().__init__()
                self.mode = mode
                self.padding_mode = padding_mode
                self.align_corners = align_corners

            def forward(self, x, grid):
                return torch.nn.functional.grid_sample(
                    x,
                    grid,
                    mode=self.mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners,
                )

        for mode, padding_mode, align_corners, opset_version in itertools.product(
            ("bilinear", "nearest", "bicubic"),
            ("zeros", "border", "reflection"),
            (True, False),
            (16, 20),
        ):

            def test_eval_and_training(
                ops, opset_version, mode, padding_mode, align_corners, x_shape, grid
            ):
                args = (
                    torch.randn(*x_shape),  # x
                    torch.randn(grid),  # grid,
                )
                check_onnx_opsets_operator(
                    MyModule(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    ),
                    args,
                    ops,
                    opset_versions=[opset_version],
                    training=torch.onnx.TrainingMode.TRAINING,
                )
                check_onnx_opsets_operator(
                    MyModule(
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=align_corners,
                    ),
                    args,
                    ops,
                    opset_versions=[opset_version],
                    training=torch.onnx.TrainingMode.EVAL,
                )

            ops = {opset_version: [{"op_name": "GridSample"}]}
            # mode = convert_grid_sample_mode(mode) if opset_version == 20 else mode
            n, c, d_in, h_in, w_in, d_out, h_out, w_out = 1, 1, 2, 3, 2, 3, 2, 4
            test_eval_and_training(
                ops,
                opset_version,
                mode,
                padding_mode,
                align_corners,
                (n, c, h_in, w_in),
                (n, h_out, w_out, 2),
            )
            if opset_version == 20 and mode != "bicubic":
                test_eval_and_training(
                    ops,
                    opset_version,
                    mode,
                    padding_mode,
                    align_corners,
                    (n, c, d_in, h_in, w_in),
                    (n, d_out, h_out, w_out, 3),
                )

    def test_flatten(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.flatten(x)

        module = MyModule()

        ops_0d = [{"op_name": "Constant"}, {"op_name": "Reshape"}]
        ops_1d = [{"op_name": "Identity"}]
        for shape in ([], [3]):
            x = torch.randn(shape)
            for opset_version in [9, 10]:
                ops = {opset_version: (ops_0d if len(shape) == 0 else ops_1d)}
                check_onnx_opsets_operator(
                    module, x, ops, opset_versions=[opset_version]
                )


if __name__ == "__main__":
    common_utils.run_tests()
