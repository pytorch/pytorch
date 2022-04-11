import os
import sys

import torch


class Setup(object):
    def setup(self):
        raise NotImplementedError()

    def shutdown(self):
        raise NotImplementedError()


class FileSetup(object):
    path = None

    def shutdown(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            pass


class ModelWithDTypeDeviceLayoutPinMemory(FileSetup):
    path = 'ones.ptl'

    def setup(self):
        class Model(torch.nn.Module):
            def forward(self, x: int):
                a = torch.ones(size=[3, x], dtype=torch.int64, layout=torch.strided, device="cpu", pin_memory=False)
                return a

        model = Model()

        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


class ModelWithTensorOptional(FileSetup):
    path = 'index.ptl'

    def setup(self):
        class Model(torch.nn.Module):
            def forward(self, index):
                a = torch.zeros(2, 2)
                a[0][1] = 1
                a[1][0] = 2
                a[1][1] = 3
                return a[index]

        model = Model()

        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


# gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
class ModelWithScalarList(FileSetup):
    path = 'gradient.ptl'

    def setup(self):

        class Model(torch.nn.Module):
            def forward(self, a: int):
                values = torch.tensor([4., 1., 1., 16.], )
                if a == 0:
                    return torch.gradient(values, spacing=torch.scalar_tensor(2., dtype=torch.float64))
                elif a == 1:
                    return torch.gradient(values, spacing=[torch.tensor(1.).item()])

        model = Model()

        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


# upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
class ModelWithFloatList(FileSetup):
    path = 'upsample.ptl'

    def setup(self):
        model = torch.nn.Upsample(scale_factor=(2.0,), mode="linear", align_corners=False, recompute_scale_factor=True)

        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


# index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
class ModelWithListOfOptionalTensors(FileSetup):
    path = 'index_Tensor.ptl'

    def setup(self):
        class Model(torch.nn.Module):
            def forward(self, index):
                values = torch.tensor([[4., 1., 1., 16.]])
                return values[torch.tensor(0), index]

        model = Model()
        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


# conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1,
# int groups=1) -> Tensor
class ModelWithArrayOfInt(FileSetup):
    path = 'conv2d.ptl'

    def setup(self):
        model = torch.nn.Conv2d(1, 2, (2, 2), stride=(1, 1), padding=(1, 1))
        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


# add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
# ones_like(Tensor self, *, ScalarType?, dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None,
# MemoryFormat? memory_format=None) -> Tensor
class ModelWithTensors(FileSetup):
    path = 'add_Tensor.ptl'

    def setup(self):
        class Model(torch.nn.Module):
            def forward(self, a):
                b = torch.ones_like(a)
                return a + b
        model = Model()
        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


class ModelWithStringOptional(FileSetup):
    path = 'divide_Tensor.ptl'

    def setup(self):
        class Model(torch.nn.Module):
            def forward(self, b):
                a = torch.tensor(3, dtype=torch.int64)
                out = torch.empty(size=[1], dtype=torch.float)
                torch.div(b, a, out=out)
                return [torch.div(b, a, rounding_mode='trunc'), out]
        model = Model()
        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


class ModelWithMultipleOps(FileSetup):
    path = 'multiple_ops.ptl'

    def setup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.ops = torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Flatten(),
                )

            def forward(self, x):
                x[1] = -2
                return self.ops(x)

        model = Model()
        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(self.path)


tests = [
    ModelWithDTypeDeviceLayoutPinMemory(),
    ModelWithTensorOptional(),
    ModelWithScalarList(),
    ModelWithFloatList(),
    ModelWithListOfOptionalTensors(),
    ModelWithArrayOfInt(),
    ModelWithTensors(),
    ModelWithStringOptional(),
    ModelWithMultipleOps(),
]


def setup():
    for test in tests:
        test.setup()


def shutdown():
    for test in tests:
        test.shutdown()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "setup":
        setup()
    elif command == "shutdown":
        shutdown()
