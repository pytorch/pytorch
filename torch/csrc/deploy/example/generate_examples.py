"""
Generate the example files that torchpy_test uses.
"""
from pathlib import Path
import torch
import torchvision

from torch.package import PackageExporter


def traced_resnet():
    model = torchvision.models.resnet18()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    return traced_script_module


class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        output = self.weight + input
        return output


if __name__ == "__main__":
    p = Path(__file__).parent / "generated"
    p.mkdir(exist_ok=True)
    model = traced_resnet()
    model.save(str(p / "resnet.pt"))

    my_module = MyModule(10, 20)
    sm = torch.jit.script(my_module)
    sm.save(str(p / "simple.pt"))

    resnet = torchvision.models.resnet18()
    with PackageExporter(str(p / "resnet.zip")) as e:
        # put the pickled resnet in the package, by default
        # this will also save all the code files references by
        # the objects in the pickle
        e.save_pickle('model', 'model.pkl', resnet)
