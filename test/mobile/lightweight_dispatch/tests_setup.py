import sys
import os
from typing import List

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


tests = [
    ModelWithDTypeDeviceLayoutPinMemory(),
    ModelWithTensorOptional()
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
