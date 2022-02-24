import sys
import os
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
            def forward(self):
                a = torch.ones(3, 4, dtype=torch.int64, layout=torch.strided, device="cpu", requires_grad=False)
                return a

        model = Model()

        # Script the model and save
        script_model = torch.jit.script(model)
        script_model._save_for_lite_interpreter(script_model, self.path)


tests = [
    ModelWithDTypeDeviceLayoutPinMemory()
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
