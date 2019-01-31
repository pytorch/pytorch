import sys
import os
import torch

testEvalModeForLoadedModule_module_path = 'dropout_model.pt'


def testEvalModeForLoadedModule_setup():
    class Model(torch.jit.ScriptModule):
        def __init__(self):
            super(Model, self).__init__()
            self.dropout = torch.nn.Dropout(0.1)

        def forward(self, x):
            x = self.dropout(x)
            return x

    model = Model()
    model = model.train()
    model.save(testEvalModeForLoadedModule_module_path)


def testEvalModeForLoadedModule_shutdown():
    if os.path.exists(testEvalModeForLoadedModule_module_path):
        os.remove(testEvalModeForLoadedModule_module_path)


def setup():
    testEvalModeForLoadedModule_setup()


def shutdown():
    testEvalModeForLoadedModule_shutdown()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "setup":
        setup()
    elif command == "shutdown":
        shutdown()
