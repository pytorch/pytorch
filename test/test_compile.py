# Owner(s): ["module: dynamo"]

import torch
import tempfile
import os

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def test_compiled_model_can_be_saved():
    model = ToyModel()

    assert model._compiled_call_impl is None
    model.compile()
    assert model._compiled_call_impl is not None

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(model, os.path.join(tmpdirname, "model.pt"))
        torch.load(os.path.join(tmpdirname, "model.pt"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        torch.save(model.state_dict(), os.path.join(tmpdirname, "model.pt"))
        loaded_model = ToyModel()
        loaded_model.load_state_dict(torch.load(os.path.join(tmpdirname, "model.pt")))
