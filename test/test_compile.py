import torch 
import time 

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

def test_compiled_model_can_be_saved():
    model = ToyModel()

    tic = time.time()
    model(torch.randn(1, 10))
    toc = time.time()
    first_inference_duration = toc - tic

    model.compile()
    tic = time.time()
    model(torch.randn(1, 10))
    toc = time.time()
    compilation_duration = toc - tic 

    assert compilation_duration > first_inference_duration

    # assert that this doesn't crash
    torch.save(model, "model.pt")
