import torch
from torch._export import aot_compile
from torch.export import Dim


torch.manual_seed(1337)


class Net(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.w_pre = torch.randn(4, 4, device=device)
        self.w_add = torch.randn(4, 4, device=device)

    def forward(self, x):
        w_transpose = torch.transpose(self.w_pre, 0, 1)
        w_relu = torch.nn.functional.relu(w_transpose)
        w = w_relu + self.w_add
        return torch.matmul(x, w)


class NetWithTensorConstants(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.randn(30, 1, device="cuda")

    def forward(self, x, y):
        z = self.w * x * y
        return z[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]]


data = {}
data_with_tensor_constants = {}


# Basice AOTI model test generation.
def generate_basic_tests():
    for device in ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]:
        for use_runtime_constant_folding in [True, False]:
            if device == "cpu" and use_runtime_constant_folding:
                # We do not test runtime const folding for cpu mode.
                continue
            model = Net(device).to(device=device)
            x = torch.randn((4, 4), device=device)
            with torch.no_grad():
                ref_output = model(x)

            torch._dynamo.reset()
            with torch.no_grad():
                dim0_x = Dim("dim0_x", min=1, max=1024)
                dynamic_shapes = {"x": {0: dim0_x}}
                model_so_path = aot_compile(
                    model,
                    (x,),
                    dynamic_shapes=dynamic_shapes,
                    options={
                        "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                    },
                )

            suffix = f"{device}"
            if use_runtime_constant_folding:
                suffix += "_use_runtime_constant_folding"
            data.update(
                {
                    f"model_so_path_{suffix}": model_so_path,
                    f"inputs_{suffix}": [x],
                    f"outputs_{suffix}": [ref_output],
                    f"w_pre_{suffix}": model.w_pre,
                    f"w_add_{suffix}": model.w_add,
                }
            )


# AOTI model which will create additional tensors during autograd.
def generate_test_with_additional_tensors():
    if not torch.cuda.is_available():
        return

    model = NetWithTensorConstants()
    x = torch.randn((30, 1), device="cuda")
    y = torch.randn((30, 1), device="cuda")
    with torch.no_grad():
        ref_output = model(x, y)

    torch._dynamo.reset()
    with torch.no_grad():
        model_so_path = aot_compile(model, (x, y))

    data_with_tensor_constants.update(
        {
            "model_so_path": model_so_path,
            "inputs": [x, y],
            "outputs": [ref_output],
            "w": model.w,
        }
    )


generate_basic_tests()
generate_test_with_additional_tensors()


# Use this to communicate tensors to the cpp code
class Serializer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


torch.jit.script(Serializer(data)).save("data.pt")
torch.jit.script(Serializer(data_with_tensor_constants)).save(
    "data_with_tensor_constants.pt"
)
