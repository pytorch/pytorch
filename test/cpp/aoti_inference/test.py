import torch
import torch._inductor.config
from torch._export import aot_compile
from torch.export import Dim


torch.manual_seed(1337)


class Net(torch.nn.Module):
    def __init__(self, device, size=4):
        super().__init__()
        self.w_pre = torch.randn(size, size, device=device)
        self.w_add = torch.randn(size, size, device=device)

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
large_data = {}
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
            # patch freezing off to make constant names more predictable
            with torch.no_grad(), torch._inductor.config.patch(freezing=False):
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
                # Also store a .pt2 file using the aoti_compile_and_package API
                pt2_package_path = torch._inductor.aoti_compile_and_package(
                    torch.export.export(
                        model,
                        (x,),
                        dynamic_shapes=dynamic_shapes,
                    ),
                    inductor_configs={
                        "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                    },
                )

            suffix = f"{device}"
            if use_runtime_constant_folding:
                suffix += "_use_runtime_constant_folding"
            data.update(
                {
                    f"model_so_path_{suffix}": model_so_path,
                    f"pt2_package_path_{suffix}": pt2_package_path,
                    f"inputs_{suffix}": [x],
                    f"outputs_{suffix}": [ref_output],
                    f"w_pre_{suffix}": model.w_pre,
                    f"w_add_{suffix}": model.w_add,
                }
            )


def generate_basic_tests_consts_cpp():
    backup_consts_asm_cfg: bool = (
        torch._inductor.config.aot_inductor.use_consts_asm_build
    )
    torch._inductor.config.aot_inductor.use_consts_asm_build = False

    # Test consts cpp build again.
    generate_basic_tests()

    torch._inductor.config.aot_inductor.use_consts_asm_build = backup_consts_asm_cfg


def generate_large_tests():
    device = "cuda"
    model = Net(device, size=4096).to(device=device)
    x = torch.randn((4096, 4096), device=device)
    with torch.no_grad():
        ref_output = model(x)

    torch._dynamo.reset()
    for use_runtime_constant_folding in [True, False]:
        # patch freezing off to make constant names more predictable
        with torch.no_grad(), torch._inductor.config.patch(freezing=False):
            model_so_path = aot_compile(
                model,
                (x,),
                options={
                    "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                },
            )
            # Also store a .pt2 file using the aoti_compile_and_package API
            pt2_package_path = torch._inductor.aoti_compile_and_package(
                torch.export.export(
                    model,
                    (x,),
                ),
                inductor_configs={
                    "aot_inductor.use_runtime_constant_folding": use_runtime_constant_folding
                },
            )

        suffix = "_use_runtime_constant_folding" if use_runtime_constant_folding else ""
        large_data.update(
            {  # noqa: F541
                f"model_so_path{suffix}": model_so_path,
                f"pt2_package_path{suffix}": pt2_package_path,
                "inputs": [x],
                "outputs": [ref_output],
                "w_pre": model.w_pre,
                "w_add": model.w_add,
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
    # patch freezing off to make constant names more predictable
    with torch.no_grad(), torch._inductor.config.patch(freezing=False):
        model_so_path = aot_compile(model, (x, y))
        # Also store a .pt2 file using the aoti_compile_and_package API
        pt2_package_path = torch._inductor.aoti_compile_and_package(
            torch.export.export(model, (x, y))
        )

    data_with_tensor_constants.update(
        {
            "model_so_path": model_so_path,
            "pt2_package_path": pt2_package_path,
            "inputs": [x, y],
            "outputs": [ref_output],
            "w": model.w,
        }
    )


generate_basic_tests()
generate_basic_tests_consts_cpp()
generate_large_tests()
generate_test_with_additional_tensors()


# Use this to communicate tensors to the cpp code
class Serializer(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for key in data:
            setattr(self, key, data[key])


torch.jit.script(Serializer(data)).save("data.pt")
torch.jit.script(Serializer(large_data)).save("large_data.pt")
torch.jit.script(Serializer(data_with_tensor_constants)).save(
    "data_with_tensor_constants.pt"
)
