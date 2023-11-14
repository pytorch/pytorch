import torch
import torchvision

from torch.backends._coreml.preprocess import CompileSpec, CoreMLComputeUnit, TensorSpec


def mobilenetv2_spec():
    return {
        "forward": CompileSpec(
            inputs=(
                TensorSpec(
                    shape=[1, 3, 224, 224],
                ),
            ),
            outputs=(
                TensorSpec(
                    shape=[1, 1000],
                ),
            ),
            backend=CoreMLComputeUnit.CPU,
            allow_low_precision=True,
        ),
    }


def main():
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    model = torch.jit.trace(model, example)
    compile_spec = mobilenetv2_spec()
    mlmodel = torch._C._jit_to_backend("coreml", model, compile_spec)
    print(mlmodel._c._get_method("forward").graph)
    mlmodel._save_for_lite_interpreter("../models/model_coreml.ptl")
    torch.jit.save(mlmodel, "../models/model_coreml.pt")


if __name__ == "__main__":
    main()
