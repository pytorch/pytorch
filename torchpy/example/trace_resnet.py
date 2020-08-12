import argparse
import torch
import torchvision

def traced_resnet():
    model = torchvision.models.resnet18()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    return traced_script_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_file", help="Where to save the model")
    args = parser.parse_args()

    model = traced_resnet()
    model.save(args.save_file)
