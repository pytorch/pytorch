import argparse
import torch
import torchvision.models as models

def get_model(model_name):
    try:
        model_getter = getattr(models, model_name)
    except e:
        raise RuntimeError(model_name, " does not exist in torchvision.")
    return model_getter(pretrained=True)

def trace_model(model, input_size):
    sample_input = torch.rand(input_size)
    return torch.jit.trace(model, sample_input)

def main():
    parser = argparse.ArgumentParser(description='PyTorch mobile cpu benchmark')
    parser.add_argument("--model_name", type=str, required=True, help="Name of the input torchvision model, e.g. mobilenet_v2")
    parser.add_argument("--input_size", type=str, required=True, help="Comma separated list of integers specifying size of input.")
    parser.add_argument("--output_name", type=str, default="", help="Name of the output model")
    args = parser.parse_args()

    model = get_model(args.model_name)

    if (args.input_size == ""):
        raise RuntimeError("Input size must be specified.")
    sizes = args.input_size.split(",")
    input_sizes = []
    for s in sizes:
        input_sizes.append(int(s))

    traced_model = trace_model(model, input_sizes)

    output_name = args.output_name
    if (args.output_name == ""):
        output_name = args.model_name + "_traced.pt"
    torch.jit.save(traced_model, output_name)

if __name__ == "__main__":
    main()
