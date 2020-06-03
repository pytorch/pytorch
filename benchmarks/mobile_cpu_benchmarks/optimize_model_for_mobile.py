import argparse
import torch
from torch.utils.mobile_optimizer import *
import os

def main():
    parser = argparse.ArgumentParser(description='PyTorch mobile cpu benchmark')
    parser.add_argument("--model_name", type=str, required=True, help="Input model")
    parser.add_argument("--output_name", type=str, help="Optimized output model")
    args = parser.parse_args()

    traced_model = torch.jit.load(args.model_name)
    traced_model.eval()
    optimized_traced_model = optimize_for_mobile(traced_model)

    output_name = args.output_name
    if (args.output_name == "" or not args.output_name):
        output_name = os.path.splitext(args.model_name)[0] + "_mobile_optimized.pt"
    torch.jit.save(optimized_traced_model, output_name)

if __name__ == "__main__":
    main()
