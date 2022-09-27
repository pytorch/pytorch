import argparse
import torch

def run_model(level):
    m = torch.nn.Linear(20, 30)
    input = torch.randn(128, 20)
    with torch.backends.mkl.verbose(level):
        m(input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", default=0, type=int)
    args = parser.parse_args()
    try:
        run_model(args.verbose_level)
    except Exception as e:
        print(e)
