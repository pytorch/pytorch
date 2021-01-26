import argparse
import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        output = self.weight + input
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_file", help="Where to save the model")
    args = parser.parse_args()

    my_module = MyModule(10, 20)
    sm = torch.jit.script(my_module)
    sm.save(args.save_file)
