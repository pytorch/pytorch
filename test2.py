import torch
import torch.nn as nn
from torch.utils.debug_log import debug_log


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        y = self.relu(x)
        debug_log(x, "after_fc1")
        x = self.fc2(y)
        return x


def main():
    torch.manual_seed(42)
    model = ToyModel()
    x = torch.randn(2, 8, requires_grad=True)
    x1 = x.clone().detach().requires_grad_(True)
    x2 = x.clone().detach().requires_grad_(True)

    # Eager
    print("=== Eager ===")
    out = model(x)
    loss = out.sum()
    loss.backward()
    print()

    # Compiled
    print("=== Compiled (aot_eager) ===")
    torch._dynamo.reset()
    compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)
    out = compiled_model(x1)
    loss = out.sum()
    loss.backward()
    print()

    # aot_function
    print("=== aot_function ===")
    from torch._functorch.aot_autograd import aot_function
    from torch.nn.utils._named_member_accessor import NamedMemberAccessor

    def nop_compiler(gm, _):
        return gm

    param_names = [n for n, _ in model.named_parameters()]
    params = list(model.parameters())

    def fn(x, *params):
        accessor = NamedMemberAccessor(model)
        named = dict(zip(param_names, params))
        orig, _ = accessor.swap_tensors_dict(named)
        try:
            return model(x).sum()
        finally:
            accessor.swap_tensors_dict(orig)

    aot_fn = aot_function(fn, nop_compiler)
    loss = aot_fn(x2, *params)
    loss.backward()
    print()


if __name__ == "__main__":
    main()
