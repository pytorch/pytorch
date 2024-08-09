import torch
from typing import Callable, Any


class Functional2Layer(torch.nn.Module):
    def __init__(
        self, func: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.func(input, *self.args, **self.kwargs)

    def extra_repr(self) -> str:
        func_name = (
            self.func.__name__ if hasattr(self.func, "__name__") else str(self.func)
        )
        args_repr = ", ".join(map(repr, self.args))
        kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"func={func_name}, args=({args_repr}), kwargs={{{kwargs_repr}}}"


if __name__ == "__main__":

    print("Permute Example")
    test_layer_permute = Functional2Layer(func=torch.permute, dims=(0, 2, 3, 1))
    input = torch.zeros((10, 11, 12, 13))
    output = test_layer_permute(input)
    print(input.shape)
    print(output.shape)
    print(test_layer_permute)

    print()
    print("Clamp Example")
    test_layer_clamp = Functional2Layer(func=torch.clamp, min=5, max=100)
    output = test_layer_permute(input)
    print(output[0, 0, 0, 0])
    print(test_layer_clamp)
