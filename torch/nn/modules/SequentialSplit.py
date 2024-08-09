import torch
from typing import Callable


class SequentialSplit(torch.nn.Module):
    """
    A PyTorch module that splits the processing path of a input tensor
    and processes it through multiple torch.nn.Sequential segments,
    then combines the outputs using a specified methods.

    This module allows for creating split paths within a `torch.nn.Sequential`
    model, making it possible to implement architectures with skip connections
    or parallel paths without abandoning the sequential model structure.

    Attributes:
        segments (torch.nn.Sequential[torch.nn.Sequential]): A list of sequential modules to
            process the input tensor.
        combine_func (Callable | None): A function to combine the outputs
            from the segments.
        dim (int | None): The dimension along which to concatenate
            the outputs if `combine_func` is `torch.cat`.

    Args:
        segments (torch.nn.Sequential[torch.nn.Sequential]): A torch.nn.Sequential
            with a list of sequential modules to process the input tensor.
        combine (str, optional): The method to combine the outputs.
            "cat" for concatenation (default), "sum" for a summation,
            or "func" to use a custom combine function.
        dim (int | None, optional): The dimension along which to
            concatenate the outputs if `combine` is "cat".
            Defaults to 1.
        combine_func (Callable | None, optional): A custom function
            to combine the outputs if `combine` is "func".
            Defaults to None.

    Example:
        A simple example for the `SequentialSplit` module with two sub-torch.nn.Sequential:

                                 ----- segment_a -----
            main_Sequential ----|                     |---- main_Sequential
                                 ----- segment_b -----

        segments = [segment_a, segment_b]
        y_split = SequentialSplit(segments)
        result = y_split(input_tensor)

    Methods:
        forward(input: torch.Tensor) -> torch.Tensor:
            Processes the input tensor through the segments and
            combines the results.
    """

    segments: torch.nn.Sequential
    combine_func: Callable
    dim: int | None

    def __init__(
        self,
        segments: torch.nn.Sequential,
        combine: str = "cat",  # "cat", "sum", "func",
        dim: int | None = 1,
        combine_func: Callable | None = None,
    ):
        super().__init__()
        self.segments = segments
        self.dim = dim

        self.combine = combine

        if combine.upper() == "CAT":
            self.combine_func = torch.cat
        elif combine.upper() == "SUM":
            self.combine_func = self.sum
            self.dim = None
        else:
            assert combine_func is not None
            self.combine_func = combine_func

    def sum(self, input: list[torch.Tensor]) -> torch.Tensor | None:

        if len(input) == 0:
            return None

        if len(input) == 1:
            return input[0]

        output: torch.Tensor = input[0]

        for i in range(1, len(input)):
            output = output + input[i]

        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        results: list[torch.Tensor] = []
        for segment in self.segments:
            results.append(segment(input))

        if self.dim is None:
            return self.combine_func(results)
        else:
            return self.combine_func(results, dim=self.dim)

    def extra_repr(self) -> str:
        return self.combine


if __name__ == "__main__":

    print("Example CAT")
    strain_a = torch.nn.Sequential(torch.nn.Identity())
    strain_b = torch.nn.Sequential(torch.nn.Identity())
    strain_c = torch.nn.Sequential(torch.nn.Identity())
    test_cat = SequentialSplit(
        torch.nn.Sequential(strain_a, strain_b, strain_c), combine="cat", dim=2
    )
    print(test_cat)
    input = torch.ones((10, 11, 12, 13))
    output = test_cat(input)
    print(input.shape)
    print(output.shape)
    print(input[0, 0, 0, 0])
    print(output[0, 0, 0, 0])
    print()

    print("Example SUM")
    strain_a = torch.nn.Sequential(torch.nn.Identity())
    strain_b = torch.nn.Sequential(torch.nn.Identity())
    strain_c = torch.nn.Sequential(torch.nn.Identity())
    test_sum = SequentialSplit(
        torch.nn.Sequential(strain_a, strain_b, strain_c), combine="sum", dim=2
    )
    print(test_sum)
    input = torch.ones((10, 11, 12, 13))
    output = test_sum(input)
    print(input.shape)
    print(output.shape)
    print(input[0, 0, 0, 0])
    print(output[0, 0, 0, 0])
    print()

    print("Example Labeling")
    strain_a = torch.nn.Sequential()
    strain_a.add_module("Label for first strain", torch.nn.Identity())
    strain_b = torch.nn.Sequential()
    strain_b.add_module("Label for second strain", torch.nn.Identity())
    strain_c = torch.nn.Sequential()
    strain_c.add_module("Label for third strain", torch.nn.Identity())
    test_label = SequentialSplit(torch.nn.Sequential(strain_a, strain_b, strain_c))
    print(test_label)
    print()

    print("Example Get Parameter")
    input = torch.ones((10, 11, 12, 13))
    strain_a = torch.nn.Sequential()
    strain_a.add_module("Identity", torch.nn.Identity())
    strain_b = torch.nn.Sequential()
    strain_b.add_module(
        "Conv2d",
        torch.nn.Conv2d(
            in_channels=input.shape[1],
            out_channels=input.shape[1],
            kernel_size=(1, 1),
        ),
    )
    test_parameter = SequentialSplit(torch.nn.Sequential(strain_a, strain_b))
    print(test_parameter)
    for name, param in test_parameter.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
