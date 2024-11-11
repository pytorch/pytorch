# mypy: allow-untyped-defs
import torch


class UserInputMutation(torch.nn.Module):
    """
    Directly mutate user input in forward
    """

    def forward(self, x):
        x.mul_(2)
        return x.cos()


example_args = (torch.randn(3, 2),)
tags = {"torch.mutation"}
model = UserInputMutation()
