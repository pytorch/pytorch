# Owner(s): ["module: unknown"]

from typing import Dict, Any, Tuple
from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn

class ImplementedSparsifier(BaseSparsifier):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(defaults=kwargs)

    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs: Dict[str, Any]) -> None:
        module.parametrizations.weight[0].mask[0] = 0
        linear_state = self.state['linear1.weight']
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1


class MockSparseLinear(nn.Linear):
    """
    This class is a MockSparseLinear class to check convert functionality.
    It is the same as a normal Linear layer, except with a different type, as
    well as an additional from_dense method.
    """
    @classmethod
    def from_dense(cls, mod: nn.Linear) -> 'MockSparseLinear':
        """
        """
        linear = cls(mod.in_features,
                     mod.out_features)
        return linear


def rows_are_subset(subset_tensor: torch.Tensor, superset_tensor: torch.Tensor) -> bool:
    """
    Checks to see if all rows in subset tensor are present in the superset tensor
    """
    i = 0
    for row in subset_tensor:
        while i < len(superset_tensor):
            if not torch.equal(row, superset_tensor[i]):
                i += 1
            else:
                break
        else:
            return False
    return True


class SimpleLinear(nn.Module):
    r"""Model with only Linear layers without biases, some wrapped in a Sequential,
    some following the Sequential. Used to test basic pruned Linear-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=False),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 4, bias=False),
        )
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class LinearBias(nn.Module):
    r"""Model with only Linear layers, alternating layers with biases,
    wrapped in a Sequential. Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 3, bias=True),
            nn.Linear(3, 3, bias=True),
            nn.Linear(3, 10, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return x


class LinearActivation(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.Tanh(),
            nn.Linear(6, 4, bias=True),
        )
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(3, 10, bias=False)
        self.act2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x


class LinearActivationFunctional(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and functional
    activationals are called in between each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True),
        )
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.linear2 = nn.Linear(3, 8, bias=False)
        self.linear3 = nn.Linear(8, 10, bias=False)
        self.act1 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return x


class SimpleConv2d(nn.Module):
    r"""Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Used to test pruned Conv2d-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=False),
            nn.Conv2d(32, 64, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dBias(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some outside.
    Used to test pruned Conv2d-Bias-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
            nn.Conv2d(32, 32, 3, 1, bias=True),
            nn.Conv2d(32, 64, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=True)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dActivation(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some following.
    Activation function modules in between each Sequential layer, functional activations called
    in-between each outside layer.
    Used to test pruned Conv2d-Bias-Activation-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, bias=True),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, 1, bias=False),
            nn.ReLU(),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.relu(x)
        x = self.conv2d2(x)
        x = F.hardtanh(x)
        return x


class Conv2dPadBias(nn.Module):
    r"""Model with only Conv2d layers, all with bias and some with padding > 0,
    some in a Sequential and some following. Activation function modules in between each layer.
    Used to test that bias is propagated correctly in the special case of
    pruned Conv2d-Bias-(Activation)Conv2d fusion, when the second Conv2d layer has padding > 0."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, bias=True),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, padding=1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, padding=1, bias=True)
        self.act2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dPool(nn.Module):
    r"""Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function modules in between each layer, Pool2d modules in between each layer.
    Used to test pruned Conv2d-Pool2d-Conv2d fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(48, 52, kernel_size=3, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(52, 52, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.maxpool(x)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = F.relu(x)
        x = self.conv2d3(x)
        return x


class Conv2dPoolFlattenFunctional(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a functional Flatten followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(11, 13, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # test functional flatten
        x = self.fc(x)
        return x


class Conv2dPoolFlatten(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a Flatten module followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(44, 13, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class LSTMLinearModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.lstm(input)
        decoded = self.linear(output)
        return decoded, output


class LSTMLayerNormLinearModel(nn.Module):
    """Container module with an LSTM, a LayerNorm, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, state = self.lstm(x)
        x = self.norm(x)
        x = self.linear(x)
        return x, state
