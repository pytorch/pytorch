# Owner(s): ["module: unknown"]
import unittest
from dataclasses import dataclass
from typing import Any, Callable, cast, Tuple, Union

import torch
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


@dataclass
class ConvArgs:
    image_size: int
    num_classes: int


class SimpleCNN(nn.Module):
    def __init__(self, conv_args: ConvArgs):
        super().__init__()
        image_size = conv_args.image_size
        num_classes = conv_args.num_classes
        self.image_size = image_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.fc1_size = self._calculate_fc1_size()
        self.fc1 = nn.Linear(self.fc1_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def _calculate_fc1_size(self):
        size = self.image_size
        size = (size - 5 + 1) // 2  # conv1 and pool
        size = (size - 5 + 1) // 2  # conv2 and pool
        size = size - 3 + 1  # conv3
        size = (size - 3 + 1) // 2  # conv4 and pool
        return 512 * size * size

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = x.view(-1, self.fc1_size)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestRuntimeEstimator(TestCase):
    def _train_step(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        inp: torch.Tensor,
    ):
        out = model(inp)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _measure_actual_cuda_time(
        self,
        func: Callable,
        args: Tuple[Any, ...],
    ) -> float:
        num_iters = 5
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            func(*args)
        end_event.record()
        torch.cuda.synchronize()
        measured_time = start_event.elapsed_time(end_event) / num_iters
        return measured_time

    def _runtime_estimate(
        self,
        estimate_mode: str,
        func: Callable,
        args: Tuple[Any, ...],
    ) -> float:
        runtime_estimator = RuntimeEstimator()
        with runtime_estimator(estimate_mode_type=estimate_mode):
            func(*args)
        return runtime_estimator.total_runtime

    def _init_model_and_args(
        self,
        model_type: str,
        model_args: Union[ConvArgs, ModelArgs],
        bsz: int,
    ) -> Tuple[nn.Module, optim.Optimizer, torch.Tensor]:
        if model_type == "Transformer":
            model_args = cast(ModelArgs, model_args)
            model = Transformer(model_args)
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            inp = torch.randint(0, model_args.vocab_size, (bsz, model_args.max_seq_len))
        elif model_type == "CNN":
            model_args = cast(ConvArgs, model_args)
            model = SimpleCNN(model_args)
            optimizer = optim.SGD(model.parameters(), lr=1e-2)
            inp = torch.randn(bsz, 3, model_args.image_size, model_args.image_size)
        else:
            raise NotImplementedError("Only Transformer and CNN is supported")
        return (model, optimizer, inp)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_transformer_runtime(
        self,
    ):
        dev = torch.cuda.current_device()
        vocab_size = 8192
        bsz, seq_len = 32, 1024
        model_args = ModelArgs(
            n_layers=2,
            n_heads=16,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dropout_p=0.1,
        )
        with torch.device(dev):
            args = self._init_model_and_args("Transformer", model_args, bsz)
            actual_runtime = self._measure_actual_cuda_time(self._train_step, args)

            with FakeTensorMode():
                fake_args = self._init_model_and_args("Transformer", model_args, bsz)
                estimated_runtime = self._runtime_estimate(
                    "operator-level-cost-model", self._train_step, fake_args
                )
        accuracy = actual_runtime / estimated_runtime
        print(
            f"Actual: {actual_runtime} Estimated: {estimated_runtime} Accuracy: {accuracy}"
        )
        self.assertAlmostEqual(accuracy, 1.0, delta=0.15)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_conv_model_runtime(
        self,
    ):
        dev = torch.cuda.current_device()
        num_classes = 10
        bsz, img_sz = 32, 224
        model_args = ConvArgs(img_sz, num_classes)
        with torch.device(dev):
            args = self._init_model_and_args("CNN", model_args, bsz)
            actual_runtime = self._measure_actual_cuda_time(self._train_step, args)

            with FakeTensorMode():
                fake_args = self._init_model_and_args("CNN", model_args, bsz)
                estimated_runtime = self._runtime_estimate(
                    "operator-level-cost-model", self._train_step, fake_args
                )
        accuracy = actual_runtime / estimated_runtime
        print(
            f"Actual: {actual_runtime} Estimated: {estimated_runtime} Accuracy: {accuracy}"
        )
        self.assertAlmostEqual(accuracy, 1.0, delta=0.15)


if __name__ == "__main__":
    run_tests()
