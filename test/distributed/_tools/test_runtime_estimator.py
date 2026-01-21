# Owner(s): ["oncall: distributed"]
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, Union

import torch
from torch import nn, optim
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1_size = self._calculate_fc1_size()
        self.fc1 = nn.Linear(self.fc1_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

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
        args: tuple[Any, ...],
    ) -> float:
        warmup_iters, actual_iters = 2, 5
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup_iters):
            func(*args)
        start_event.record()
        for _ in range(actual_iters):
            func(*args)
        end_event.record()
        torch.cuda.synchronize()
        measured_time = start_event.elapsed_time(end_event) / actual_iters
        return measured_time

    def _runtime_estimate(
        self,
        estimate_mode: str,
        func: Callable,
        args: tuple[Any, ...],
    ) -> float:
        # Optimizer init step
        func(*args)
        runtime_estimator = RuntimeEstimator()
        with runtime_estimator(estimate_mode_type=estimate_mode):
            func(*args)
        return runtime_estimator.total_runtime

    def _init_model_and_args(
        self,
        model_type: str,
        model_args: Union[ConvArgs, ModelArgs],
        bsz: int,
    ) -> tuple[nn.Module, optim.Optimizer, torch.Tensor]:
        dev = torch.cuda.current_device()
        if model_type == "Transformer":
            model_args = cast(ModelArgs, model_args)
            with torch.device(dev):
                model = Transformer(model_args)
            optimizer = optim.Adam(model.parameters(), lr=1e-2, foreach=True)
            inp = torch.randint(
                0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev
            )
        elif model_type == "CNN":
            model_args = cast(ConvArgs, model_args)
            with torch.device(dev):
                model = SimpleCNN(model_args)
            optimizer = optim.SGD(model.parameters(), lr=1e-2, foreach=True)
            inp = torch.randn(
                bsz, 3, model_args.image_size, model_args.image_size, device=dev
            )
        else:
            raise NotImplementedError("Only Transformer and CNN is supported")
        return (model, optimizer, inp)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_transformer_runtime(
        self,
    ):
        """Runs a basic GPT-2 model"""
        vocab_size = 8192
        bsz, seq_len = 8, 1024
        model_args = ModelArgs(
            n_layers=4,
            n_heads=12,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dim=768,
            dropout_p=0.1,
        )

        args = self._init_model_and_args("Transformer", model_args, bsz)
        actual_runtime = self._measure_actual_cuda_time(self._train_step, args)
        with FakeTensorMode():
            fake_args = self._init_model_and_args("Transformer", model_args, bsz)
            benchmark_estimate = self._runtime_estimate(
                "operator-level-benchmark", self._train_step, fake_args
            )
            roofline_estimate = self._runtime_estimate(
                "operator-level-cost-model", self._train_step, fake_args
            )
        benchmark_accuracy = actual_runtime / benchmark_estimate
        roofline_accuracy = actual_runtime / roofline_estimate
        print(
            f"Actual: {actual_runtime} Benchmark Estimate: {benchmark_estimate} Accuracy: {benchmark_accuracy}"
            f"\n Actual: {actual_runtime} Roofline Estimatee: {roofline_estimate} Accuracy: {roofline_accuracy}"
        )
        # No accuracy check for benchmark in CI as it is highly variable
        # self.assertAlmostEqual(benchmark_accuracy, 1.0, delta=0.2)
        # self.assertAlmostEqual(roofline_accuracy, 1.0, delta=0.3)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_conv_model_runtime(
        self,
    ):
        """Runs a simple CNN model"""
        num_classes = 100
        bsz, img_sz = 256, 128
        model_args = ConvArgs(img_sz, num_classes)
        args = self._init_model_and_args("CNN", model_args, bsz)
        actual_runtime = self._measure_actual_cuda_time(self._train_step, args)
        with FakeTensorMode():
            fake_args = self._init_model_and_args("CNN", model_args, bsz)
            benchmark_estimate = self._runtime_estimate(
                "operator-level-benchmark", self._train_step, fake_args
            )
            roofline_estimate = self._runtime_estimate(
                "operator-level-cost-model", self._train_step, fake_args
            )
        benchmark_accuracy = actual_runtime / benchmark_estimate
        roofline_accuracy = actual_runtime / roofline_estimate
        print(
            f"Actual: {actual_runtime} Benchmark Estimate: {benchmark_estimate} Accuracy: {benchmark_accuracy}\n"
            f"Actual: {actual_runtime} Roofline Estimatee: {roofline_estimate} Accuracy: {roofline_accuracy}"
        )
        # No accuracy check for benchmark in CI as it is highly variable
        # self.assertAlmostEqual(benchmark_accuracy, 1.0, delta=0.2)
        # self.assertAlmostEqual(roofline_accuracy, 1.0, delta=0.4)


if __name__ == "__main__":
    run_tests()
