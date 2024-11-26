#!/usr/bin/env python3
# Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

# mypy: allow-untyped-defs

from parameterized import parameterized

import torch
from torch._dynamo.test_case import TestCase
from torch.testing._internal.common_utils import run_tests


class TestPackLinear(TestCase):
    @parameterized.expand(
        [
            (None, 128, 256, 512, True),
            (None, 128, 256, 512, False),
            (8, 128, 256, 512, True),
            (8, 128, 256, 512, False),
        ]
    )
    def test_one_linear(self, batch_size, M, N, K, bias):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(N, K, dtype=torch.float32, bias=bias)

            def forward(self, x):
                return self.linear(x)

        model = Foo()
        packed_model = torch.nn.utils.pack_linear.pack_linear_weights(model)
        inputs = (
            torch.randint(0, 128, (batch_size, M, N), dtype=torch.float32)
            if batch_size
            else torch.randint(0, 128, (M, N), dtype=torch.float32)
        )

        self.assertEqual(model(inputs), packed_model(inputs))

    @parameterized.expand(
        [
            (None, 16, 128, 256, 512, True),
            (None, 16, 128, 256, 512, False),
            (8, 16, 128, 256, 512, True),
            (8, 16, 128, 256, 512, False),
        ]
    )
    def test_multiple_layers(self, batch_size, channels, M, N, K, bias):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(channels, 32, 5, padding=2)
                self.linear = torch.nn.Linear(N, N, dtype=torch.float32, bias=bias)
                self.linear2 = torch.nn.Linear(N, K, dtype=torch.float32, bias=bias)

            def forward(self, x):
                x = self.conv(x)
                x = self.linear(x)
                return self.linear2(x)

        model = Foo()
        packed_model = torch.nn.utils.pack_linear.pack_linear_weights(model)
        inputs = (
            torch.randint(0, 128, (batch_size, channels, M, N), dtype=torch.float32)
            if batch_size
            else torch.randint(0, 128, (channels, M, N), dtype=torch.float32)
        )

        self.assertEqual(model(inputs), packed_model(inputs))

    @parameterized.expand(
        [
            (None, 16, 128, 256, 512, True),
            (None, 16, 128, 256, 512, False),
            (8, 16, 128, 256, 512, True),
            (8, 16, 128, 256, 512, False),
        ]
    )
    def test_multiple_modules(self, batch_size, channels, M, N, K, bias):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(N, N, bias=bias)

            def forward(self, x):
                return self.linear(x)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(channels, 32, 5, padding=2)
                self.linear = torch.nn.Linear(N, N, dtype=torch.float32, bias=bias)
                self.bar = Bar()
                self.linear2 = torch.nn.Linear(N, K, dtype=torch.float32, bias=bias)

            def forward(self, x):
                x = self.conv(x)
                x = self.linear(x)
                x = self.bar(x)
                return self.linear2(x)

        model = Foo()
        packed_model = torch.nn.utils.pack_linear.pack_linear_weights(model)
        inputs = (
            torch.randint(0, 128, (batch_size, channels, M, N), dtype=torch.float32)
            if batch_size
            else torch.randint(0, 128, (channels, M, N), dtype=torch.float32)
        )

        self.assertEqual(model(inputs), packed_model(inputs))

    @parameterized.expand(
        [
            (None, 16, 128, 256, 512, True),
            (None, 16, 128, 256, 512, False),
            (8, 16, 128, 256, 512, True),
            (8, 16, 128, 256, 512, False),
        ]
    )
    def test_autocast(self, batch_size, channels, M, N, K, bias):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(N, N, bias=bias)

            def forward(self, x):
                return self.linear(x)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(channels, 32, 5, padding=2)
                self.linear = torch.nn.Linear(N, N, dtype=torch.float32, bias=bias)
                self.bar = Bar()
                self.linear2 = torch.nn.Linear(N, K, dtype=torch.float32, bias=bias)

            def forward(self, x):
                x = self.conv(x)
                x = self.linear(x)
                x = self.bar(x)
                return self.linear2(x)

        model = Foo().to(device=torch.device("cpu"))
        model.eval()
        inputs = (
            torch.randint(0, 128, (batch_size, channels, M, N), dtype=torch.float32)
            if batch_size
            else torch.randint(0, 128, (channels, M, N), dtype=torch.float32)
        )

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            packed_model = torch.nn.utils.pack_linear.pack_linear_weights(model)
            self.assertEqual(model(inputs), packed_model(inputs))

    @parameterized.expand(
        [
            (16, 128, 256, 512, True),
            (16, 128, 256, 512, False),
        ]
    )
    def test_compile(self, channels, M, N, K, bias):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(N, N, bias=bias)

            def forward(self, x):
                return self.linear(x)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(channels, 32, 5, padding=2)
                self.linear = torch.nn.Linear(N, N, dtype=torch.float32, bias=bias)
                self.bar = Bar()
                self.linear2 = torch.nn.Linear(N, K, dtype=torch.float32, bias=bias)

            def forward(self, x):
                x = self.conv(x)
                x = self.linear(x)
                x = self.bar(x)
                return self.linear2(x)

        model = torch.compile(Foo())
        packed_model = torch.nn.utils.pack_linear.pack_linear_weights(model)
        inputs = torch.randint(0, 128, (channels, M, N), dtype=torch.float32)

        self.assertEqual(model(inputs), packed_model(inputs))

    @parameterized.expand(
        [
            (None, 16, 128, 256, 512, True),
            (None, 16, 128, 256, 512, False),
            (8, 16, 128, 256, 512, True),
            (8, 16, 128, 256, 512, False),
        ]
    )
    def test_dynamic_quant(self, batch_size, channels, M, N, K, bias):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(N, N, bias=bias)

            def forward(self, x):
                return self.linear(x)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(channels, 32, 5, padding=2)
                self.linear = torch.nn.Linear(N, N, dtype=torch.float32, bias=bias)
                self.bar = Bar()
                self.linear2 = torch.nn.Linear(N, K, dtype=torch.float32, bias=bias)

            def forward(self, x):
                x = self.conv(x)
                x = self.linear(x)
                x = self.bar(x)
                return self.linear2(x)

        model = torch.ao.quantization.quantize_dynamic(
            Foo(), {torch.nn.Linear}, dtype=torch.qint8
        )
        packed_model = torch.nn.utils.pack_linear.pack_linear_weights(model)
        inputs = (
            torch.randint(0, 128, (batch_size, channels, M, N), dtype=torch.float32)
            if batch_size
            else torch.randint(0, 128, (channels, M, N), dtype=torch.float32)
        )

        self.assertEqual(model(inputs), packed_model(inputs))


if __name__ == "__main__":
    run_tests()
