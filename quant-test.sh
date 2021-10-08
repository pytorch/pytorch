#!/bin/bash

./build/bin/test_tensorexpr --gtest_filter="Kernel.Quant"
./build/bin/test_tensorexpr --gtest_filter="Kernel.QuantDequant"
