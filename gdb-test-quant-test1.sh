#!/bin/bash
gdb --args   ./build/bin/test_tensorexpr --gtest_filter="Kernel.QuantConv2dDequantInt8*"
