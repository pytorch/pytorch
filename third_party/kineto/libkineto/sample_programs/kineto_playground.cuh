/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace kineto {

// Warms up CUDA before the tracing starts
void warmup(void);

// Basic usage of cudaMemcpy and cudaMemset
void basicMemcpyToDevice(void);

// Basic usage of cudaMemcpy from device to host
void basicMemcpyFromDevice(void);

// Your experimental code goes in here!
void playground(void);

// Run a simple elementwise kernel
void compute(void);

} // namespace kineto
