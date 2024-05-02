// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <cuda_runtime.h>

#include <ATen/cuda/tunable/Tunable.h>

namespace at::cuda::tunable {

class StreamTimer : public ITimer {
  public:
    StreamTimer();
    virtual ~StreamTimer();

    void Start() override;

    void End() override;

    float Duration() override;

  private:
    cudaEvent_t start_;
    cudaEvent_t end_;
};

} // namespace at::cuda::tunable
