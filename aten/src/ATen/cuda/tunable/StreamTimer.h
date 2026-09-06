// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/cuda/tunable
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
    StreamTimer(const StreamTimer&) = delete;
    StreamTimer& operator=(const StreamTimer&) = delete;
    StreamTimer(StreamTimer&&) = delete;
    StreamTimer& operator=(StreamTimer&&) = delete;
    ~StreamTimer() override;

    void Start() override;

    void End() override;

    float Duration() override;

  private:
    cudaEvent_t start_{};
    cudaEvent_t end_{};
};

class StreamTimerNoSync : public ITimer {
  public:
    StreamTimerNoSync();
    StreamTimerNoSync(const StreamTimerNoSync&) = delete;
    StreamTimerNoSync& operator=(const StreamTimerNoSync&) = delete;
    StreamTimerNoSync(StreamTimerNoSync&&) = delete;
    StreamTimerNoSync& operator=(StreamTimerNoSync&&) = delete;
    ~StreamTimerNoSync() override;

    void Start() override;

    void End() override;

    float Duration() override;

  private:
    cudaEvent_t start_{};
    cudaEvent_t end_{};
};

} // namespace at::cuda::tunable
