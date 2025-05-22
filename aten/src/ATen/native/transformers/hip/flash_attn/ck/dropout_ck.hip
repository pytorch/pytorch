/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>
#include <torch/types.h>
#include <ATen/hip/PhiloxUtils.cuh>

#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>

#include <dropout_ck.h>

#include <iostream>
namespace {
/**
 * generate a tensor with random uniform values. only used for testing, not much
 * attention is paid to performance
 */
at::Tensor rand_uniform_int(
    double dropout_prob,
    const at::Tensor& out_pattern) // [Batches, num_head, query_len, key_len]
{
  std::cout << "ANDY LUGO: " << dropout_prob << std::endl;
    
  return out_pattern;
  /*
  int B = out_pattern.size(0);
  int num_heads = out_pattern.size(1);
  int M = out_pattern.size(2);
  int N = out_pattern.size(3);

  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

  at::CUDAGeneratorImpl* gen =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(
          c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  at::PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs =
        gen->philox_cuda_state((B + 3) * (num_heads + 1) * (M + 1) * (N + 1));
  }

  const auto seeds = at::cuda::philox::unpack(rng_engine_inputs);

  int64_t philox_seed = std::get<0>(seeds);
  int64_t philox_offset = std::get<1>(seeds);

  at::Tensor randvals;

  randvals = at::empty(
      {B, num_heads, M, N}, out_pattern.options().dtype(at::ScalarType::Byte));

  {
    // only work for batched mode
    using FmhaRandUniformKernel_ = FmhaRandUniformKernel<uint8_t, false>;

    const auto kargs = FmhaRandUniformKernel_::MakeKargs(
        randvals.data_ptr(),
        M,
        N,
        num_heads,
        B,
        static_cast<int>(randvals.stride(2)),
        static_cast<int>(randvals.stride(3)),
        static_cast<int>(randvals.stride(1)),
        static_cast<int>(randvals.stride(0)),
        {philox_seed, philox_offset});

    dim3 kGridSize = FmhaRandUniformKernel_::GridSize(B, num_heads, M, N);
    constexpr dim3 kBlockSize = FmhaRandUniformKernel_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu =
        FmhaRandUniformKernel_::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockSize.x, kBlockPerCu>(
            FmhaRandUniformKernel_{}, kGridSize, kBlockSize, 0, kargs));
  }

  (void)hipStreamSynchronize(stream);

  return randvals;
  */
}


double andy_lugo(double in) {
  std::cout << "ANDY LUGO: " << in << std::endl;
  return -1.0;
}



} // namespace



TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA("xformers::_ck_rand_uniform(float p, Tensor out) -> Tensor"));
//  m.def(TORCH_SELECTIVE_SCHEMA("xformers::_andy_lugo(float in) -> float"));
}


TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("xformers::_ck_rand_uniform"), TORCH_FN(rand_uniform_int));
//  m.impl(TORCH_SELECTIVE_NAME("xformers::_andy_lugo"),TORCH_FN(andy_lugo));
}
