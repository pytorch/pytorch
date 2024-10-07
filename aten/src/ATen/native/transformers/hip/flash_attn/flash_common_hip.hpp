/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/TensorIndexing.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPGraphsUtils.cuh>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/zeros.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>


#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/hip/HIPGeneratorImpl.h>
#endif

#include <ATen/native/transformers/hip/flash_attn/flash_api.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace flash {
inline __global__ void ParsePhiloxCudaState(at::PhiloxCudaState arg, uint64_t* rng_state)
{
    // Imitate from PyTorch
    // https://github.com/pytorch/pytorch/blob/8b61daaf7349e9102117e1aeefaa51666d887547/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L17
    if (arg.captured_) {
        rng_state[0] = static_cast<uint64_t>(*arg.seed_.ptr);
        rng_state[1] = static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_);
    } else {
        rng_state[0] = arg.seed_.val;
        rng_state[1] = arg.offset_.val;
    }
}


} // namespace flash
