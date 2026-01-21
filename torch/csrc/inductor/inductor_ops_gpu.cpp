#include <c10/core/SymInt.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/library.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/from_blob.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/zeros.h>
#endif

#if defined(USE_CUDA) || defined(USE_ROCM)
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

namespace torch::inductor {
using namespace at;

#if defined(USE_CUDA) || defined(USE_ROCM)

// Reserves RNG state for Inductor with CUDA Graph support.
//
// This function allows Inductor to reserve a specific amount of RNG offset
// (increment) for a kernel. It is designed to be safe for CUDA Graph capture
// by explicitly handling the internal generator state via public APIs.
//
// Behavior:
// - Graph Mode: Advances the generator state and returns pointers (wrapped as
// tensors) to the extragraph state. These tensors effectively point to the
// GPU memory that will be updated by `replay_prologue`.
// - Eager Mode: Advances the generator state and returns concrete values
// wrapped in 1D tensors to maintain shape consistency.
//
// -param gen The CUDA generator to use.
// -param increment The number of RNG values to reserve.
// -return A tuple of (Seed Tensor, Offset Tensor, Intragraph Offset CPU
// Tensor).
static std::tuple<Tensor, Tensor, Tensor> inductor_reserve_rng_state_impl(
    const Generator& generator,
    c10::SymInt increment) {
  auto* gen_impl = at::check_generator<at::CUDAGeneratorImpl>(generator);

  const auto dev_opts =
      at::TensorOptions().dtype(at::kLong).device(generator.device());
  const auto cpu_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);

  int64_t inc = increment.expect_int();
  const at::PhiloxCudaState st =
      gen_impl->philox_cuda_state(static_cast<uint64_t>(inc));

  if (st.captured_) {
    auto seed_t = at::from_blob(
        static_cast<void*>(st.seed_.ptr), {1}, [](void*) {}, dev_opts);
    auto off_t = at::from_blob(
        static_cast<void*>(st.offset_.ptr), {1}, [](void*) {}, dev_opts);
    auto intra_t =
        at::scalar_tensor(static_cast<int64_t>(st.offset_intragraph_), cpu_opts)
            .unsqueeze(0);
    return {seed_t, off_t, intra_t};
  }

  auto seed_t = at::scalar_tensor(static_cast<int64_t>(st.seed_.val), dev_opts)
                    .unsqueeze(0);
  auto off_t = at::scalar_tensor(static_cast<int64_t>(st.offset_.val), dev_opts)
                   .unsqueeze(0);
  auto intra_t = at::zeros({1}, cpu_opts);
  return {seed_t, off_t, intra_t};
}

TORCH_LIBRARY_IMPL(inductor_prims, BackendSelect, m) {
  m.impl(
      "inductor_reserve_rng_state", TORCH_FN(inductor_reserve_rng_state_impl));
}

TORCH_LIBRARY_IMPL(inductor_prims, CUDA, m) {
  m.impl(
      "inductor_reserve_rng_state", TORCH_FN(inductor_reserve_rng_state_impl));
}

TORCH_LIBRARY_IMPL(inductor_prims, HIP, m) {
  m.impl(
      "inductor_reserve_rng_state", TORCH_FN(inductor_reserve_rng_state_impl));
}

#endif

} // namespace torch::inductor
