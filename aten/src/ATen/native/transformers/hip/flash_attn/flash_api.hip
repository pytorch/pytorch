/******************************************************************************
 * Copyright (c) 2023, Advanced Micro Devices, Inc.
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <cstdint>
#include <tuple>

#include <ATen/ops/zeros.h>

#ifdef USE_FLASH_ATTENTION
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPGraphsUtils.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/pad.h>
#endif

#include <ATen/native/transformers/hip/flash_attn/flash_api.h>

#include <c10/util/Exception.h>
#include <c10/util/CallOnce.h>

// OORT headers
#include <oort/attn_fwd.h>
#include <oort/bwd_kernel_dk_dv.h>
#include <oort/bwd_kernel_dq.h>
#include <oort/bwd_preprocess.h>

namespace pytorch_flash {

namespace {

c10::once_flag fa_gcn_arch_override_flag;
const char* fa_override_arch = nullptr;

void init_fa_override_arch() {
  fa_override_arch = std::getenv("PYTORCH_DEBUG_FLASH_ATTENTION_GCN_ARCH_OVERRIDE");
  if (fa_override_arch) {
      TORCH_WARN("ROCM flash attention backend only loads value from PYTORCH_DEBUG_FLASH_ATTENTION_GCN_ARCH_OVERRIDE once. "
                 "Later changes to this environment variable with os.environ "
                 "(or other methods) will not affect this backend's behavior.");
  }
}

void check_gpu_arch() {
  auto dprops = at::cuda::getCurrentDeviceProperties();

  constexpr std::string_view mi200 = "gfx90a:sramecc+:xnack-";
  c10::call_once(fa_gcn_arch_override_flag, init_fa_override_arch);
  if (fa_override_arch) {
      TORCH_CHECK(mi200 == fa_override_arch,
                  "FlashAttention only supports MI200/MI250 GPUs (gfx90a:sramecc+:xnack-), current gcnArchName: " + std::string(dprops->gcnArchName) + " override as " + fa_override_arch);
  } else {
      TORCH_CHECK(mi200 == dprops->gcnArchName,
                  "FlashAttention only supports MI200/MI250 GPUs (gfx90a:sramecc+:xnack-), current gcnArchName: " + std::string(dprops->gcnArchName));
  }
}

}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        const int window_size_left,
        int window_size_right,
        const bool return_softmax,
        c10::optional<at::Generator> gen_) {
  check_gpu_arch();

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

  // FIXME: ROCM probably does not need this
  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size_og % 8 == 0, "head_size must be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  if (seqlen_q == 1) { is_causal = false; }  // causal=true is the same as causal=false in this case
  if (is_causal) { window_size_right = 0; }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

  at::Tensor q_padded, k_padded, v_padded;
  q_padded = q;
  k_padded = k;
  v_padded = v;

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
    if (head_size_og % 8 != 0) { out = at::empty_like(q_padded); }
  } else {
    out = at::empty_like(q_padded);
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::hip::HIPGuardMasqueradingAsCUDA device_guard{(char)q.get_device()};

  // We want to checkpoint and save the RNG state for backward if dropout
  // We get the default generator and return the seed and offset which will
  // be used in the backward function
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::Tensor seed_t, offset_t;

  if (p_dropout > 0.0)  {
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = batch_size * num_heads * 32;
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    at::PhiloxCudaState philox_state = gen->philox_cuda_state(counter_offset);
    if (at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None) {
      auto [seed, offset] = at::cuda::philox::unpack(philox_state);
      seed_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(seed)), at::dtype(at::kLong));
      offset_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(offset)), at::dtype(at::kLong));
    } else {
      seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
      offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    }
  } else {
    if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
      seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
      offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    } else {
      seed_t = at::empty({}, at::dtype(at::kLong));
      offset_t = at::empty({}, at::dtype(at::kLong));
    }

  }

  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  //reorder tensors and make contiguous
  at::Tensor q_t = q_padded.permute({0,2,1,3}).contiguous();
  at::Tensor k_t = k_padded.permute({0,2,1,3}).contiguous();
  at::Tensor v_t = v_padded.permute({0,2,1,3}).contiguous();
  at::Tensor output_t = out.permute({0,2,1,3}).contiguous();

  at::Tensor M = at::empty({batch_size, num_heads, seqlen_q}, at::dtype(at::kFloat).device(q.device())); // aka softmax_lse

  constexpr int BLOCK_M = 16;
  constexpr int BLOCK_N = 16;
  dim3  grid;
  grid.x = (q_t.sizes()[2] + BLOCK_M - 1) / BLOCK_M;
  grid.y = q_t.sizes()[0] * q_t.sizes()[1];
  grid.z = 1;
  dim3 block { 64 * 4, 1, 1 };  // compiled triton kernel intrinsic

  at::Tensor softmax_fa_t = at::empty({batch_size, num_heads, seqlen_q, seqlen_k},
                                      at::dtype(q.dtype()).device(q.device()));

  hipError_t err; // TODO: Error handling
#define CALL_FWD(FP, STAGE, BLOCK_M, BLOCK_DMODEL, BLOCK_N, pre_load_v, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX)               \
  do {                                                                                                                      \
    oort::attn_fwd<STAGE,BLOCK_M, BLOCK_DMODEL, BLOCK_N, pre_load_v, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX> fwd_opt;       \
    err   = fwd_opt(grid, block,                                                                                            \
                    (FP*)(q_t.data_ptr()), (FP*)(k_t.data_ptr()), (FP*)(v_t.data_ptr()),                                    \
                    softmax_scale, (float*)M.data_ptr(), (FP*)output_t.data_ptr(),                                          \
                    q_t.stride(0), q_t.stride(1), q_t.stride(2), q_t.stride(3),                                             \
                    k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),                                             \
                    v_t.stride(0), v_t.stride(1), v_t.stride(2), v_t.stride(3),                                             \
                    output_t.stride(0), output_t.stride(1), output_t.stride(2), output_t.stride(3),                         \
                    q_t.sizes()[0], q_t.sizes()[1], seqlen_q, seqlen_k, p_dropout,                                          \
                    *(uint64_t*)(seed_t.data_ptr()), *(uint32_t*)(offset_t.data_ptr()),                                     \
                    (FP*)(softmax_fa_t.data_ptr()),                                                                         \
                    stream);                                                                                                \
  } while(0)

  // TODO: Ugly but works
  constexpr int kFwdUseCausal = 3;
  constexpr int kFwdNoCausal  = 1;
  int d_head = q_t.sizes()[3];
  constexpr int BM = BLOCK_M;
  constexpr int BN = BLOCK_N;
  if (q_dtype == at::kHalf) {
    if (is_causal) {
      if (d_head == 16)
        CALL_FWD(__fp16,kFwdUseCausal,BM,16,BN,true,true,true);
      else if (d_head == 32)
        CALL_FWD(__fp16,kFwdUseCausal,BM,32,BN,true,true,true);
      else if (d_head == 64)
        CALL_FWD(__fp16,kFwdUseCausal,BM,64,BN,true,true,true);
      else if (d_head == 128)
        CALL_FWD(__fp16,kFwdUseCausal,BM,128,BN,true,true,true);
    } else {
      if (d_head == 16)
        CALL_FWD(__fp16,kFwdNoCausal,BM,16,BN,true,true,true);
      else if (d_head == 32)
        CALL_FWD(__fp16,kFwdNoCausal,BM,32,BN,true,true,true);
      else if (d_head == 64)
        CALL_FWD(__fp16,kFwdNoCausal,BM,64,BN,true,true,true);
      else if (d_head == 128)
        CALL_FWD(__fp16,kFwdNoCausal,BM,128,BN,true,true,true);
    }
  } else if (q_dtype == at::kBFloat16) {
    if (is_causal) {
      if (d_head == 16)
        CALL_FWD(__bf16,kFwdUseCausal,BM,16,BN,true,true,true);
      else if (d_head == 32)
        CALL_FWD(__bf16,kFwdUseCausal,BM,32,BN,true,true,true);
      else if (d_head == 64)
        CALL_FWD(__bf16,kFwdUseCausal,BM,64,BN,true,true,true);
      else if (d_head == 128)
        CALL_FWD(__bf16,kFwdUseCausal,BM,128,BN,true,true,true);
    } else {
      if (d_head == 16)
        CALL_FWD(__bf16,kFwdNoCausal,BM,16,BN,true,true,true);
      else if (d_head == 32)
        CALL_FWD(__bf16,kFwdNoCausal,BM,32,BN,true,true,true);
      else if (d_head == 64)
        CALL_FWD(__bf16,kFwdNoCausal,BM,64,BN,true,true,true);
      else if (d_head == 128)
        CALL_FWD(__bf16,kFwdNoCausal,BM,128,BN,true,true,true);
    }
  }

  //undo reorder tensors
  q_padded = q_t.permute({0,2,1,3}).contiguous();
  k_padded = k_t.permute({0,2,1,3}).contiguous();
  v_padded = v_t.permute({0,2,1,3}).contiguous();
  out = output_t.permute({0,2,1,3}).contiguous();

  return {out, q_padded, k_padded, v_padded, M, seed_t, offset_t, softmax_fa_t};
#undef CALL_FWD
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               const int window_size_left,
               int window_size_right,
               const bool return_softmax,
               c10::optional<at::Generator> gen_) {

  TORCH_CHECK(false, "mha_varlen_fwd not supported on ROCm");

  at::Tensor softmax_lse = at::empty({}, at::dtype(at::kFloat));
  at::Tensor p = at::empty({}, at::dtype(at::kFloat));
  at::Tensor offset_t = at::empty({}, at::dtype(at::kLong));
  at::Tensor seed_t = at::empty({}, at::dtype(at::kLong));
  at::Tensor out = at::empty({}, at::dtype(at::kFloat));

  return {out, q, k, v, softmax_lse, seed_t, offset_t, p};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        const int window_size_left,
        int window_size_right,
        const at::Tensor philox_seed,
        const at::Tensor philox_offset) {
  check_gpu_arch();

  bool is_dropout = p_dropout > 0.0;
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
  CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = dout.size(3);
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);

  if (is_causal){
    TORCH_CHECK((seqlen_q == seqlen_k), "For backwards kernel seqlen_q must equal seqlen_k for causal kernels");
  }

  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(head_size_og % 8 == 0, "head_size_og should be a multiple of 8, this is ensured by padding!");
  TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

  at::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
    CHECK_DEVICE(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
  } else {
    dq = at::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dk = at::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
    CHECK_DEVICE(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    dv = at::empty_like(k);
  }

  // const at::Tensor& dout_padded = dout;

  // bool loop = seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::hip::HIPGuardMasqueradingAsCUDA device_guard{(char)q.get_device()};

  auto opts = q.options();
  auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
  at::Tensor dq_accum;
  at::Tensor dk_accum, dv_accum;
  if (loop) {
    dq_accum = at::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
    // dk_accum = at::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    // dv_accum = at::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
  }

  at::Tensor dk_expanded, dv_expanded;
  if (num_heads_k != num_heads) {  // MQA / GQA
    dk_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    dv_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
  } else {
    dk_expanded = dk;
    dv_expanded = dv;
  }

  at::PhiloxCudaState philox_args;
  if (is_dropout) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None)
    {
      philox_args = at::PhiloxCudaState(*philox_seed.data_ptr<int64_t>(), *philox_offset.data_ptr<int64_t>());
    } else { // dropout + capture
      philox_args = at::PhiloxCudaState(
                                        philox_seed.data_ptr<int64_t>(), philox_offset.data_ptr<int64_t>(), 0);
    }
  }

  //JCG TODO WE GO IN HERE TODO backwards
  //reorder tensors and make contiguous
  at::Tensor q_t = q.permute({0,2,1,3}).contiguous();
  at::Tensor k_t = k.permute({0,2,1,3}).contiguous();
  at::Tensor v_t = v.permute({0,2,1,3}).contiguous();
  at::Tensor out_t = out.permute({0,2,1,3}).contiguous();

  //reorder tensors and make contiguous
  at::Tensor dq_t = dq.permute({0,2,1,3}).contiguous();
  at::Tensor dk_t = dk.permute({0,2,1,3}).contiguous();
  at::Tensor dv_t = dv.permute({0,2,1,3}).contiguous();
  at::Tensor dout_t = dout.permute({0,2,1,3}).contiguous();

  dim3 block { 64 * 4, 1, 1 };

  at::Tensor new_do = at::empty_like(dout_t).contiguous();
  at::Tensor delta = at::empty_like(softmax_lse).contiguous();

  int d_head = head_size_og;
  hipError_t err; // TODO: Error handling
#define CALL_BWD_PP(FP, PP_BLOCK, PP_DMODEL)                                                                                \
  do {                                                                                                                      \
    dim3 pp_grid;                                                                                                           \
    pp_grid.x = batch_size * num_heads * ((dout_t.size(2) + PP_BLOCK - 1) / PP_BLOCK);                                      \
    pp_grid.y = 1;                                                                                                          \
    pp_grid.z = 1;                                                                                                          \
    oort::bwd_preprocess<PP_BLOCK, PP_DMODEL> pre_opt;                                                                      \
    err = pre_opt(pp_grid, block,                                                                                           \
                  (FP*)(out_t.data_ptr()),                                                                                  \
                  (FP*)(dout_t.data_ptr()),                                                                                 \
                  (FP*)(new_do.data_ptr()),                                                                                 \
                  (float*)(delta.data_ptr()),                                                                               \
                  stream);                                                                                                  \
  } while (0)

#define CALL_BWD_PP_DMODEL(FP, PP_BLOCK)                                                                                    \
  do {                                                                                                                      \
    if (d_head == 16)                                                                                                       \
      CALL_BWD_PP(FP, PP_BLOCK, 16);                                                                                        \
    else if (d_head == 32)                                                                                                  \
      CALL_BWD_PP(FP, PP_BLOCK, 32);                                                                                        \
    else if (d_head == 64)                                                                                                  \
      CALL_BWD_PP(FP, PP_BLOCK, 64);                                                                                        \
    else if (d_head == 128)                                                                                                 \
      CALL_BWD_PP(FP, PP_BLOCK, 128);                                                                                       \
  } while (0)

  if(q_dtype == at::kHalf) {
    if (seqlen_q >= 64)
      CALL_BWD_PP_DMODEL(__fp16, 16);
    else
      CALL_BWD_PP_DMODEL(__fp16, 16);
  } else if (q_dtype == at::kBFloat16) {
    if (seqlen_q >= 64)
      CALL_BWD_PP_DMODEL(__bf16, 16);
    else
      CALL_BWD_PP_DMODEL(__bf16, 16);
  }
#undef CALL_BWD_PP

#define CALL_BWD(FP, BLOCK_M, BLOCK_DMODEL, BLOCK_N, CAUSAL, ENABLE_DROPOUT)                                                \
  do {                                                                                                                      \
    dim3  grid;                                                                                                             \
    grid.x = (seqlen_k + BLOCK_M - 1) / BLOCK_M;                                                                            \
    grid.y = batch_size * num_heads;                                                                                        \
    grid.z = 1;                                                                                                             \
    oort::bwd_kernel_dk_dv<BLOCK_M, BLOCK_DMODEL, BLOCK_N, CAUSAL, ENABLE_DROPOUT> dk_dv_opt;                               \
    err = dk_dv_opt(grid, block,                                                                                            \
                    (FP*)(q_t.data_ptr()), (FP*)(k_t.data_ptr()), (FP*)(v_t.data_ptr()),                                    \
                    softmax_scale, (FP*)out_t.data_ptr(), (FP*)dout_t.data_ptr(),                                           \
                    (FP*)dk_t.data_ptr(),(FP*)dv_t.data_ptr(),                                                              \
                    (float*)(softmax_lse.data_ptr()),                                                                       \
                    (float*)(delta.data_ptr()),                                                                             \
                    q_t.stride(0), q_t.stride(1), q_t.stride(2), q_t.stride(3),                                             \
                    k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),                                             \
                    v_t.stride(0), v_t.stride(1), v_t.stride(2), v_t.stride(3),                                             \
                    q_t.sizes()[0], q_t.sizes()[1], seqlen_q, seqlen_k, p_dropout,                                          \
                    (uint64_t)(philox_args.seed_.val), (uint32_t)(philox_args.offset_.val), stream);                        \
    grid.x = (seqlen_q + BLOCK_M - 1) / BLOCK_M;                                                                            \
    oort::bwd_kernel_dq<BLOCK_M, BLOCK_DMODEL, BLOCK_N, CAUSAL, ENABLE_DROPOUT> dq_opt;                                     \
    err = dq_opt(grid, block,                                                                                               \
                 (FP*)(q_t.data_ptr()), (FP*)(k_t.data_ptr()), (FP*)(v_t.data_ptr()),                                       \
                 softmax_scale, (FP*)out_t.data_ptr(), (FP*)dout_t.data_ptr(),                                              \
                 (FP*)dq_t.data_ptr(),                                                                                      \
                 (float*)(softmax_lse.data_ptr()),                                                                          \
                 (float*)(delta.data_ptr()),                                                                                \
                 q_t.stride(0), q_t.stride(1), q_t.stride(2), q_t.stride(3),                                                \
                 k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),                                                \
                 v_t.stride(0), v_t.stride(1), v_t.stride(2), v_t.stride(3),                                                \
                 q_t.sizes()[0], q_t.sizes()[1], seqlen_q, seqlen_k, p_dropout,                                             \
                 (uint64_t)(philox_args.seed_.val), (uint32_t)(philox_args.offset_.val), stream);                           \
  } while(0)

#define CALL_BWD_DROPOUT(FP, BLOCK_M, BLOCK_DMODEL, BLOCK_N, CAUSAL)                                                        \
  do {                                                                                                                      \
    if (p_dropout > 0.0) {                                                                                                  \
      CALL_BWD(FP, BLOCK_M, BLOCK_DMODEL, BLOCK_N, CAUSAL, true);                                                           \
    } else {                                                                                                                \
      CALL_BWD(FP, BLOCK_M, BLOCK_DMODEL, BLOCK_N, CAUSAL, false);                                                          \
    }                                                                                                                       \
  } while (0)

#define CALL_BWD_DROPOUT_DMODEL(FP, BLOCK_M, BLOCK_N, CAUSAL)                                                               \
  do {                                                                                                                      \
    if (d_head == 16)                                                                                                       \
      CALL_BWD_DROPOUT(FP, BLOCK_M, 16, BLOCK_N, CAUSAL);                                                                   \
    else if (d_head == 32)                                                                                                  \
      CALL_BWD_DROPOUT(FP, BLOCK_M, 32, BLOCK_N, CAUSAL);                                                                   \
    else if (d_head == 64)                                                                                                  \
      CALL_BWD_DROPOUT(FP, BLOCK_M, 64, BLOCK_N, CAUSAL);                                                                   \
    else if (d_head == 128)                                                                                                 \
      CALL_BWD_DROPOUT(FP, BLOCK_M, 128, BLOCK_N, CAUSAL);                                                                  \
  } while (0)

  if (q_dtype == at::kHalf) {
    if (is_causal) {
      CALL_BWD_DROPOUT_DMODEL(__fp16, 16, 16, true);
    } else {
      CALL_BWD_DROPOUT_DMODEL(__fp16, 16, 16, false);
    }
  } else if (q_dtype == at::kBFloat16) {
    if (is_causal) {
      CALL_BWD_DROPOUT_DMODEL(__bf16, 16, 16, true);
    } else {
      CALL_BWD_DROPOUT_DMODEL(__bf16, 16, 16, false);
    }
  }

  //undo reorder tensors for returns
  dq = dq_t.permute({0,2,1,3}).contiguous();
  dk = dk_t.permute({0,2,1,3}).contiguous();
  dv = dv_t.permute({0,2,1,3}).contiguous();

  // For MQA/GQA we need to sum dK and dV across the groups
  if (num_heads_k != num_heads) {
    at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
  }
  return { dq, dk, dv, softmax_d };
#undef CALL_BWD_DROPOUT
#undef CALL_BWD
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
               c10::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               const int window_size_left,
               int window_size_right,
               const at::Tensor philox_seed,
               const at::Tensor philox_offset) {
  TORCH_CHECK(false, "mha_varlen_bwd not supported on ROCm");

  at::Tensor softmax_d = at::empty({}, at::dtype(at::kFloat));

  return { q, k, v, softmax_d };
}
} // namespace pytorch_fmha

#endif
