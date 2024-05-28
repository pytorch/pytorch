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

// AOTriton headers
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>

namespace pytorch_flash {

namespace {

void check_gpu_arch(hipStream_t stream) {
  auto ret = aotriton::v2::flash::check_gpu(stream);
  if (hipSuccess != ret) {
      TORCH_CHECK(false,
                  "FlashAttention only supports MI200/MI300X GPUs (gfx90a:sramecc+:xnack- or gfx94a:sramecc+:xnack-)")
  }
}

aotriton::DType cast_dtype(caffe2::TypeMeta t_dtype)
{
#define CAST_TYPE(aname, dtname) if (t_dtype == at::aname) return aotriton::DType::dtname
  CAST_TYPE(kByte, kUInt8);
  CAST_TYPE(kUInt16, kUInt16);
  CAST_TYPE(kUInt32, kUInt32);
  CAST_TYPE(kUInt64, kUInt64);
  CAST_TYPE(kChar, kInt8);
  CAST_TYPE(kShort, kInt16);
  CAST_TYPE(kInt, kInt32);
  CAST_TYPE(kLong, kInt64);
  CAST_TYPE(kHalf, kFloat16);
  CAST_TYPE(kFloat, kFloat32);
  CAST_TYPE(kBFloat16, kBFloat16);
  return aotriton::DType::kUnknown;
#undef CAST_TYPE
}

template<typename TargetType, int Rank>
struct IntArrayRefCaster {
  // std::array<TargetType, Rank> cast(IntArrayRef);
};

template<typename TargetType>
struct IntArrayRefCaster<TargetType, 1> {
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 1>{{ static_cast<TargetType>(ref.at(0)) }};
  }
};

template<typename TargetType>
struct IntArrayRefCaster<TargetType, 2> {
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 2>{{
      static_cast<TargetType>(ref.at(0)),
      static_cast<TargetType>(ref.at(1))
    }};
  }
};

template<typename TargetType>
struct IntArrayRefCaster<TargetType, 3> {
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 3>{{
      static_cast<TargetType>(ref.at(0)),
      static_cast<TargetType>(ref.at(1)),
      static_cast<TargetType>(ref.at(2))
    }};
  }
};

template<typename TargetType>
struct IntArrayRefCaster<TargetType, 4> {
  static auto cast(at::IntArrayRef ref) {
    return std::array<TargetType, 4>{{
      static_cast<TargetType>(ref.at(0)),
      static_cast<TargetType>(ref.at(1)),
      static_cast<TargetType>(ref.at(2)),
      static_cast<TargetType>(ref.at(3))
    }};
  }
};


template<int Rank = 4>
aotriton::TensorView<Rank> mk_aotensor(const at::Tensor& q, c10::string_view tensor_name)
{
  const auto strides = q.strides();
  int real_rank = strides.size();
  if (real_rank != Rank) {  // Lazy convertion of tensor_name
    TORCH_CHECK(false,
                std::string(tensor_name) + "'s rank should be " + std::to_string(Rank)
                + " but is " + std::to_string(real_rank));
  }
  return aotriton::TensorView<Rank>(reinterpret_cast<intptr_t>(q.data_ptr()),
                                    IntArrayRefCaster<uint64_t, Rank>::cast(q.sizes()),
                                    IntArrayRefCaster<uint64_t, Rank>::cast(strides),
                                    cast_dtype(q.dtype()));
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
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool return_softmax,
        c10::optional<at::Generator> gen_) {
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  check_gpu_arch(stream);

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

  at::PhiloxCudaState philox_args;
  if (p_dropout > 0.0) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None)
    {
      philox_args = at::PhiloxCudaState(*seed_t.data_ptr<int64_t>(), *offset_t.data_ptr<int64_t>());
    } else { // dropout + capture
      philox_args = at::PhiloxCudaState(seed_t.data_ptr<int64_t>(), offset_t.data_ptr<int64_t>(), 0);
    }
  }

  // Transpose tensors to meet AOTriton's Flash API
  at::Tensor q_t = q_padded.permute({0,2,1,3});
  at::Tensor k_t = k_padded.permute({0,2,1,3});
  at::Tensor v_t = v_padded.permute({0,2,1,3});
  at::Tensor output_t = out.permute({0,2,1,3});

  at::Tensor M = at::empty({batch_size * num_heads, seqlen_q}, at::dtype(at::kFloat).device(q.device())); // aka softmax_lse

  at::Tensor softmax_fa_t;
  if (return_softmax) {
    softmax_fa_t = at::empty({batch_size, num_heads, seqlen_q, seqlen_k},
                             at::dtype(q.dtype()).device(q.device()));
  } else {
    softmax_fa_t = at::empty({ 0, 0, 0, 0 }, at::dtype(q.dtype()).device(q.device()));
  }

  hipError_t err; // TODO: Error handling
  using aotriton::v2::flash::attn_fwd;
  err = attn_fwd(mk_aotensor(q_t, "q"),
                 mk_aotensor(k_t, "k"),
                 mk_aotensor(v_t, "v"),
                 softmax_scale,
                 mk_aotensor<2>(M, "M"),
                 mk_aotensor(output_t, "Out"),
                 p_dropout,
                 philox_args.seed_.val,
                 philox_args.offset_.val,
                 mk_aotensor(softmax_fa_t, "encoded_softmax"),
                 is_causal,
                 stream);

  return {out, q_padded, k_padded, v_padded, M, seed_t, offset_t, softmax_fa_t};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_varlen_fwd(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
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
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool deterministic,
        const at::Tensor philox_seed,
        const at::Tensor philox_offset) {
  auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  check_gpu_arch(stream);

  bool is_dropout = p_dropout > 0.0;

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

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::hip::HIPGuardMasqueradingAsCUDA device_guard{(char)q.get_device()};

  auto opts = q.options();
  auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));

  at::Tensor dk_expanded, dv_expanded;
  if (num_heads_k != num_heads) {  // MQA / GQA
    dk_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    dv_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
  } else {
    dk_expanded = dk;
    dv_expanded = dv;
  }

  at::PhiloxCudaState philox_args;
  if (p_dropout > 0.0) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None)
    {
      philox_args = at::PhiloxCudaState(*philox_seed.data_ptr<int64_t>(), *philox_offset.data_ptr<int64_t>());
    } else { // dropout + capture
      philox_args = at::PhiloxCudaState(philox_seed.data_ptr<int64_t>(), philox_offset.data_ptr<int64_t>(), 0);
    }
  }

  at::Tensor q_t = q.permute({0,2,1,3});
  at::Tensor k_t = k.permute({0,2,1,3});
  at::Tensor v_t = v.permute({0,2,1,3});
  at::Tensor out_t = out.permute({0,2,1,3});
  at::Tensor dq_t = dq.permute({0,2,1,3});
  at::Tensor dk_t = dk.permute({0,2,1,3});
  at::Tensor dv_t = dv.permute({0,2,1,3});
  at::Tensor dout_t = dout.permute({0,2,1,3});

  at::Tensor softmax_lse_cont = softmax_lse.contiguous();
  at::Tensor delta = at::empty_like(softmax_lse).contiguous();

  int d_head = head_size_og;
  hipError_t err; // TODO: Error handling
  {
    using aotriton::v2::flash::attn_bwd;
    err = attn_bwd(mk_aotensor(q_t, "q"),
                   mk_aotensor(k_t, "k"),
                   mk_aotensor(v_t, "v"),
                   softmax_scale,
                   mk_aotensor(out_t, "out"),
                   mk_aotensor(dout_t, "dout"),
                   mk_aotensor(dq_t, "dq"),
                   mk_aotensor(dk_t, "dk"),
                   mk_aotensor(dv_t, "dv"),
                   mk_aotensor<2>(softmax_lse_cont, "L"),
                   mk_aotensor<2>(delta, "delta"),
                   p_dropout,
                   philox_args.seed_.val,
                   philox_args.offset_.val,
                   is_causal,
                   stream);
  }

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
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool deterministic,
               const at::Tensor philox_seed,
               const at::Tensor philox_offset) {
  TORCH_CHECK(false, "mha_varlen_bwd not supported on ROCm");

  at::Tensor softmax_d = at::empty({}, at::dtype(at::kFloat));

  return { q, k, v, softmax_d };
}
} // namespace pytorch_fmha

#endif
