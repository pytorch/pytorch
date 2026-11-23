#pragma once

#ifdef USE_ROCM

// Expect to be included after headers of at::zeros_like and at::empty_like

#include <aotriton/dtypes.h>
#include <aotriton/util.h>
#include <aotriton/config.h>
#include <aotriton/flash.h>
#include <ATen/native/transformers/hip/aotriton_versions.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <limits>
#include <tuple>
#include <optional>

#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 12)
#define AOTRITON_V2_API_FLASH_ATTN_H  // Suppress the include of deprecated flash/v2.h
#endif

////////////////////////////////////////////////////////////////////////////////
// Common macros copied from cuda/mem_eff_attention/gemm_kernel_utils.h
////////////////////////////////////////////////////////////////////////////////

namespace sdp {

namespace aotriton_adapter {

inline aotriton::DType cast_dtype(caffe2::TypeMeta t_dtype)
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


// Never call at::Tensor::data_ptr() here: it aliases mutable_data_ptr() and so
// materializes lazily-cloned (COW) storage even for tensors AOTriton only reads.
template<int Rank, bool kMutable>
aotriton::TensorView<Rank> mk_aotensor_impl(const at::Tensor& q, std::string_view tensor_name)
{
  const auto strides = q.strides();
  int real_rank = strides.size();
  if (real_rank != Rank) {  // Lazy conversion of tensor_name
    TORCH_CHECK(false,
                std::string(tensor_name) + "'s rank should be " + std::to_string(Rank)
                + " but is " + std::to_string(real_rank));
  }
  intptr_t base;
  if constexpr (kMutable) {
    base = reinterpret_cast<intptr_t>(q.mutable_data_ptr());
  } else {
    base = reinterpret_cast<intptr_t>(q.const_data_ptr());
  }
  return aotriton::TensorView<Rank>(base,
                                    IntArrayRefCaster<uint64_t, Rank>::cast(q.sizes()),
                                    IntArrayRefCaster<uint64_t, Rank>::cast(strides),
                                    cast_dtype(q.dtype()));
}

// For tensors the kernel only reads.
template<int Rank = 4>
aotriton::TensorView<Rank> mk_input_aotensor(const at::Tensor& q, std::string_view tensor_name)
{
  return mk_aotensor_impl<Rank, false>(q, tensor_name);
}

// For tensors the kernel writes into.
template<int Rank = 4>
aotriton::TensorView<Rank> mk_output_aotensor(const at::Tensor& q, std::string_view tensor_name)
{
  return mk_aotensor_impl<Rank, true>(q, tensor_name);
}

// Philox seed/offset are only ever read; the written-to counterparts go through
// mk_philoxtensor() instead.
inline aotriton::TensorView<0> mk_aoscalartensor(const at::Tensor& q)
{
  return aotriton::TensorView<0>(reinterpret_cast<intptr_t>(q.const_data_ptr()),
                                 cast_dtype(q.dtype()));
}

inline aotriton::TensorView<0> mk_philoxtensor(const int64_t* ptr)
{
  return aotriton::TensorView<0>(reinterpret_cast<intptr_t>(ptr),
                                 aotriton::DType::kUInt64);  // AOTriton accepts unsigned int64
}

inline aotriton::TensorView<0> mk_atomictensor(const int32_t* ptr)
{
  return aotriton::TensorView<0>(reinterpret_cast<intptr_t>(ptr),
                                 aotriton::DType::kInt32);
}

#if AOTRITON_VARLEN_BITS_API
// AOTriton 0.14+ replaced the four-value VarlenType enum with a VarlenBits
// bit-field struct: Q and K each pick how their length is given and where
// their sequence starts, independently. Dense needs no bits (zero-init).
constexpr aotriton::v3::flash::VarlenMode mk_varlen_mode(uint32_t length, uint32_t position)
{
  aotriton::v3::flash::VarlenMode mode{};
  mode.stacked = aotriton::v3::flash::VarlenStacked::THD;
  mode.length = length;
  mode.position = position;
  return mode;
}

// Packed varlen: one (N+1,) cumulative array per side gives that side both its
// length (by differencing) and its start position.
constexpr aotriton::v3::flash::VarlenBits mk_varlen_bits_packed()
{
  using aotriton::v3::flash::VarlenLength;
  using aotriton::v3::flash::VarlenPosition;
  const auto packed = mk_varlen_mode(VarlenLength::CUMULATIVE, VarlenPosition::REUSE);
  return aotriton::v3::flash::VarlenBits{.qmode = packed, .kmode = packed};
}

// seqused_k against a packed KV cache: K reads its length from a (N,) array of
// per-sequence counts (seqinfo_k0) and its start from a separate (N+1,)
// cumulative array (seqinfo_k1). Q stays packed. No VarlenType could spell
// this, which is why it only arrives through the struct.
constexpr aotriton::v3::flash::VarlenBits mk_varlen_bits_seqused_k()
{
  using aotriton::v3::flash::VarlenLength;
  using aotriton::v3::flash::VarlenPosition;
  return aotriton::v3::flash::VarlenBits{
    .qmode = mk_varlen_mode(VarlenLength::CUMULATIVE, VarlenPosition::REUSE),
    .kmode = mk_varlen_mode(VarlenLength::INDIVIDUAL, VarlenPosition::ARRAY),
  };
}

#if AOTRITON_PRINCIPAL_DIAGONAL_API
// Numeric windows are shifted per sequence by (seqlen_k - seqlen_q), i.e.
// measured from the bottom-right diagonal. Sentinel windows are not shifted.
constexpr aotriton::v3::flash::VarlenBits with_bottom_right_diagonal(aotriton::v3::flash::VarlenBits bits)
{
  bits.principal_diagonal_choice = aotriton::v3::flash::VarlenPrincipalDiagonal::BOTTOM_RIGHT;
  return bits;
}
#endif // AOTRITON_PRINCIPAL_DIAGONAL_API
#endif // AOTRITON_VARLEN_BITS_API

// Places a numeric window, measured from the bottom-right diagonal of each
// sequence (d = seqlen_k - seqlen_q), in AOTriton's terms. Dense calls shift it
// on the host; varlen callers keep it and set principal_diagonal_choice (see
// with_bottom_right_diagonal), because d differs per sequence. Without that API
// (AOTriton < 0.15) the window stays measured from the top-left diagonal.
inline std::tuple<int, int>
bottom_right_window(int window_left,
                    int window_right,
                    int max_seqlen_q,
                    int max_seqlen_k,
                    bool is_varlen) {
#if AOTRITON_PRINCIPAL_DIAGONAL_API
  if (!is_varlen) {
    const int diag = max_seqlen_k - max_seqlen_q;
    return std::make_tuple(window_left - diag, window_right + diag);
  }
#endif
  return std::make_tuple(window_left, window_right);
}

// The mem-efficient API's window_size is a number of keys ending at the causal
// diagonal: key j is visible to query i iff i + d - window_size < j <= i + d,
// where d = 0 (top-left) or seqlen_k - seqlen_q (bottom-right). AOTriton windows
// are inclusive: (window_size - 1, 0) around that diagonal. This is not FA2's
// window, so it skips calculate_swa's normalization; varlen callers must
// additionally set with_bottom_right_diagonal on VarlenBits.
inline std::tuple<int, int>
mem_eff_window_aot(int64_t window_size,
                   bool bottom_right,
                   int64_t max_seqlen_q,
                   int64_t max_seqlen_k,
                   bool is_varlen) {
  TORCH_CHECK(window_size > 0 && window_size <= std::numeric_limits<int>::max(),
              "window_size must be positive and fit in int32");
  const int window_left = static_cast<int>(window_size - 1);
  if (!bottom_right) {
    return std::make_tuple(window_left, 0);
  }
#if !AOTRITON_PRINCIPAL_DIAGONAL_API
  // bottom_right_window cannot place the window there; do not silently fall
  // back to a top-left window.
  TORCH_CHECK(false,
              "window_size with a bottom-right causal mask requires AOTriton >= 0.15");
#endif
  return bottom_right_window(window_left,
                             0,
                             static_cast<int>(max_seqlen_q),
                             static_cast<int>(max_seqlen_k),
                             is_varlen);
}

// Applies the mem-efficient API's window_size, if set and positive, to
// attn_fwd_params or attn_bwd_params. Call after the causal window sentinels
// and, for varlen, varlen_bits have been set: both are overridden here.
template <typename Params>
void mem_eff_apply_window(Params& params,
                          std::optional<int64_t> window_size,
                          int64_t custom_mask_type,
                          int64_t max_seqlen_q,
                          int64_t max_seqlen_k,
                          bool is_varlen) {
  if (!window_size.has_value() || *window_size <= 0) {
    return;
  }
#if AOTRITON_V3_API
  TORCH_CHECK(static_cast<int64_t>(sdp::CustomMaskType::NoCustomMask) != custom_mask_type,
              "window_size requires a causal custom_mask_type");
  const bool bottom_right =
      static_cast<int64_t>(sdp::CustomMaskType::CausalFromBottomRight) == custom_mask_type;
  std::tie(params.window_left, params.window_right) =
      mem_eff_window_aot(*window_size,
                         bottom_right,
                         max_seqlen_q,
                         max_seqlen_k,
                         is_varlen);
#if AOTRITON_PRINCIPAL_DIAGONAL_API
  if (bottom_right && is_varlen) {
    params.varlen_bits = with_bottom_right_diagonal(params.varlen_bits);
  }
#endif
#else
  TORCH_CHECK(false, "window_size requires AOTriton >= 0.10");
#endif
}

#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 11)

struct LazyTensorContext {
  at::Tensor like_tensor;
  std::string_view tensor_name;
  at::Tensor tensor;
};

template<int kRank, bool kRequireZeros>
struct LazyTensorFunctions : public LazyTensorContext {
#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 12)
  using HolderType = aotriton::LazyTensor<kRank>;
#else
  using HolderType = void;
#endif
  static aotriton::TensorView<kRank> acquire(HolderType* self) {
#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 12)
    auto ctx = (LazyTensorContext*)self->cookie;
#else
    auto ctx = (LazyTensorContext*)cookie;
#endif
    if (!ctx->tensor.defined()) {
      auto q = ctx->like_tensor;
      if constexpr (kRequireZeros) {
        ctx->tensor = at::zeros(q.sizes(),
                                q.options().dtype(at::kFloat));
      } else {
        ctx->tensor = at::empty_like(q);
      }
    }
    return mk_output_aotensor<kRank>(ctx->tensor, ctx->tensor_name);
  }

  static void dispose(HolderType* cookie) {
  }
};

template<int kRank, bool kRequireZeros>
aotriton::LazyTensor<kRank> mklazy_common(LazyTensorContext* cookie)
{
  using LTF = LazyTensorFunctions<kRank, kRequireZeros>;
  return aotriton::LazyTensor<kRank> {
    .cookie = cookie,
    .acquire = &LTF::acquire,
    .dispose = &LTF::dispose
  };
}

template<int kRank>
auto mklazy_empty_like(LazyTensorContext* cookie)
{
  return mklazy_common<kRank, false>(cookie);
}


// Note: this will not keep the original strides
template<int kRank>
auto mklazy_fp32zeros(LazyTensorContext* cookie)
{
  return mklazy_common<kRank, true>(cookie);
}

#endif  // >= 0.11

} // namespace aotriton_adapter

} // namespace sdp

namespace at::native {

inline int64_t ceil_div(int64_t numerator, int64_t denominator) {
  return (numerator + (denominator - 1)) / denominator;
}

}

#endif // USE_ROCM
