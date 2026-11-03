#pragma once

#ifdef USE_ROCM

// Expect to be included after headers of at::zeros_like and at::empty_like

#include <aotriton/dtypes.h>
#include <aotriton/util.h>
#include <aotriton/config.h>
#include <aotriton/flash.h>
#include <ATen/native/transformers/hip/aotriton_versions.h>
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
#endif // AOTRITON_VARLEN_BITS_API

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

inline auto parse_window_size(std::optional<int64_t> window_size_left,
                              std::optional<int64_t> window_size_right)
{
  const int fa_left = window_size_left.value_or(-1);
  const int fa_right = window_size_right.value_or(-1);
  auto get_window_value = [](const int window) -> std::optional<int64_t> {
    if (window < 0) {
      return std::nullopt;
    }
    return window;
  };
  const auto window_left = get_window_value(fa_left);
  const auto window_right = get_window_value(fa_right);
  return std::make_tuple(window_left, window_right);
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
