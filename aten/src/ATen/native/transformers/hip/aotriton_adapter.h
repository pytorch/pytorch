#pragma once

#ifdef USE_ROCM

// Expect to be included after headers of at::zeros_like and at::empty_like

#include <aotriton/dtypes.h>
#include <aotriton/util.h>
#include <aotriton/config.h>
#include <ATen/native/transformers/hip/aotriton_versions.h>

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


template<int Rank = 4>
aotriton::TensorView<Rank> mk_aotensor(const at::Tensor& q, std::string_view tensor_name)
{
  const auto strides = q.strides();
  int real_rank = strides.size();
  if (real_rank != Rank) {  // Lazy conversion of tensor_name
    TORCH_CHECK(false,
                std::string(tensor_name) + "'s rank should be " + std::to_string(Rank)
                + " but is " + std::to_string(real_rank));
  }
  return aotriton::TensorView<Rank>(reinterpret_cast<intptr_t>(q.data_ptr()),
                                    IntArrayRefCaster<uint64_t, Rank>::cast(q.sizes()),
                                    IntArrayRefCaster<uint64_t, Rank>::cast(strides),
                                    cast_dtype(q.dtype()));
}

inline aotriton::TensorView<0> mk_aoscalartensor(const at::Tensor& q)
{
  return aotriton::TensorView<0>(reinterpret_cast<intptr_t>(q.data_ptr()),
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

#if AOTRITON_VERSION_CURRENT >= AOTRITON_VERSION_INT(0, 11)

struct LazyTensorContext {
  at::Tensor like_tensor;
  std::string_view tensor_name;
  at::Tensor tensor;
};

template<int kRank, bool kRequireZeros>
struct LazyTensorFunctions : public LazyTensorContext {
  static aotriton::TensorView<kRank> acquire(void* cookie) {
    auto ctx = (LazyTensorContext*)cookie;
    if (!ctx->tensor.defined()) {
      auto q = ctx->like_tensor;
      if constexpr (kRequireZeros) {
        ctx->tensor = at::zeros(q.sizes(),
                                q.options().dtype(at::kFloat));
      } else {
        ctx->tensor = at::empty_like(q);
      }
    }
    return mk_aotensor<kRank>(ctx->tensor, ctx->tensor_name);
  }

  static void dispose(void* cookie) {
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
