#pragma once

#ifdef USE_ROCM

#include <aotriton/dtypes.h>
#include <aotriton/util.h>

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

} // namespace aotriton_adapter

} // namespace sdp

namespace at::native {

inline int64_t ceil_div(int64_t numerator, int64_t denominator) {
  return (numerator + (denominator - 1)) / denominator;
}

}

#endif // USE_ROCM
