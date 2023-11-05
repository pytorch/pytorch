#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <mkl_types.h>

static inline MKL_INT mkl_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<MKL_INT>(value);
  TORCH_CHECK(
      static_cast<int64_t>(result) == value,
      "mkl_int_cast: The value of ",
      varname,
      "(",
      (long long)value,
      ") is too large to fit into a MKL_INT (",
      sizeof(MKL_INT),
      " bytes)");
  return result;
}

#ifdef MKL_ILP64
#define TORCH_COMPATIPLE_MKL_INT int64_t
static constexpr c10::ScalarType TORCH_COMPATIPLE_MKL_INT_TYPE = at::kLong;
static_assert(
    sizeof(MKL_INT)==sizeof(TORCH_COMPATIPLE_MKL_INT),
    "MKL_INT is assumed to be castable to int64_t when MKL model is ILP64");
#else
#define TORCH_COMPATIPLE_MKL_INT int32_t
static constexpr c10::ScalarType TORCH_COMPATIPLE_MKL_INT_TYPE = at::kInt;
static_assert(
    sizeof(MKL_INT)==sizeof(TORCH_COMPATIPLE_MKL_INT),
    "MKL_INT is assumed to be castable to int32_t when MKL model is LP64");
#endif

#define MKL_TENSOR_DATA_PTR(INPUT) reinterpret_cast<MKL_INT*>((INPUT).data_ptr<TORCH_COMPATIPLE_MKL_INT>())
#define MKL_TENSOR_PTR_DATA_PTR(INPUT) reinterpret_cast<MKL_INT*>((INPUT)->data_ptr<TORCH_COMPATIPLE_MKL_INT>())
#define MKL_TENSOR_MUTABLE_DATA_PTR(INPUT) reinterpret_cast<MKL_INT*>((INPUT).mutable_data_ptr<TORCH_COMPATIPLE_MKL_INT>())
