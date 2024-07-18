/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#undef __HIP_NO_HALF_CONVERSIONS__

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/torch.h>
#include <ATen/native/hip/ck_gemm.h>


#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

namespace at::native {

template <>
void gemm_internal_ck<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  return;
}

template <typename T>
struct CkMathType {
  using dtype = T;
};

template <>
struct CkMathType<at::BFloat16> {
  using dtype = ck::bhalf_t;
};

template <>
struct CkMathType<at::Half> {
  using dtype = ck::half_t;
};

// Elementwise Operators
struct AlphaBetaAdd
{
  AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

  template <typename C, typename AB>
  __host__ __device__ constexpr void operator()(C& c, const AB& ab) const;

  template<>
  __host__ __device__ constexpr void operator()<float, float>
    (float& c, const float& ab) const
    {
      c = alpha_ * ab;
    };

  template<>
  __host__ __device__ constexpr void operator()<ck::bhalf_t, ck::bhalf_t>
    (ck::bhalf_t& c, const ck::bhalf_t& ab) const
    {
      c = alpha_ * ab;
    };

  template<>
  __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>
    (ck::half_t& c, const ck::half_t& ab) const
    {
      c = alpha_ * ab;
    };

    float alpha_;
    // TODO: Leaving for now, will use later
    float beta_;
};

template <
    typename Dtype,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int AK1,
    int BK1,
    int MPER_XDL,
    int NPER_XDL,
    int MPER_WAVE,
    int NPER_WAVE,
    int CNPER_WAVE = 1,
    bool PADDING = false>
void gemm_impl(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  // Get input information.
  int M = m;
  int N = n;
  int K = k;

  int StrideA = lda;
  int StrideB = ldb;
  int StrideC = ldc;

  float falpha = alpha;
  float fbeta = beta;


  using ADataType = typename CkMathType<Dtype>::dtype;
  using BDataType = typename CkMathType<Dtype>::dtype;
  using CDataType = typename CkMathType<Dtype>::dtype;
  using DDataType = typename CkMathType<Dtype>::dtype;

  using AccDataType = float;
  using CShuffleDataType = typename CkMathType<Dtype>::dtype;


  // NOTE: in our example, transa = t and transb = n;
  // since default for cublas is Column-major, since the value is T, ALayout is Row
  // same for B. transb = N = NO Transpose so B is column Major
  using ALayout = Row;
  using BLayout = Col;
  using DLayout = Row;
  using CLayout = Row;

  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CElementOp = AlphaBetaAdd;

  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::Default;
  static constexpr auto GemmMNKPadding =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;
  static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;


  // TODO: Flesh out template parameters
  using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<ALayout,
                                                                   BLayout,
                                                                   ck::Tuple<>,
                                                                   CLayout,
                                                                   ADataType,
                                                                   BDataType,
                                                                   ck::Tuple<>,
                                                                   CDataType,
                                                                   AccDataType,
                                                                   CShuffleDataType,
                                                                   AElementOp,
                                                                   BElementOp,
                                                                   CElementOp,
                                                                   GemmSpec,
                                                                   BLOCK_SIZE, //256,
                                                                   MBLOCK, //256,
                                                                   NBLOCK, //128,
                                                                   KBLOCK, //32,
                                                                   AK1, //4,
                                                                   BK1, //4,
                                                                   MPER_XDL, //32,
                                                                   NPER_XDL, //32,
                                                                   MPER_WAVE, //4,
                                                                   NPER_WAVE, //2,
                                                                   S<8, 32, 1>,
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   4,
                                                                   4,
                                                                   0,
                                                                   S<8, 32, 1>,
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   4,
                                                                   4,
                                                                   0,
                                                                   1,
                                                                   1,
                                                                   S<1, 32, 1, 8>,
                                                                   S<4>>;


  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto c_element_op = CElementOp{alpha, beta};

  using DDataArrayType = std::array<const void*, 0>;
  DDataArrayType DDataArray;

  // Note: CK only supports row-major output.
  // We swap A and B inputs here as a temporary workaround
  auto argument = gemm.MakeArgument(
     reinterpret_cast<const void*>(b),
     reinterpret_cast<const void*>(a),
     DDataArray,
     reinterpret_cast<void*>(c),
     N,
     M,
     K,
     StrideB,
     StrideA,
     std::array<ck::index_t, 0>{},
     StrideC,
     a_element_op,
     b_element_op,
     c_element_op);


 if(!gemm.IsSupportedArgument(argument))
 {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
 }


 auto stream = at::cuda::getCurrentHIPStream().stream();
 invoker.Run(argument, StreamConfig{stream, false});
}

void dispatch_float_gemm(CUDABLAS_GEMM_ARGTYPES(float)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.
  if (use_padding) {
    if (m <= 128) {
      gemm_impl<float, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(float));
    } else {
      gemm_impl<float, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(float));
    }
  } else {
    {
      gemm_impl<float, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, false>(CUDABLAS_GEMM_ARGS(float));
    }
  }
}

void dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.
  if (use_padding) {
    if (m <= 128) {
      gemm_impl<at::BFloat16, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else {
      gemm_impl<at::BFloat16, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    }
  } else {
    {
      gemm_impl<at::BFloat16, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, false>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    }
  }
}

void dispatch_half_gemm(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation.
  // TODO add more configurations. Optimize.
  if (use_padding) {
    if (m <= 128) {
      gemm_impl<at::Half, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::Half));
    } else {
      gemm_impl<at::Half, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, true>(CUDABLAS_GEMM_ARGS(at::Half));
    }
  } else {
    {
      gemm_impl<at::Half, 256, 256, 128, 32, 4, 4, 32, 32, 4, 2, 1, false>(CUDABLAS_GEMM_ARGS(at::Half));
    }
  }
}

template <>
void gemm_internal_ck<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  dispatch_float_gemm(CUDABLAS_GEMM_ARGS(float));
}

template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  dispatch_bfloat16_gemm(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

template <>
void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  dispatch_half_gemm(CUDABLAS_GEMM_ARGS(at::Half));
}

} // namespace at::native

