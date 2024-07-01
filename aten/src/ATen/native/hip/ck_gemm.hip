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
using Add = ck::tensor_operation::element_wise::Add;

namespace at::native {

template <>
void gemm_internal_ck<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  return;
}

template <>
void gemm_internal_ck<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  return;
}

template <>
void gemm_internal_ck<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  return;
}

template <>
void gemm_internal_ck<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  return;
}

template <>
void gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  return;
}

/*
TODO - Remove this blurb
Parameters traditionally being passed to gemms
- char transa
- char transb
- int64 m
- int64 n
- int64 k
- opMath_type alpha

- const dtype * a
- int64 lda
- const dtype * b
- int64 ldb
- opMath_type beta
- dtype * c
- ldc

*/



template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int MPER_XDL,
    int NPER_XDL,
    int MPER_WAVE,
    int NPER_WAVE,
    int CNPER_WAVE = 1,
    bool PADDING = false>
void bf16_gemm_impl(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // Get input information.
  int M = m;
  int N = n;
  int K = k;

  int StrideA = lda;
  int StrideB = ldb;
  int StrideC = ldc;

  using ADataType = ck::bhalf_t;
  using BDataType = ck::bhalf_t;
  using DDataType = ck::bhalf_t;
  using DsDataType = ck::Tuple<>;
  using AccDataType = float;
  using CShuffleDataType = ck::bhalf_t;
  using CDataType = ck::bhalf_t;

  using ALayout = Row;
  using BLayout = Col;
  using DLayout = Row;
  using DsLayout = ck::Tuple<>;
  using CLayout = Row;

  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CElementOp = PassThrough;

  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::Default;
  static constexpr auto GemmMNKPadding =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;
  static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;

  // Define derivative constants based on template parameters.
  static constexpr int BLOCK_CLUSTER = BLOCK_SIZE / 4;
  static constexpr int CBLOCK_N = NBLOCK / 16;
  static constexpr int CBLOCK_M = BLOCK_SIZE / CBLOCK_N;

  using DeviceGemmV2Instance =
      ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
          ALayout,
          BLayout,
          DsLayout,
          CLayout,
          ADataType,
          BDataType,
          DsDataType,
          CDataType,
          AccDataType,
          CShuffleDataType,
          AElementOp,
          BElementOp,
          CElementOp,
          GemmSpec,
          BLOCK_SIZE,
          MBLOCK,
          NBLOCK,
          KBLOCK,
          8, // AK1
          8, // BK1
          MPER_XDL, // MPerXDL
          NPER_XDL, // NPerXDL
          MPER_WAVE, // MXdlPerWave
          NPER_WAVE, // NXdlPerWave
          S<4, BLOCK_CLUSTER, 1>, // ABlockTransferThreadClusterLengths
          S<1, 0, 2>, // ABlockTransferThreadClusterOrder
          S<1, 0, 2>, // ABlockTransferSrcAccessOrder
          2, // ABlockTransferSrcVectorDim
          8, // ABlockTransferSrcScalarPerVector
          8, // ABlockTransferDstScalarPerVector
          0, // ABlockLdsExtraM
          S<4, BLOCK_CLUSTER, 1>, // BBlockTransferThreadClusterLengths
          S<1, 0, 2>, // BBlockTransferThreadClusterArrangeOrder
          S<1, 0, 2>, // BBlockTransferSrcAccessOrder
          2, // BBlockTransferSrcVectorDim
          8, // BBlockTransferSrcScalarPerVector
          8, // BBlockTransferDstScalarPerVector
          0, // BBlockLdsExtraN
          1, // CShuffleMXdlPerWavePerShuffle
          CNPER_WAVE, // CShuffleNXdlPerWavePerShuffle
          S<1, CBLOCK_M, 1, CBLOCK_N>, // CShuffleBlockTransferClusterLengths
          S<8, 8, 1>, // CShuffleBlockTransferScalarPerVector
          ck::BlockGemmPipelineScheduler::Intrawave, // Pipeline Schedule
          ck::BlockGemmPipelineVersion::v3>; // Pipeline Version

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmV2Instance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto c_element_op = CElementOp{};

  using DDataArrayType = std::array<const void*, 0>;
  using DStrideArrayType = std::array<ck::index_t, 0>;

  DDataArrayType DDataArray;
  DStrideArrayType DStrideArray;

  auto arguments = gemm.MakeArgument(
      reinterpret_cast<const ADataType*>(a),
      reinterpret_cast<const BDataType*>(b),
      DDataArray,
      reinterpret_cast<CDataType*>(c),
      M,
      N,
      K,
      StrideA,
      StrideB,
      DStrideArray,
      StrideC,
      a_element_op,
      b_element_op,
      c_element_op);

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(arguments, StreamConfig{stream, false});
}

void dispatch_bf16_gemm(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((m % 256 != 0) || (n % 128 != 0) || (k % 64 != 0));
  // Dispatch to best implementation. TODO add more configurations.
  if (use_padding) {
    if (m <= 128) {
      bf16_gemm_impl<256, 128, 64, 32, 32, 32, 2, 1, 1, true>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else if ((m >= 8192 && n >= 4096) || (n >= 8192 && m >= 4096)) {
      bf16_gemm_impl<256, 256, 128, 64, 16, 16, 8, 4, 2, true>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else {
      bf16_gemm_impl<256, 128, 128, 64, 16, 16, 4, 4, 2, true>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    }
  } else {
    if ((m >= 8192 && n >= 4096) || (n >= 8192 && m >= 4096)) {
      bf16_gemm_impl<256, 256, 128, 64, 16, 16, 8, 4, 2, false>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    } else {
      bf16_gemm_impl<256, 128, 128, 64, 16, 16, 4, 4, 2, false>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    }
  }
}

template <>
void gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  dispatch_bf16_gemm(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

} // namespace at::native

