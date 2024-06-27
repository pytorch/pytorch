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
//#include <c10/cuda/CUDAStream.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/torch.h>

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
    bool PADDING = false,
    bool BIAS = false>
at::Tensor bf16_gemm_impl(at::Tensor A, at::Tensor B, std::optional<at::Tensor> bias) {
  // Get input information.
  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);

  int StrideA = K;
  int StrideB = K;
  int StrideC = N;

  auto C = at::empty({M, N}, A.options().dtype(at::kBFloat16));

  using ADataType = ck::bhalf_t;
  using BDataType = ck::bhalf_t;
  using DDataType = ck::bhalf_t;
  using DsDataType = std::conditional_t<BIAS, ck::Tuple<DDataType>, ck::Tuple<>>;
  using AccDataType = float;
  using CShuffleDataType = ck::bhalf_t;
  using CDataType = ck::bhalf_t;

  using ALayout = Row;
  using BLayout = Col;
  using DLayout = Row;
  using DsLayout = std::conditional_t<BIAS, ck::Tuple<DLayout>, ck::Tuple<>>;
  using CLayout = Row;

  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CElementOp = std::conditional_t<BIAS, Add, PassThrough>;

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

  using DDataArrayType = std::conditional_t<BIAS, std::array<const void*, 1>, std::array<const void*, 0>>;
  using DStrideArrayType = std::conditional_t<BIAS, std::array<ck::index_t, 1>, std::array<ck::index_t, 0>>;

  DDataArrayType DDataArray;
  DStrideArrayType DStrideArray;

  if constexpr (BIAS) {
    DDataArray = {reinterpret_cast<DDataType*>(bias.value().data_ptr())};
    DStrideArray = {0};
  }

  auto arguments = gemm.MakeArgument(
      reinterpret_cast<ADataType*>(A.data_ptr()),
      reinterpret_cast<BDataType*>(B.data_ptr()),
      DDataArray,
      reinterpret_cast<CDataType*>(C.data_ptr()),
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

  return C;
}

template<bool USE_BIAS=false>
at::Tensor dispatch_bf16_gemm(at::Tensor A, at::Tensor B, std::optional<at::Tensor> bias) {
  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);
  // If any of the shapes cant be tiled, we must use padding.
  bool use_padding = ((M % 256 != 0) || (N % 128 != 0) || (K % 64 != 0));
  // Dispatch to best implementation. TODO add more configurations.
  if (use_padding) {
    if (M <= 128) {
      return bf16_gemm_impl<256, 128, 64, 32, 32, 32, 2, 1, 1, true, USE_BIAS>(A, B, bias);
    } else if ((M >= 8192 && N >= 4096) || (N >= 8192 && M >= 4096)) {
      return bf16_gemm_impl<256, 256, 128, 64, 16, 16, 8, 4, 2, true, USE_BIAS>(A, B, bias);
    } else {
      return bf16_gemm_impl<256, 128, 128, 64, 16, 16, 4, 4, 2, true, USE_BIAS>(A, B, bias);
    }
  } else {
    if ((M >= 8192 && N >= 4096) || (N >= 8192 && M >= 4096)) {
      return bf16_gemm_impl<256, 256, 128, 64, 16, 16, 8, 4, 2, false, USE_BIAS>(A, B, bias);
    } else {
      return bf16_gemm_impl<256, 128, 128, 64, 16, 16, 4, 4, 2, false, USE_BIAS>(A, B, bias);
    }
  }
}

at::Tensor bf16_gemm(at::Tensor A, at::Tensor B, std::optional<at::Tensor> bias = c10::nullopt) {
  TORCH_CHECK(
      A.dtype() == at::kBFloat16 && B.dtype() == at::kBFloat16,
      "Inputs must be bfloat16.");
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dtype() == at::kBFloat16, "Only bfloat16 currently supported for bias.");
    return dispatch_bf16_gemm<true>(A, B, bias);
  } else {
    return dispatch_bf16_gemm<false>(A, B, bias);
  }
}

} // namespace at::native

