/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#undef __HIP_NO_HALF_CONVERSIONS__
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/blkgemmpipe_scheduler.hpp>
#include <ck/utility/data_type.hpp>

#include <ck/library/reference_tensor_operation/cpu/reference_gemm.hpp>
#include <ck/library/utility/check_err.hpp>
#include <ck/library/utility/device_memory.hpp>
#include <ck/library/utility/fill.hpp>
#include <ck/library/utility/host_tensor.hpp>
#include <ck/library/utility/host_tensor_generator.hpp>
#include <ck/library/utility/literals.hpp>

#include <ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp>

namespace at::cuda::detail {

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using D0DataType = float;
using D1DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType>;
using EDataType = ck::bhalf_t;
using AccDataType = float;
using CShuffleDataType = float;

using ALayout = Row;
using BLayout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;

using ComputeType = ck::f8_t;

struct RowwiseScale {
  template <typename E, typename C, typename D0, typename D1>
  __host__ __device__ constexpr void
  operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

  template <>
  __host__ __device__ constexpr void
  operator()<ck::bhalf_t, float, float, float>(
      ck::bhalf_t& e,
      const float& c,
      const float& d0,
      const float& d1) const {
    const float x0_f = c * d0 * d1;

    e = ck::type_convert<ck::bhalf_t>(x0_f);
  }
};

using CDEElementOp = RowwiseScale;

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::tensor_operation::device::GemmSpecialization GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::MNPadding>
using DeviceGemmHelper =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        ADataType,
        BDataType,
        DsDataType,
        EDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        BElementOp,
        CDEElementOp,
        GEMM_SPEC,
        BLOCK_SIZE, // Block Size
        MBLOCK, // M per Block
        NBLOCK, // N per Block
        KBLOCK, // K per Block
        16, // AK1
        16, // BK1
        WAVE_TILE_M, // M per Xdl
        WAVE_TILE_N, // N per Xdl
        WAVE_MAP_M, // Mxdl per Wave
        WAVE_MAP_N, // Nxdl per Wave
        ABLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        BBLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        LOOP_SCHED,
        PIPELINE_VERSION,
        ComputeType>;

template <typename DeviceGemmInstance>
at::Tensor f8f8bf16_rowwise_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // Get input information.
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  int StrideA = K;
  int StrideB = K;
  int StrideE = N;

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};

  constexpr ck::index_t NumDTensor = ck::Number<2>{};

  auto argument = gemm.MakeArgument(
      reinterpret_cast<ADataType*>(XQ.data_ptr()),
      reinterpret_cast<BDataType*>(WQ.data_ptr()),
      std::array<const void*, NumDTensor>{
          reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
          reinterpret_cast<D1DataType*>(x_scale.data_ptr())},
      reinterpret_cast<EDataType*>(Y.data_ptr()),
      M,
      N,
      K,
      StrideA,
      StrideB,
      std::array<ck::index_t, NumDTensor>{0, 0},
      StrideE,
      1,
      a_element_op,
      b_element_op,
      cde_element_op);

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});

  return Y;
}

} // namespace at::cuda::detail
