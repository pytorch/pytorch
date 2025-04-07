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
#include <numeric>

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <ATen/native/hip/ck_gemm.h>
#include <ATen/native/hip/ck_types.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/data_type.hpp>

#include <ck/library/reference_tensor_operation/cpu/reference_gemm.hpp>
#include <ck/library/utility/check_err.hpp>
#include <ck/library/utility/device_memory.hpp>
#include <ck/library/utility/fill.hpp>
#include <ck/library/utility/host_tensor.hpp>
#include <ck/library/utility/host_tensor_generator.hpp>
#include <ck/library/utility/literals.hpp>

#include <ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp>

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

namespace at::native {

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
    typename ABLOCK_CLUSTER_LENS,
    typename ABLOCK_CLUSTER_ORDER,
    typename ABLOCK_SRC_ORDER,
    int ABLOCK_VECTOR_DIM,
    int ABLOCK_SCALAR_VEC,
    int ABLOCK_SCALAR_VEC_AK1,
    bool ABLOCK_LDS_EXTRAM,
    typename BBLOCK_CLUSTER_LENS,
    typename BBLOCK_CLUSTER_ORDER,
    typename BBLOCK_SRC_ORDER,
    int BBLOCK_VECTOR_DIM,
    int BBLOCK_SCALAR_VEC,
    int BBLOCK_SCALAR_VEC_AK1,
    bool BBLOCK_LDS_EXTRAN,
    int CMPER_WAVE,
    int CNPER_WAVE,
    typename BLOCK_CLUSTER_LENS,
    typename CDE_SCALAR_VEC,
    bool PADDING = false,
    bool TRANSA = false,
    bool TRANSB = false>
void gemm_impl(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  // Get input information.
  int M = m;
  int N = n;
  int K = k;

  int StrideA = lda;
  int StrideB = ldb;
  int StrideC = ldc;

  int KBatch = 1;

  float falpha = alpha;
  float fbeta = beta;

  using ADataType = typename CkMathType<Dtype>::dtype;
  using BDataType = typename CkMathType<Dtype>::dtype;
  using CDataType = typename CkMathType<Dtype>::dtype;
  using DDataType = typename CkMathType<Dtype>::dtype;

  using AccDataType = float;
  using CShuffleDataType = typename CkMathType<Dtype>::dtype;

  using ALayout = typename CkTensorLayout<TRANSA, TRANSB>::a_layout;
  using BLayout = typename CkTensorLayout<TRANSA, TRANSB>::b_layout;

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
                                                                   BLOCK_SIZE,
                                                                   MBLOCK,
                                                                   NBLOCK,
                                                                   KBLOCK,
                                                                   AK1,
                                                                   BK1,
                                                                   MPER_XDL,
                                                                   NPER_XDL,
                                                                   MPER_WAVE,
                                                                   NPER_WAVE,
                                                                   ABLOCK_CLUSTER_LENS,
                                                                   ABLOCK_CLUSTER_ORDER,
                                                                   ABLOCK_SRC_ORDER,
                                                                   ABLOCK_VECTOR_DIM,
                                                                   ABLOCK_SCALAR_VEC,
                                                                   ABLOCK_SCALAR_VEC_AK1,
                                                                   ABLOCK_LDS_EXTRAM,
                                                                   BBLOCK_CLUSTER_LENS,
                                                                   BBLOCK_CLUSTER_ORDER,
                                                                   BBLOCK_SRC_ORDER,
                                                                   BBLOCK_VECTOR_DIM,
                                                                   BBLOCK_SCALAR_VEC,
                                                                   BBLOCK_SCALAR_VEC_AK1,
                                                                   BBLOCK_LDS_EXTRAN,
                                                                   CMPER_WAVE,
                                                                   CNPER_WAVE,
                                                                   BLOCK_CLUSTER_LENS,
                                                                   CDE_SCALAR_VEC>;


  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto c_element_op = CElementOp{alpha, beta};


  using DDataArrayType = std::array<const void*, 0>;
  DDataArrayType DDataArray;

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
     KBatch,
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


template <
    typename Dtype,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int K1,
    int MPER_WMMA,
    int NPER_WMMA,
    int MPER_WAVE,
    int NPER_WAVE,
    typename ABLOCK_CLUSTER_LENS,
    typename ABLOCK_CLUSTER_ORDER,
    typename ABLOCK_SRC_ORDER,
    int ABLOCK_VECTOR_DIM,
    int ABLOCK_SCALAR_VEC,
    int ABLOCK_SCALAR_VEC_K1,
    bool ABLOCK_LDS_EXTRAM,
    typename BBLOCK_CLUSTER_LENS,
    typename BBLOCK_CLUSTER_ORDER,
    typename BBLOCK_SRC_ORDER,
    int BBLOCK_VECTOR_DIM,
    int BBLOCK_SCALAR_VEC,
    int BBLOCK_SCALAR_VEC_AK1,
    bool BBLOCK_LDS_EXTRAN,
    int CMPER_WAVE,
    int CNPER_WAVE,
    typename CBLOCK_CLUSTER_LENS,
    int CNPER_BLOCK,
    bool PADDING = false,
    bool TRANSA = false,
    bool TRANSB = false>
void gemm_impl_wmma(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  // Get input information.
  int M = m;
  int N = n;
  int K = k;

  int StrideA = lda;
  int StrideB = ldb;
  int StrideC = ldc;

  int KBatch = 1;

  float falpha = alpha;
  float fbeta = beta;

  using ADataType = typename CkMathType<Dtype>::dtype;
  using BDataType = typename CkMathType<Dtype>::dtype;
  using CDataType = typename CkMathType<Dtype>::dtype;
  using DDataType = typename CkMathType<Dtype>::dtype;

  using AccDataType = float;
  using CShuffleDataType = typename CkMathType<Dtype>::dtype;

  using ALayout = typename CkTensorLayout<TRANSA, TRANSB>::a_layout;
  using BLayout = typename CkTensorLayout<TRANSA, TRANSB>::b_layout;

  using DLayout = Row;
  using CLayout = Row;

  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CElementOp = PassThrough;


  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::Default;
  static constexpr auto GemmMNKPadding =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;
  static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;


  using DeviceGemmInstance =
            ck::tensor_operation::device::DeviceGemmWmma_CShuffle<ALayout,
                                                                  BLayout,
                                                                  CLayout,
                                                                  ADataType,
                                                                  BDataType,
                                                                  CDataType,
                                                                  AccDataType,
                                                                  CShuffleDataType,
                                                                  AElementOp,
                                                                  BElementOp,
                                                                  CElementOp,
                                                                  GemmSpec,
                                                                  1,   // NumPrefetch
                                                                  BLOCK_SIZE,
                                                                  MBLOCK,
                                                                  NBLOCK,
                                                                  KBLOCK,
                                                                  K1,
                                                                  MPER_WMMA,
                                                                  NPER_WMMA,
                                                                  MPER_WAVE,
                                                                  NPER_WAVE,
                                                                  ABLOCK_CLUSTER_LENS,
                                                                  ABLOCK_CLUSTER_ORDER,
                                                                  ABLOCK_SRC_ORDER,
                                                                  ABLOCK_VECTOR_DIM,
                                                                  ABLOCK_SCALAR_VEC,
                                                                  ABLOCK_SCALAR_VEC_K1,
                                                                  ABLOCK_LDS_EXTRAM,
                                                                  BBLOCK_CLUSTER_LENS,
                                                                  BBLOCK_CLUSTER_ORDER,
                                                                  BBLOCK_SRC_ORDER,
                                                                  BBLOCK_VECTOR_DIM,
                                                                  BBLOCK_SCALAR_VEC,
                                                                  BBLOCK_SCALAR_VEC_AK1,
                                                                  BBLOCK_LDS_EXTRAN,
                                                                  CMPER_WAVE,
                                                                  CNPER_WAVE,
                                                                  CBLOCK_CLUSTER_LENS,
                                                                  CNPER_BLOCK>;

  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto c_element_op = CElementOp{};


  using DDataArrayType = std::array<const void*, 0>;
  DDataArrayType DDataArray;

  // We swap A and B inputs here as a temporary workaround
  auto argument = gemm.MakeArgument(
     reinterpret_cast<const ADataType*>(b),
     reinterpret_cast<const BDataType*>(a),
     reinterpret_cast<CDataType*>(c),
     N,
     M,
     K,
     StrideB,
     StrideA,
     StrideC,
     b_element_op,
     a_element_op,
     c_element_op);


 if(!gemm.IsSupportedArgument(argument))
 {
        printf("error shape = %ld %ld %ld TRANSA=%d TRANSB=%d \n",
                        n, m, k,TRANSA, TRANSB);

        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
 }


 auto stream = at::cuda::getCurrentHIPStream().stream();
#if 1
 invoker.Run(argument, StreamConfig{stream, false});
#else
  float ave_time = invoker.Run(argument, StreamConfig{stream, true});
  std::size_t flop = std::size_t(2) * M * N * K;

  std::size_t num_btype =
              sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

  float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

  float gb_per_sec = num_btype / 1.E6 / ave_time;

  std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                          << gb_per_sec << " GB/s, " << N <<" " <<M<<" " << k <<" "
                          << "stride: "<<StrideA <<" "<<StrideB <<" "<<StrideC <<" "
                          <<  gemm.GetTypeString()
                          << std::endl;
#endif
}

} // namespace at::native
