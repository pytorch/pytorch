#undef __HIP_NO_HALF_CONVERSIONS__

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include <ck/library/utility/check_err.hpp>
#include <ck/library/utility/device_memory.hpp>
#include <ck/library/utility/host_tensor.hpp>
#include <ck/library/utility/host_tensor_generator.hpp>
#include <ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp>
#include <ck/library/utility/literals.hpp>

namespace at::native {

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = BF16;
using BDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

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
    typename CSHUFFLEBLOCK_TRANSFER,
    typename CDESHUFFLEBLOCK_TRANSFER,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::tensor_operation::device::GemmSpecialization GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::MNPadding>
using DeviceBGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
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
    BLOCK_SIZE,            // BlockSize
    MBLOCK,            // MPerBlock
    NBLOCK,            // NPerBlock
    KBLOCK,             // KPerBlock
    8,              // AK1
    8,              // BK1
    WAVE_TILE_M,             // MPerXDL
    WAVE_TILE_N,             // NPerXDL
    WAVE_MAP_M,              // MXdlPerWave
    WAVE_MAP_N,              // NXdlPerWave
    ABLOCK_TRANSFER,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_AK1
    0,              // ABlockLdsExtraM
    BBLOCK_TRANSFER,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    8,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_BK1
    0,              // BBlockLdsExtraN
    1,              // CShuffleMXdlPerWavePerShuffle
    1,              // CShuffleNXdlPerWavePerShuffle
    CSHUFFLEBLOCK_TRANSFER, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    CDESHUFFLEBLOCK_TRANSFER,           // CDEShuffleBlockTransferScalarPerVectors
    LOOP_SCHED, // BlockGemmPipelineScheduler
    PIPELINE_VERSION           // BlockGemmPipelineVersion
    >;

template <typename DeviceBGemmInstance>
void bgemm_kernel_impl(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // CK's A matrix is Pytorch's B matrix
  // CK's B matrix is Pytorch's A matrix

  // Create gemm launcher and arguments.
  auto gemm = DeviceBGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};

  // do GEMM
  auto argument = gemm.MakeArgument(
    b,
    a,
    {},
    c,
    n,
    m,
    k,
    num_batches,
    ldb,
    lda,
    {},
    ldc,
    n * k,  // batch_stride_a
    m * k,  // batch_stride_b
    {},
    m * n,  // batch_stride_c
    a_element_op,
    b_element_op,
    cde_element_op
  );

  if(!gemm.IsSupportedArgument(argument))
  {
      throw std::runtime_error(
          "wrong! device_gemm with the specified compilation parameters does "
          "not support this GEMM problem");
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});
}

}; // namespace at::native
