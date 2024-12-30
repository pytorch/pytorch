#undef __HIP_NO_HALF_CONVERSIONS__


#include <ATen/OpMathType.h>
#include <ATen/hip/HIPBlas.h>
#include <ATen/native/hip/ck_types.h>

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

using AccDataType      = F32;
using DsDataType       = ck::Tuple<>;
using CDataType        = BF16;
using CShuffleDataType = BF16;
using DsLayout         = ck::Tuple<>;
using CLayout          = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <
    typename A_DATA_TYPE,
    typename B_DATA_TYPE,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int AK1,
    int BK1,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    int ABLOCK_TRANSFER_SSPV,
    int ABLOCK_TRANSFER_DSPV_K1,
    typename BBLOCK_TRANSFER,
    int BBLOCK_TRANSFER_SSPV,
    int BBLOCK_TRANSFER_SSPV_K1,
    int CSHUFFLE_MXDL_PWPS,
    int CSHUFFLE_NXDL_PWPS,
    typename CSHUFFLEBLOCK_TRANSFER,
    typename CDESHUFFLEBLOCK_TRANSFER,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::tensor_operation::device::GemmSpecialization GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::MNPadding,
    bool TRANSA = false,
    bool TRANSB = false>
void bgemm_kernel_impl(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {

  using ADataType = typename CkMathType<A_DATA_TYPE>::dtype;
  using BDataType = typename CkMathType<B_DATA_TYPE>::dtype;

  using ALayout = typename CkTensorLayout<TRANSA, TRANSB>::a_layout;
  using BLayout = typename CkTensorLayout<TRANSA, TRANSB>::b_layout;

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};

  auto gemm = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
      ALayout,                  // ALayout
      BLayout,                  // BLayout
      DsLayout,                 // DsLayout
      CLayout,                  // CLayout
      ADataType,                // ADataType
      BDataType,                // BDataType
      DsDataType,               // DsDataType
      CDataType,                // CDataType
      AccDataType,              // AccDataType
      CShuffleDataType,         // CshuffleType
      AElementOp,               // AElementwiseOperation
      BElementOp,               // BElementwiseOperation
      CDEElementOp,             // CElementwiseOperation
      GEMM_SPEC,                // GEMMSpecialization
      BLOCK_SIZE,               // BlockSize
      MBLOCK,                   // MPerBlock
      NBLOCK,                   // NPerBlock
      KBLOCK,                   // KPerBlock
      AK1,                      // AK1
      BK1,                      // BK1
      WAVE_TILE_M,              // MPerXDL
      WAVE_TILE_N,              // NPerXDL
      WAVE_MAP_M,               // MXdlPerWave
      WAVE_MAP_N,               // NXdlPerWave
      ABLOCK_TRANSFER,          // ABlockTransferThreadClusterLengths_AK0_M_AK1
      S<1, 0, 2>,               // ABlockTransferThreadClusterArrangeOrder
      S<1, 0, 2>,               // ABlockTransferSrcAccessOrder
      2,                        // ABlockTransferSrcVectorDim
      ABLOCK_TRANSFER_SSPV,     // ABlockTransferSrcScalarPerVector
      ABLOCK_TRANSFER_DSPV_K1,  // ABlockTransferDstScalarPerVector_AK1
      0,                        // ABlockLdsExtraM
      BBLOCK_TRANSFER,          // BBlockTransferThreadClusterLengths_BK0_N_BK1
      S<1, 0, 2>,               // BBlockTransferThreadClusterArrangeOrder
      S<1, 0, 2>,               // BBlockTransferSrcAccessOrder
      2,                        // BBlockTransferSrcVectorDim
      BBLOCK_TRANSFER_SSPV,     // BBlockTransferSrcScalarPerVector
      BBLOCK_TRANSFER_SSPV_K1,  // BBlockTransferDstScalarPerVector_BK1
      0,                        // BBlockLdsAddExtraN
      CSHUFFLE_MXDL_PWPS,       // CShuffleMXdlPerWavePerShuffle
      CSHUFFLE_NXDL_PWPS,       // CShuffleNXdlPerWavePerShuffle
      CSHUFFLEBLOCK_TRANSFER,   // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
      CDESHUFFLEBLOCK_TRANSFER, // CDEShuffleBlockTransferScalarPerVectors
      LOOP_SCHED,               // BlockGemmPipelineScheduler
      PIPELINE_VERSION          // BlockGemmPipelineVersion
      >{};
  auto invoker = gemm.MakeInvoker();
  auto argument = gemm.MakeArgument(
    b, // A and B are swapped for CK
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
