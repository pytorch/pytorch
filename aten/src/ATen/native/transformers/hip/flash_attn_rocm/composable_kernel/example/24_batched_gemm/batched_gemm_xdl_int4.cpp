#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/literals.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = ck::int4_t;
using BDataType        = ck::int4_t;
using AccDataType      = int32_t;
using CShuffleDataType = int32_t;
using DsDataType       = ck::Tuple<>;
using EDataType        = ck::int4_t;

using KernelADataType = int8_t;
using KernelBDataType = int8_t;
using KernelEDataType = int8_t;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl
    // clang-format off
        < ALayout,              //ALayout
          BLayout,              //BLayout
          DsLayout,             //DsLayout
          ELayout,              //ELayout
          KernelADataType,      //ADataType    
          KernelBDataType,      //BDataType   
          AccDataType,          //AccDataType
          CShuffleDataType,     //CShuffleDataType
          DsDataType,           //DsDataType
          KernelEDataType,      //EDataType
          AElementOp,           //AElementwiseOperation
          BElementOp,           //BElementwiseOperation
          CDEElementOp,         //CDEElementwiseOperation
          GemmDefault,          //GEMMSpecialization
          1,                    // NumGemmKPrefetchStage
          256,                  // BlockSize
          256,                  // MPerBlock
          128,                  // NPerBlock
          64,                   // KPerBlock
          16,                   // AK1
          16,                   // BK1
          32,                   // MPerXdl
          32,                   // NPerXdl
          4,                    // MXdlPerWave
          2,                    // NXdlPerWave
          S<4, 64, 1>,          // ABlockTransfer ThreadCluster Lengths_K0_M_K1
          S<1, 0, 2>,           // ABlockTransfer ThreadCluster ArrangeOrder
          S<1, 0, 2>,           // ABlockTransfer SrcAccessOrder
          2,                    // ABlockTransfer SrcVectorDim
          16,                   // ABlockTransfer SrcScalarPerVector
          16,                   // ABlockTransfer DstScalarPerVector_K1
          1,                    // ABlockLdsExtraM
          S<4, 64, 1>,          // BBlockTransfer ThreadCluster Lengths_K0_N_K1
          S<1, 0, 2>,           // BBlockTransfer ThreadCluster ArrangeOrder
          S<1, 0, 2>,           // BBlockTransfer SrcAccessOrder
          2,                    // BBlockTransfer SrcVectorDim
          16,                   // BBlockTransfer SrcScalarPerVector
          16,                   // BBlockTransfer DstScalarPerVector_K1
          1,                    // BBlockLdsExtraN
          1,                    // CShuffleMXdlPerWavePerShuffle
          1,                    // CShuffleNXdlPerWavePerShuffle
          S<1, 64, 1, 4>,       // CBlockTransferClusterLengths_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl
          16>;                  // CBlockTransferScalarPerVector_NWaveNPerXdl
// clang-format on

#define BUILD_INT4_EXAMPLE
#include "run_batched_gemm_example.inc"

int main(int argc, char* argv[]) { return !run_batched_gemm_example(argc, argv); }
