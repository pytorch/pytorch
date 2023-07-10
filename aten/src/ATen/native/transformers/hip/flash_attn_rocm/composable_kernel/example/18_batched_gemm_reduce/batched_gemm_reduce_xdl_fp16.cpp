// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType         = F16;
using BDataType         = F16;
using CDataType         = F16;
using ReduceAccDataType = F32;
using ReduceDataType    = F32;
using ReducePtrsGlobal  = ck::Tuple<ReduceDataType*, ReduceDataType*>;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;
using ReduceOp0  = ck::reduce::Add;
using ReduceOp1  = ck::reduce::Add;
using ReduceOps  = ck::Tuple<ReduceOp0, ReduceOp1>;

using UnaryIdenticElementOp = ck::tensor_operation::element_wise::PassThrough;
using UnarySquareElementOp  = ck::tensor_operation::element_wise::UnarySquare;
using ReduceInElementOps    = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using ReduceOutElementOps   = ck::Tuple<UnaryIdenticElementOp, UnaryIdenticElementOp>;

using ReduceGlobalMemOps =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicAdd,
                                          ck::InMemoryDataOperationEnum::AtomicAdd>;

static constexpr auto GemmSpecialization =
    ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceBatchedGemmReduceInstance = ck::tensor_operation::device::DeviceBatchedGemmReduce_Xdl_CShuffle
//######| ALayout| BLayout| CLayout|AData| BData| CData|  GemmAcc| CShuffle| ReduceAcc|         DData|           A|           B|           C|         Dxs|     DxsInEleOp|     DxsAccEleOp|            D|               GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|              CReduce| CReduceThreadLds2VGprCopy| CReduceThreadVgpr2GlobalCopy|
//######|        |        |        | Type|  Type|  Type| DataType| DataType|  DataType|    Type Tuple| Elementwise| Elementwise| Elementwise|      Reduce|               |                |   MemoryData|     Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN| MXdlPerWave| NXdlPerWave|            _MBlock_MPerBlock| ScalarPerVector| ThreadClusterLengths|     SrcDstScalarPerVector|        SrcDstScalarPerVector|
//######|        |        |        |     |      |      |         |         |          |              |   Operation|   Operation|   Operation|   Operation|               |                |    Operation|                   |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|            _NBlock_NPerBlock|      _NPerBlock| _MPerBlock_NPerBlock|                _NPerBlock|                   _MPerBlock|
//######|        |        |        |     |      |      |         |         |          |              |            |            |            |            |               |                |             |                   |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                     |                          |                             |
        <     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   ReducePtrsGlobal,  AElementOp,  BElementOp,  CElementOp, ReduceOps, ReduceInElementOps, ReduceOutElementOps, ReduceGlobalMemOps, GemmSpecialization,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8,             S<64, 4>,                         4,                            1>;
// clang-format on

using ReferenceBatchedGemmInstance =
    ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     ReduceAccDataType,
                                                     AElementOp,
                                                     BElementOp,
                                                     CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 2048;
    ck::index_t N = 1920;
    ck::index_t K = 2048;

    ck::index_t StrideA = 2048;
    ck::index_t StrideB = 2048;
    ck::index_t StrideC = 1920;

    ck::index_t BatchCount = 4;

    if(argc == 1)
    {
        // do nothing
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 11)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);

        BatchCount = std::stoi(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 10: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC, BatchCount\n");
        exit(0);
    }

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       auto layout) {
        using namespace ck::literals;

        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor({batch_count, row, col}, {row * stride, stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({batch_count, row, col}, {col * stride, 1_uz, stride});
        }
    };

    Tensor<ADataType> a_g_m_k(f_host_tensor_descriptor(BatchCount, M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_g_k_n(f_host_tensor_descriptor(BatchCount, K, N, StrideB, BLayout{}));

    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> d0_g_m_host_result({BatchCount, M});
    Tensor<ReduceDataType> d1_g_m_host_result({BatchCount, M});

    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> d0_g_m_device_result({BatchCount, M});
    Tensor<ReduceDataType> d1_g_m_device_result({BatchCount, M});

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;
    std::cout << "d0_g_m: " << d0_g_m_host_result.mDesc << std::endl;
    std::cout << "d1_g_m: " << d1_g_m_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_g_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce0_device_buf(sizeof(ReduceDataType) *
                                 d0_g_m_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce1_device_buf(sizeof(ReduceDataType) *
                                 d1_g_m_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());

    auto a_element_op                     = AElementOp{};
    auto b_element_op                     = BElementOp{};
    auto c_element_op                     = CElementOp{};
    std::array<void*, 3> gemm_element_ops = {&a_element_op, &b_element_op, &c_element_op};

    auto passthrough                            = UnaryIdenticElementOp{};
    auto square                                 = UnarySquareElementOp{};
    std::array<void*, 2> reduce_in_element_ops  = {&passthrough, &square};
    std::array<void*, 2> reduce_out_element_ops = {&passthrough, &passthrough};

    std::array<void*, 2> p_reduces = {reduce0_device_buf.GetDeviceBuffer(),
                                      reduce1_device_buf.GetDeviceBuffer()};

    // do GEMM
    auto batched_gemm = DeviceBatchedGemmReduceInstance{};
    auto invoker      = batched_gemm.MakeInvoker();
    auto argument     = batched_gemm.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                              b_device_buf.GetDeviceBuffer(),
                                              nullptr,
                                              {},
                                              c_device_buf.GetDeviceBuffer(),
                                              p_reduces,
                                              M,
                                              N,
                                              K,
                                              StrideA,
                                              StrideB,
                                              StrideC,
                                              {},
                                              gemm_element_ops,
                                              {},
                                              reduce_in_element_ops,
                                              reduce_out_element_ops,
                                              BatchCount);

    if(!batched_gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    // init DO, D1 to 0
    reduce0_device_buf.SetZero();
    reduce1_device_buf.SetZero();

    // if time_kernel == true, kernel will run multiple times. This kernel use atomic-add so result
    // will not be correct. need to set time_kernel = false for correctness test
    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = std::size_t(2) * BatchCount * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * BatchCount * M * K +
                            sizeof(BDataType) * BatchCount * K * N +
                            sizeof(CDataType) * BatchCount * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << batched_gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_device_buf.FromDevice(c_g_m_n_device_result.mData.data());
        reduce0_device_buf.FromDevice(d0_g_m_device_result.mData.data());
        reduce1_device_buf.FromDevice(d1_g_m_device_result.mData.data());

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, c_g_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        auto reduce0_op = ReduceOp0{};
        auto reduce1_op = ReduceOp1{};

        for(int batch = 0; batch < BatchCount; ++batch)
        {
            for(int m = 0; m < M; ++m)
            {
                auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
                auto reduce1_acc = reduce1_op.GetIdentityValue<ReduceAccDataType>();

                for(int n = 0; n < N; ++n)
                {
                    auto c_val =
                        ck::type_convert<ReduceAccDataType>(c_g_m_n_host_result(batch, m, n));
                    ReduceAccDataType d0_val;
                    ReduceAccDataType d1_val;

                    UnaryIdenticElementOp{}(d0_val, c_val);
                    UnarySquareElementOp{}(d1_val, c_val);
                    reduce0_op(reduce0_acc, d0_val);
                    reduce1_op(reduce1_acc, d1_val);
                }

                d0_g_m_host_result(batch, m) = ck::type_convert<ReduceDataType>(reduce0_acc);
                d1_g_m_host_result(batch, m) = ck::type_convert<ReduceDataType>(reduce1_acc);
            }
        }

        pass = ck::utils::check_err(
                   c_g_m_n_host_result, c_g_m_n_device_result, "Error: Incorrect results c") &&
               ck::utils::check_err(d0_g_m_device_result,
                                    d0_g_m_host_result,
                                    "Error: Incorrect results! D0",
                                    1e-4,
                                    1e-5) &&
               ck::utils::check_err(d1_g_m_device_result,
                                    d1_g_m_host_result,
                                    "Error: Incorrect results! D1",
                                    1e-3,
                                    1e-5);
    }

    return pass ? 0 : 1;
}
