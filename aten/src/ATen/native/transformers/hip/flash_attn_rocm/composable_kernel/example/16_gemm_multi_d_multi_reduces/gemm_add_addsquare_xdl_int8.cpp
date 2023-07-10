// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_reduce_xdl_common.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

// DataType
using ADataType         = INT8;
using BDataType         = INT8;
using GemmAccDataType   = INT32;
using CShuffleDataType  = INT32;
using DsDataType        = ck::Tuple<>;
using EDataType         = INT8;
using ReduceAccDataType = INT32;
using R0DataType        = INT32;
using R1DataType        = INT32;
using RsDataType        = ck::Tuple<R0DataType, R1DataType>;

// Layout
using ALayout = Row;
using BLayout = Col;
using ELayout = Row;

// Elementwise op
using Square       = ck::tensor_operation::element_wise::UnarySquare;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;
using QsElementOp  = ck::Tuple<PassThrough, Square>;
using RsElementOp  = ck::Tuple<PassThrough, PassThrough>;

// ReduceOp
using R0ThreadReduceOp = ck::reduce::Add;
using R1ThreadReduceOp = ck::reduce::Add;
using RsThreadReduceOp = ck::Tuple<R0ThreadReduceOp, R1ThreadReduceOp>;

static constexpr auto R0GlobalReduceOp = ck::InMemoryDataOperationEnum::AtomicAdd;
static constexpr auto R1GlobalReduceOp = ck::InMemoryDataOperationEnum::AtomicAdd;
using RsGlobalReduceOp = ck::InMemoryDataOperationEnumSequence<R0GlobalReduceOp, R1GlobalReduceOp>;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleDMultipleR_Xdl_CShuffle
        <ALayout,                   // ALayout
         BLayout,                   // BLayout
         ELayout,                   // ELayout
         ADataType,                 // ADataType
         BDataType,                 // BDataType
         GemmAccDataType,           // GemmAccDataType
         CShuffleDataType,          // CShuffleDataType
         DsDataType,                // DsDataType
         EDataType,                 // EDataType
         ReduceAccDataType,         // ReduceAccDataType
         RsDataType,                // RsDataType
         AElementOp,                // AElementwiseOperation
         BElementOp,                // BElementwiseOperation
         CDEElementOp,              // CDE ElementwiseOperation
         QsElementOp,               // Qs Elementwise Operation
         RsElementOp,               // Rs Elementwise Operation
         RsThreadReduceOp,          // Thread Reduce Operation
         RsGlobalReduceOp,          // Global Reduce Operation
         GemmDefault,               // GEMM Specialization
         1,                         // NumGemmKPrefetchStage
         256,                       // BlockSize
         256,                       // MPerBlock
         128,                       // NPerBlock
         64,                        // KPerBlock
         16,                        // AK1
         16,                        // BK1
         32,                        // MPerXdl
         32,                        // NPerXdl
         4,                         // MXdlPerWave
         2,                         // NXdlPerWave
         S<4, 64, 1>,               // ABlockTransfer ThreadCluster Lengths_K0_M_K1
         S<1, 0, 2>,                // ABlockTransfer ThreadCluster ArrangeOrder
         S<1, 0, 2>,                // ABlockTransfer SrcAccessOrder
         2,                         // ABlockTransfer SrcVectorDim
         16,                        // ABlockTransfer SrcScalarPerVector
         16,                        // ABlockTransfer DstScalarPerVector_K1
         1,                         // ABlockLdsExtraM
         S<4, 64, 1>,               // BBlockTransfer ThreadCluster Lengths_K0_N_K1
         S<1, 0, 2>,                // BBlockTransfer ThreadCluster ArrangeOrder
         S<1, 0, 2>,                // BBlockTransfer SrcAccessOrder
         2,                         // BBlockTransfer SrcVectorDim
         16,                        // BBlockTransfer SrcScalarPerVector
         16,                        // BBlockTransfer DstScalarPerVector_K1
         1,                         // BBlockLdsExtraN
         1,                         // CShuffleMXdlPerWavePerShuffle
         1,                         // CShuffleNXdlPerWavePerShuffle
         S<64, 4>,                  // CD Reduce Thread Transfer ClusterLengths _MPerBlock_NPerBlock
         4,                         // CDE ReduceThreadTransfer ScalarPerVector _NPerBlock
         1>;                        // RThread DstScalarPerVector _MPerBlock
// clang-format on
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        ReduceAccDataType,
                                                                        GemmAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CDEElementOp>;

using namespace ck::literals;

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename R0DataType,
          typename R1DataType,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          typename QsElementOp,
          typename RsElementOp,
          typename RsThreadReduceOp,
          typename ReduceAccDataType,
          typename DeviceOpInstance,
          typename ReferenceGemmInstance>
bool run_gemm_reduce_add_addsquare_xdl(ck::index_t M,
                                       ck::index_t N,
                                       ck::index_t K,
                                       ck::index_t StrideA,
                                       ck::index_t StrideB,
                                       ck::index_t StrideE,
                                       bool do_verification,
                                       int init_method,
                                       bool time_kernel)
{

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    auto f_host_tensor_descriptor2d =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<EDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> r0_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<R1DataType> r1_m(f_host_tensor_descriptor1d(M, 1));

    switch(init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n.mDesc.GetElementSpaceSize());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_m.mDesc.GetElementSpaceSize());
    DeviceMem r1_device_buf(sizeof(R1DataType) * r1_m.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{};

    // Prepare GEMM, add, add_square
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                               b_device_buf.GetDeviceBuffer(),
                               {},
                               e_device_buf.GetDeviceBuffer(),
                               {r0_device_buf.GetDeviceBuffer(), r1_device_buf.GetDeviceBuffer()},
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               {},
                               StrideE,
                               a_element_op,
                               b_element_op,
                               cde_element_op,
                               qs_element_op,
                               rs_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    // init reducetion buffer to 0
    r0_device_buf.SetZero();
    r1_device_buf.SetZero();

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass = true;

    if(do_verification)
    {
        auto I0 = ck::Number<0>{};
        auto I1 = ck::Number<1>{};

        Tensor<ReduceAccDataType> e_m_n_host(e_m_n.mDesc);
        Tensor<R0DataType> r0_m_host(r0_m.mDesc);
        Tensor<R1DataType> r1_m_host(r1_m.mDesc);

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, e_m_n_host, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        auto reduce0_op = RsThreadReduceOp{}[I0];
        auto reduce1_op = RsThreadReduceOp{}[I1];

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.template GetIdentityValue<ReduceAccDataType>();
            auto reduce1_acc = reduce1_op.template GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                ReduceAccDataType square_e_val;
                auto e_val = ck::type_convert<ReduceAccDataType>(e_m_n_host(m, n));
                qs_element_op[I1](square_e_val, e_val);

                reduce0_op(reduce0_acc, e_val);
                reduce1_op(reduce1_acc, square_e_val);
            }

            r0_m_host(m) = ck::type_convert<R0DataType>(reduce0_acc);
            r1_m_host(m) = ck::type_convert<R1DataType>(reduce1_acc);
        }
        e_device_buf.FromDevice(e_m_n.mData.data());

        Tensor<EDataType> e_m_n_host_converted(e_m_n_host);

        pass = ck::utils::check_err(
            e_m_n, e_m_n_host_converted, "Error: Incorrect results c", 1e-2, 1e-2);

        r0_device_buf.FromDevice(r0_m.mData.data());
        r1_device_buf.FromDevice(r1_m.mData.data());

        pass &= ck::utils::check_err(r0_m, r0_m_host, "Error: Incorrect results d0", 1e-2, 1e-2);
        pass &= ck::utils::check_err(r1_m, r1_m_host, "Error: Incorrect results d1", 1e-2, 1e-2);

        if(pass)
        {
            std::cout << "Success!" << std::endl;
        }
    }

    if(time_kernel)
    {
        float ave_time            = invoker.Run(argument, StreamConfig{nullptr, time_kernel});
        std::size_t flop          = 2_uz * M * N * K + 3_uz * M * N;
        std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                    sizeof(EDataType) * M * N + sizeof(R0DataType) * M +
                                    sizeof(R1DataType) * M;

        float tflops          = static_cast<float>(flop) / 1.E9 / ave_time;
        float gemm_gb_per_sec = gemm_num_byte / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gemm_gb_per_sec
                  << " GB/s, " << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;

    // GEMM shape
    ck::index_t M = 1024;
    ck::index_t N = 1152;
    ck::index_t K = 512;

    ck::index_t StrideA = 512;
    ck::index_t StrideB = 512;
    ck::index_t StrideE = 1152;

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
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideE = std::stoi(argv[9]);
    }
    else
    {
        std::cout << "arg1: verification (0=no, 1=yes)\n"
                  << " arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
                  << " arg3: Measure kernel execution time (1=ON, 0=Off)\n"
                  << " arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideE\n"
                  << std::endl;
        exit(EXIT_SUCCESS);
    }

    return !run_gemm_reduce_add_addsquare_xdl<ADataType,
                                              BDataType,
                                              EDataType,
                                              R0DataType,
                                              R1DataType,
                                              ALayout,
                                              BLayout,
                                              ELayout,
                                              AElementOp,
                                              BElementOp,
                                              CDEElementOp,
                                              QsElementOp,
                                              RsElementOp,
                                              RsThreadReduceOp,
                                              ReduceAccDataType,
                                              DeviceOpInstance,
                                              ReferenceGemmInstance>(
        M, N, K, StrideA, StrideB, StrideE, do_verification, init_method, time_kernel);
}
