// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

// DataType
using ADataType         = F16;
using BDataType         = F16;
using GemmAccDataType   = F32;
using CShuffleDataType  = F32;
using D0DataType        = F16;
using D1DataType        = F16;
using DsDataType        = ck::Tuple<D0DataType, D1DataType>;
using EDataType         = F16;
using ReduceAccDataType = F32;
using R0DataType        = F32;
using R1DataType        = F32;
using RsDataType        = ck::Tuple<R0DataType, R1DataType>;

// Layout
using ALayout  = Row;
using BLayout  = Col;
using D1Layout = Row;
using ELayout  = D1Layout;

// Elementwise op
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AddAdd       = ck::tensor_operation::element_wise::AddAdd;
using Square       = ck::tensor_operation::element_wise::UnarySquare;
using Div          = ck::tensor_operation::element_wise::UnaryDivide;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddAdd;
using QsElementOp  = ck::Tuple<PassThrough, Square>;
using RsElementOp  = ck::Tuple<Div, Div>;

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
//######| ALayout| BLayout| ELayout|     AData|     BData|     GemmAccData|         CShuffle|     DsData|     EData|     ReduceAccData|     RsData|           A|           B|          CDE|          Qs|          Rs|           Thread|           Global|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|    CDRThreadTransfer|                  CDE|    RThreadTransfer|
//######|        |        |        |      Type|      Type|            Type|         DataType|       Type|      Type|              Type|       Type| Elementwise| Elementwise|  Elementwise| Elementwise| Elementwise|           Reduce|           Reduce| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|       ClusterLengths| ReduceThreadTransfer| DstScalarPerVector|
//######|        |        |        |          |          |                |                 |           |          |                  |           |   Operation|   Operation|    Operation|   Operation|   Operation|        Operation|        Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _MPerBlock_NPerBlock|      ScalarPerVector|         _MPerBlock|
//######|        |        |        |          |          |                |                 |           |          |                  |           |            |            |             |            |            |                 |                 |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                     |           _NPerBlock|                   |
        < ALayout, BLayout, ELayout, ADataType, BDataType, GemmAccDataType, CShuffleDataType, DsDataType, EDataType, ReduceAccDataType, RsDataType,  AElementOp,  BElementOp, CDEElementOp, QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<64, 4>,                    4,                  1>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        EDataType,
                                                                        GemmAccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        PassThrough>;

template <typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType,
          typename R0DataType,
          typename R1DataType>
void DumpPerf(float ave_time, int M, int N, int K)
{
    std::size_t flop          = std::size_t(2) * M * N * K + std::size_t(2) * M * N;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(D0DataType) * M * N + sizeof(D1DataType) * M * N +
                                sizeof(EDataType) * M * N + sizeof(R0DataType) * M +
                                sizeof(R1DataType) * M;

    float tflops          = static_cast<float>(flop) / 1.E9 / ave_time;
    float gemm_gb_per_sec = gemm_num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gemm_gb_per_sec
              << " GB/s, " << std::endl;
}

auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
    return HostTensorDescriptor({len}, {stride});
};

auto f_host_tensor_descriptor2d =
    [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
        using namespace ck::literals;

        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor({row, col}, {stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({row, col}, {1_uz, stride});
        }
    };

int main()
{
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA  = 1024;
    ck::index_t StrideB  = 1024;
    ck::index_t StrideD0 = 0;
    ck::index_t StrideD1 = 1024;
    ck::index_t StrideE  = 1024;

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<D0DataType> d0_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor2d(M, N, StrideD1, D1Layout{}));
    Tensor<EDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> r0_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<R1DataType> r1_m(f_host_tensor_descriptor1d(M, 1));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-1, 1});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-1, 1});
    d0_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-1, 1});
    d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{-1, 1});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem d0_device_buf(sizeof(D0DataType) * d0_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n.mDesc.GetElementSpaceSize());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_m.mDesc.GetElementSpaceSize());
    DeviceMem r1_device_buf(sizeof(R1DataType) * r1_m.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    d0_device_buf.ToDevice(d0_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{N, N};

    // Prepare GEMM, mean, mean_square
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                               b_device_buf.GetDeviceBuffer(),
                               {d0_device_buf.GetDeviceBuffer(), d1_device_buf.GetDeviceBuffer()},
                               e_device_buf.GetDeviceBuffer(),
                               {r0_device_buf.GetDeviceBuffer(), r1_device_buf.GetDeviceBuffer()},
                               M,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               {StrideD0, StrideD1},
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

    bool do_verification = true;
    bool pass            = true;

    if(do_verification)
    {
        auto I0 = ck::Number<0>{};
        auto I1 = ck::Number<1>{};

        Tensor<EDataType> e_m_n_host(e_m_n.mDesc);
        Tensor<R0DataType> r0_m_host(r0_m.mDesc);
        Tensor<R1DataType> r1_m_host(r1_m.mDesc);

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, e_m_n_host, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        auto reduce0_op = R0ThreadReduceOp{};
        auto reduce1_op = R1ThreadReduceOp{};

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
            auto reduce1_acc = reduce1_op.GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                ReduceAccDataType square_e_val;

                auto e_val  = ck::type_convert<GemmAccDataType>(e_m_n_host(m, n));
                auto d0_val = ck::type_convert<GemmAccDataType>(d0_n(n));
                auto d1_val = ck::type_convert<GemmAccDataType>(d1_m_n(m, n));
                cde_element_op(e_val, e_val, d0_val, d1_val);
                e_m_n_host(m, n) = ck::type_convert<EDataType>(e_val);

                auto e_val_reduce = ck::type_convert<ReduceAccDataType>(e_val);
                qs_element_op[I1](square_e_val, e_val_reduce);

                reduce0_op(reduce0_acc, e_val_reduce);
                reduce1_op(reduce1_acc, square_e_val);
            }

            rs_element_op[I0](reduce0_acc, reduce0_acc);
            rs_element_op[I1](reduce1_acc, reduce1_acc);
            r0_m_host(m) = ck::type_convert<R0DataType>(reduce0_acc);
            r1_m_host(m) = ck::type_convert<R1DataType>(reduce1_acc);
        }

        e_device_buf.FromDevice(e_m_n.mData.data());
        r0_device_buf.FromDevice(r0_m.mData.data());
        r1_device_buf.FromDevice(r1_m.mData.data());

        pass = ck::utils::check_err(e_m_n, e_m_n_host, "Error: Incorrect results c", 1e-2, 1e-2);
        pass &= ck::utils::check_err(r0_m, r0_m_host, "Error: Incorrect results d0", 1e-2, 1e-2);
        pass &= ck::utils::check_err(r1_m, r1_m_host, "Error: Incorrect results d1", 1e-2, 1e-2);
    }

    bool time_kernel = true;
    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});
        DumpPerf<ADataType, BDataType, D0DataType, D1DataType, EDataType, R0DataType, R1DataType>(
            ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}
