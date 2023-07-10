// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>

#include "ck/ck.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_layernorm_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm_layernorm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

// This example demonstrate a single kernel that runs GEMM layer and laynorm in one fused kernel
//
// The GEMM + Layernorm implementation is a specialized kernel which allows fusing both layers
// together given the condition GEMM extents N of MNK is spanned by a single workgroup. For example,
// a kernel configured with NPerBlock = 128 allows to operate on all GEMM sizes if N <= 128
//
// D = Layernorm(acc_element_op(A * B + broadcast(bias)) + add) * broadcast(gamma) + broadcast(beta)
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType        = F16;
using BDataType        = F16;
using CDataType        = F16;
using C0DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

struct Relu
{
    template <typename OutT, typename InT>
    __host__ __device__ void operator()(OutT& y, const InT& x) const
    {
        y = x > 0 ? x : 0;
    }
};

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
// Elementwise operation that operates on the output of matrix multiplication
// i.e., AccElementOp(A * B + bias)
using AccElementOp = Relu;
// Elementwise operation that operates on the output of layer normalization
using CElementOp = Relu;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmLayerNorm_Xdl_CShuffle
//######| ALayout| BLayout| CLayout|     AData|     BData|     CData|     C0Data|     GemmAcc|         CShuffle|   ReduceAcc|           A|           B|          Acc|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|              CReduce|     CReduceThreadCopy|
//######|        |        |        |      Type|      Type|      Type|       Type|    DataType|         DataType|    DataType| Elementwise| Elementwise|  Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN| MXdlPerWave| NXdlPerWave|            _MBlock_MPerBlock| ScalarPerVector| ThreadClusterLengths| SrcDstScalarPerVector|
//######|        |        |        |          |          |          |           |            |                 |            |   Operation|   Operation|    Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|            _NBlock_NPerBlock|      _NPerBlock| _MPerBlock_NPerBlock|            _NPerBlock|
//######|        |        |        |          |          |          |           |            |                 |            |            |            |             |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |                     |                      |
        <     Row,     Col,     Row, ADataType, BDataType, CDataType, C0DataType, AccDataType, CShuffleDataType, AccDataType,  AElementOp,  BElementOp, AccElementOp,  CElementOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           2,               S<1, 32, 1, 8>,               8,             S<64, 4>,                     4>;
// clang-format on

using ReferenceInstance = ck::tensor_operation::host::ReferenceGemmLayernorm<ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             C0DataType,
                                                                             AccDataType,
                                                                             AElementOp,
                                                                             BElementOp,
                                                                             AccElementOp,
                                                                             CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 128;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = 128;

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
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
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

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<AccDataType> acc_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<C0DataType> c0_n_bias({N});
    Tensor<C0DataType> c0_m_n_add(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<C0DataType> c0_n_gamma({N});
    Tensor<C0DataType> c0_n_beta({N});

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "c0_n_bias: " << c0_n_bias.mDesc << std::endl;
    std::cout << "c0_m_n_add: " << c0_m_n_add.mDesc << std::endl;
    std::cout << "c0_n_gamma: " << c0_n_gamma.mDesc << std::endl;
    std::cout << "c0_n_beta: " << c0_n_beta.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_Sequential<0>{});
        b_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
    }

    c0_n_bias.GenerateTensorValue(GeneratorTensor_2<C0DataType>{-5, 5});
    c0_m_n_add.GenerateTensorValue(GeneratorTensor_2<C0DataType>{-5, 5});
    c0_n_gamma.GenerateTensorValue(GeneratorTensor_2<C0DataType>{0, 2});
    c0_n_beta.GenerateTensorValue(GeneratorTensor_2<C0DataType>{0, 5});
    c_m_n_host_result.GenerateTensorValue(GeneratorTensor_1<CDataType>{0});
    acc_m_n_host_result.GenerateTensorValue(GeneratorTensor_1<AccDataType>{0});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem c0_bias_buf(sizeof(C0DataType) * c0_n_bias.mDesc.GetElementSpaceSize());
    DeviceMem c0_add_buf(sizeof(C0DataType) * c0_m_n_add.mDesc.GetElementSpaceSize());
    DeviceMem c0_gamma_buf(sizeof(C0DataType) * c0_n_gamma.mDesc.GetElementSpaceSize());
    DeviceMem c0_beta_buf(sizeof(C0DataType) * c0_n_beta.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    c0_bias_buf.ToDevice(c0_n_bias.mData.data());
    c0_add_buf.ToDevice(c0_m_n_add.mData.data());
    c0_gamma_buf.ToDevice(c0_n_gamma.mData.data());
    c0_beta_buf.ToDevice(c0_n_beta.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto acc_element_op = AccElementOp{};
    auto c_element_op   = CElementOp{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                      static_cast<C0DataType*>(c0_add_buf.GetDeviceBuffer()),
                                      static_cast<C0DataType*>(c0_bias_buf.GetDeviceBuffer()),
                                      static_cast<C0DataType*>(c0_gamma_buf.GetDeviceBuffer()),
                                      static_cast<C0DataType*>(c0_beta_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      a_element_op,
                                      b_element_op,
                                      acc_element_op,
                                      c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    // extra 6MN flops due to: bias + add + gamma + beta + norm_sub + norm_div,
    // excluding reduction steps
    std::size_t flop = std::size_t(2) * M * N * K + std::size_t(6) * M * N;
    // extra MN and 3N due to c0_add (MxN), bias (1xN), gamma (1xN), beta (1xN)
    std::size_t bytes = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                        sizeof(CDataType) * 2 * M * N + sizeof(C0DataType) * 3 * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = bytes / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_device_buf.FromDevice(c_m_n_device_result.mData.data());

        auto ref_gemm    = ReferenceInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  b_k_n,
                                                  c_m_n_host_result,
                                                  c0_n_bias,
                                                  c0_m_n_add,
                                                  c0_n_gamma,
                                                  c0_n_beta,
                                                  a_element_op,
                                                  b_element_op,
                                                  acc_element_op,
                                                  c_element_op);

        ref_invoker.Run(ref_argument);

        if constexpr(std::is_same<CShuffleDataType, F32>::value)
        {
            pass &= ck::utils::check_err(
                c_m_n_device_result, c_m_n_host_result, "Error: Incorrect results c");
        }
        else if constexpr(std::is_same<CShuffleDataType, F16>::value)
        {
            pass &= ck::utils::check_err(
                c_m_n_device_result, c_m_n_host_result, "Error: Incorrect results c", 1e-2, 1e-2);
        }
    }
    return pass ? 0 : 1;
}
