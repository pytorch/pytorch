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
#include "ck/tensor_operation/gpu/device/impl/device_elementwise.hpp"
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
using ADataType                = F16;
using BDataType                = F16;
using GemmAccDataType          = F32;
using CShuffleDataType         = F32;
using D0DataType               = F16;
using D1DataType               = F16;
using DsDataType               = ck::Tuple<D0DataType, D1DataType>;
using EDataType                = F16;
using ReduceAccDataType        = F32;
using R0DataType               = F32;
using R1DataType               = F32;
using RsDataType               = ck::Tuple<R0DataType, R1DataType>;
using GammaDataType            = F16;
using BetaDataType             = F16;
using LayerNormOutDataType     = F16;
using NormalizeComputeDataType = F32;

// Layout
using ALayout  = Row;
using BLayout  = Col;
using D1Layout = Row;
using ELayout  = D1Layout;

// Elementwise op
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AddReluAdd   = ck::tensor_operation::element_wise::AddReluAdd;
using Square       = ck::tensor_operation::element_wise::UnarySquare;
using Div          = ck::tensor_operation::element_wise::UnaryDivide;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddReluAdd;
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

using NormalizeFunctor = ck::tensor_operation::element_wise::Normalize;

// A:x, B:E[x], C:E[x^2], D:Gamma, E:Beta , F:y
using DeviceNormalizeInstance = ck::tensor_operation::device::DeviceElementwise<
    ck::Tuple<EDataType,
              R0DataType,
              R1DataType,
              GammaDataType,
              BetaDataType>,         // x(gemm_out), mean, meansquare, gamma, beta
    ck::Tuple<LayerNormOutDataType>, // y
    NormalizeFunctor,
    2,
    8,                           // MPerthread
    ck::Sequence<8, 1, 1, 8, 8>, // scalarPerVector: x(gemm_out), mean, meansquare, gamma, beta
    ck::Sequence<8>>;            // scalarPerVector: y(layerNorm_out)

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

void host_gemm_layernorm(Tensor<LayerNormOutDataType>& out_m_n,
                         const Tensor<ADataType>& a_m_k,
                         const Tensor<BDataType>& b_k_n,
                         const Tensor<D0DataType>& bias_n,
                         const Tensor<D1DataType>& d1_m_n,
                         const Tensor<GammaDataType>& gamma_n,
                         const Tensor<BetaDataType>& beta_n,
                         AElementOp a_element_op,
                         BElementOp b_element_op,
                         CDEElementOp cde_element_op,
                         int M,
                         int N)
{

    int StrideE = N;
    Tensor<EDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> mean_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<R1DataType> meanSquare_m(f_host_tensor_descriptor1d(M, 1));
    auto averageOpInst = Div{N};

    auto ref_gemm    = ReferenceGemmInstance{};
    auto ref_invoker = ref_gemm.MakeInvoker();

    auto ref_argument =
        ref_gemm.MakeArgument(a_m_k, b_k_n, e_m_n, a_element_op, b_element_op, PassThrough{});

    ref_invoker.Run(ref_argument);

    // c = activation(c + bias) + c1_functor(c1)
    for(int m = 0; m < M; ++m)
        for(int n = 0; n < N; ++n)
        {
            auto acc = ck::type_convert<GemmAccDataType>(e_m_n(m, n));
            cde_element_op(e_m_n(m, n), acc, bias_n(n), d1_m_n(m, n));
        }

    // reduce_mean and reduce_square_mean
    auto r0Op = R0ThreadReduceOp{};
    auto r1Op = R1ThreadReduceOp{};
    for(int m = 0; m < M; ++m)
    {
        auto mean_acc        = r0Op.GetIdentityValue<ReduceAccDataType>();
        auto mean_square_acc = r1Op.GetIdentityValue<ReduceAccDataType>();

        for(int n = 0; n < N; ++n)
        {
            auto e_val                     = ck::type_convert<ReduceAccDataType>(e_m_n(m, n));
            ReduceAccDataType square_e_val = 0;
            Square{}(square_e_val, e_val);

            r0Op(mean_acc, e_val);
            r1Op(mean_square_acc, square_e_val);
        }

        averageOpInst(mean_acc, mean_acc);
        averageOpInst(mean_square_acc, mean_square_acc);
        mean_m(m)       = ck::type_convert<R0DataType>(mean_acc);
        meanSquare_m(m) = ck::type_convert<R1DataType>(mean_square_acc);
    }

    // LayerNorm
    auto layerNormInst = NormalizeFunctor{};
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            LayerNormOutDataType out_val = 0;
            layerNormInst(out_val, e_m_n(m, n), mean_m(m), meanSquare_m(m), gamma_n(n), beta_n(n));
            out_m_n(m, n) = out_val;
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename D0DataType,
          typename D1DataType,
          typename R0DataType,
          typename R1DataType,
          typename GammaDataType,
          typename BetaDataType,
          typename NormalizeDataType>
void DumpGemmLayerNormPerf(float gemm_reduce_time, float normalize_time, int M, int N, int K)
{
    std::size_t gemm_flop     = std::size_t(2) * M * N * K + std::size_t(2) * M * N;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(EDataType) * M * N + sizeof(D0DataType) * M * N +
                                sizeof(D0DataType) * M * N + sizeof(R0DataType) * M +
                                sizeof(R1DataType) * M;

    std::size_t normalize_num_byte = sizeof(EDataType) * M * N + sizeof(R0DataType) * M +
                                     sizeof(R1DataType) * M + sizeof(GammaDataType) * N +
                                     sizeof(BetaDataType) * N + sizeof(NormalizeDataType) * M * N;

    float tflops               = static_cast<float>(gemm_flop) / 1.E9 / gemm_reduce_time;
    float gemm_gb_per_sec      = gemm_num_byte / 1.E6 / gemm_reduce_time;
    float normalize_gb_per_sec = normalize_num_byte / 1.E6 / normalize_time;

    std::cout << "gemm + reduce_mean + reduce_square_mean Perf: " << gemm_reduce_time << " ms, "
              << tflops << " TFlops, " << gemm_gb_per_sec << " GB/s, " << std::endl;

    std::cout << "5-ary elementwise Perf: " << normalize_time << " ms, " << normalize_gb_per_sec
              << " GB/s, " << std::endl;
}

int main()
{
    // GEMM shape
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
    Tensor<D0DataType> bias_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<D1DataType> d1_m_n(f_host_tensor_descriptor2d(M, N, StrideD1, ELayout{}));
    Tensor<EDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> r0_Mean_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<R1DataType> r1_MeanSquare_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<GammaDataType> gamma_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<BetaDataType> beta_n(f_host_tensor_descriptor1d(N, 1));
    Tensor<LayerNormOutDataType> layerNorm_m_n(
        f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-1, 1});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-1, 1});
    bias_n.GenerateTensorValue(GeneratorTensor_3<D0DataType>{-1, 1});
    d1_m_n.GenerateTensorValue(GeneratorTensor_3<D1DataType>{-5, 5});
    gamma_n.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-1, 1});
    beta_n.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-1, 1});

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem bias_device_buf(sizeof(D0DataType) * bias_n.mDesc.GetElementSpaceSize());
    DeviceMem d1_device_buf(sizeof(D1DataType) * d1_m_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n.mDesc.GetElementSpaceSize());
    DeviceMem r0_Mean_device_buf(sizeof(R0DataType) * r0_Mean_m.mDesc.GetElementSpaceSize());
    DeviceMem r1_MeanSquare_device_buf(sizeof(R1DataType) *
                                       r1_MeanSquare_m.mDesc.GetElementSpaceSize());
    DeviceMem gamma_device_buf(sizeof(GammaDataType) * gamma_n.mDesc.GetElementSpaceSize());
    DeviceMem beta_device_buf(sizeof(BetaDataType) * beta_n.mDesc.GetElementSpaceSize());
    DeviceMem layerNorm_device_buf(sizeof(LayerNormOutDataType) *
                                   layerNorm_m_n.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    bias_device_buf.ToDevice(bias_n.mData.data());
    d1_device_buf.ToDevice(d1_m_n.mData.data());
    gamma_device_buf.ToDevice(gamma_n.mData.data());
    beta_device_buf.ToDevice(beta_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{N, N};

    // Prepare GEMM, mean, mean_square
    auto gemmReduce          = DeviceOpInstance{};
    auto gemmReduce_invoker  = gemmReduce.MakeInvoker();
    auto gemmReduce_argument = gemmReduce.MakeArgument(
        a_device_buf.GetDeviceBuffer(),
        b_device_buf.GetDeviceBuffer(),
        {bias_device_buf.GetDeviceBuffer(), d1_device_buf.GetDeviceBuffer()},
        e_device_buf.GetDeviceBuffer(),
        {r0_Mean_device_buf.GetDeviceBuffer(), r1_MeanSquare_device_buf.GetDeviceBuffer()},
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

    if(!gemmReduce.IsSupportedArgument(gemmReduce_argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    // init reducetion buffer to 0
    r0_Mean_device_buf.SetZero();
    r1_MeanSquare_device_buf.SetZero();

    // Prepare LayerNorm
    std::array<const void*, 5> input = {e_device_buf.GetDeviceBuffer(),
                                        r0_Mean_device_buf.GetDeviceBuffer(),
                                        r1_MeanSquare_device_buf.GetDeviceBuffer(),
                                        gamma_device_buf.GetDeviceBuffer(),
                                        beta_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {layerNorm_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 2> xyLengths = {M, N};
    std::array<ck::index_t, 2> xyStrides = {StrideE, 1};

    auto normalize         = DeviceNormalizeInstance{};
    auto normalize_invoker = normalize.MakeInvoker();
    auto normalize_argument_ptr =
        normalize.MakeArgumentPointer(xyLengths,
                                      {xyStrides, {1, 0}, {1, 0}, {0, 1}, {0, 1}},
                                      {xyStrides},
                                      input,
                                      output,
                                      NormalizeFunctor{});

    if(!normalize.IsSupportedArgument(normalize_argument_ptr.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device, exiting!");
    }

    // run kernel
    gemmReduce_invoker.Run(gemmReduce_argument, StreamConfig{nullptr, false});
    normalize_invoker.Run(normalize_argument_ptr.get(), StreamConfig{nullptr, false});

    bool pass = true;
    {
        // verification
        Tensor<LayerNormOutDataType> host_layerNorm_m_n(
            f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));

        host_gemm_layernorm(host_layerNorm_m_n,
                            a_m_k,
                            b_k_n,
                            bias_n,
                            d1_m_n,
                            gamma_n,
                            beta_n,
                            a_element_op,
                            b_element_op,
                            cde_element_op,
                            M,
                            N);

        layerNorm_device_buf.FromDevice(layerNorm_m_n.mData.data());
        pass &= ck::utils::check_err(layerNorm_m_n,
                                     host_layerNorm_m_n,
                                     "Error: Incorrect results layerNorm_m_n",
                                     1e-2,
                                     1e-2);
    }

    {
        // evaluate kernel perf
        bool time_kernel = true;

        float gemm_reduce_mean_reduce_square_mean_ave_time =
            gemmReduce_invoker.Run(gemmReduce_argument, StreamConfig{nullptr, time_kernel});
        float normalize_ave_time =
            normalize_invoker.Run(normalize_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        if(time_kernel)
            DumpGemmLayerNormPerf<ADataType,
                                  BDataType,
                                  EDataType,
                                  D0DataType,
                                  D1DataType,
                                  R0DataType,
                                  R1DataType,
                                  GammaDataType,
                                  BetaDataType,
                                  LayerNormOutDataType>(
                gemm_reduce_mean_reduce_square_mean_ave_time, normalize_ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}
