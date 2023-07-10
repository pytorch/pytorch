// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/numeric.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Add         = ck::tensor_operation::element_wise::Add;

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using DDataType        = F16;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F16;

static constexpr ck::index_t NumDimG = 1;
static constexpr ck::index_t NumDimM = 3;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 1;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Add;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr auto ABSpec = ck::tensor_operation::device::TensorSpecialization::Packed;
static constexpr auto DESpec = ck::tensor_operation::device::TensorSpecialization::Default;

// clang-format off
using DeviceOpInstanceKKNN = ck::tensor_operation::device::
        //############################################| NumDimG| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           Gemm|              A|              B|             DE| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################################|        |        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Spacialization| Spacialization| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################################|        |        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |               |               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################################|        |        |        |        |      |      |        |         |           |      |             |            |             |               |               |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceBatchedContractionMultipleD_Xdl_CShuffle< NumDimG, NumDimM, NumDimN, NumDimK,   F16,   F16,     F32,      F16, DsDataType,   F16,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,         ABSpec,         ABSpec,         DESpec,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<1, 32, 1, 4>,               8>;
// clang-format on

using DeviceOpInstance = DeviceOpInstanceKKNN;

template <ck::index_t NumDimG,
          ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::enable_if_t<NumDimG == 1 && NumDimM == 3 && NumDimN == 2 && NumDimK == 1, bool> =
              false>
struct ReferenceContraction_G1_M3_N2_K1 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_gs_ms_ks,
                 const Tensor<BDataType>& b_gs_ns_ks,
                 Tensor<EDataType>& e_gs_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_gs_ms_ks_{a_gs_ms_ks},
              b_gs_ns_ks_{b_gs_ns_ks},
              e_gs_ms_ns_{e_gs_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
        }

        const Tensor<ADataType>& a_gs_ms_ks_;
        const Tensor<BDataType>& b_gs_ns_ks_;
        Tensor<EDataType>& e_gs_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceContraction_G1_M3_N2_K1::Argument;

        float Run(const Argument& arg)
        {
            auto f_gs_ms_ns = [&](auto g0, auto m0, auto m1, auto m2, auto n0, auto n1) {
                const int K0 = arg.a_gs_ms_ks_.mDesc.GetLengths()[4];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    AccDataType v_a;
                    AccDataType v_b;

                    arg.a_element_op_(
                        v_a,
                        ck::type_convert<const AccDataType>(arg.a_gs_ms_ks_(g0, m0, m1, m2, k0)));
                    arg.b_element_op_(
                        v_b, ck::type_convert<const AccDataType>(arg.b_gs_ns_ks_(g0, n0, n1, k0)));

                    v_acc += v_a * v_b;
                }

                AccDataType v_c;

                arg.cde_element_op_(v_c, v_acc);

                arg.e_gs_ms_ns_(g0, m0, m1, m2, n0, n1) = v_c;
            };

            make_ParallelTensorFunctor(f_gs_ms_ns,
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[0],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[1],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[2],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[3],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[4],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[5])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const ck::tensor_operation::device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const ck::tensor_operation::device::BaseArgument*) override
    {
        return true;
    }

    static auto MakeArgument(const Tensor<ADataType>& a_gs_ms_ks,
                             const Tensor<BDataType>& b_gs_ns_ks,
                             Tensor<EDataType>& e_gs_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{
            a_gs_ms_ks, b_gs_ns_ks, e_gs_ms_ns, a_element_op, b_element_op, cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<ck::tensor_operation::device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceContraction_G1_M3_N2_K1"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    ck::index_t G0 = 1;

    ck::index_t M0 = 4;
    ck::index_t M1 = 8;
    ck::index_t M2 = 256;

    ck::index_t N0 = 32;
    ck::index_t N1 = 128;

    ck::index_t K0 = 1024;

    // A[M0, M1, M2, K0]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, M0, M1, M2, K0};
    std::vector<ck::index_t> a_gs_ms_ks_strides{M0 * M1 * M2 * K0, M1 * M2 * K0, M2 * K0, K0, 1};

    // B[N0, N1, K0]
    std::vector<ck::index_t> b_gs_ns_ks_lengths{G0, N0, N1, K0};
    std::vector<ck::index_t> b_gs_ns_ks_strides{N0 * N1 * K0, N1 * K0, K0, 1};

    // D[M0, N0, M1, N1, M2]
    std::vector<ck::index_t> d_gs_ms_ns_lengths{G0, M0, M1, M2, N0, N1};
    std::vector<ck::index_t> d_gs_ms_ns_strides{N0 * N1, 0, 0, 0, N1, 1};

    // E[M1, M0, N0, M1, N1]
    std::vector<ck::index_t> e_gs_ms_ns_lengths{G0, M0, M1, M2, N0, N1};
    std::vector<ck::index_t> e_gs_ms_ns_strides{
        M0 * M1 * M2 * N1 * N0, N0 * M1 * N1, N1, M0 * N0 * M1 * N1, M1 * N1, 1};

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<BDataType> b_gs_ns_ks(b_gs_ns_ks_lengths, b_gs_ns_ks_strides);
    Tensor<DDataType> d_gs_ms_ns(d_gs_ms_ns_lengths, d_gs_ms_ns_strides);
    Tensor<EDataType> e_gs_ms_ns_host_result(e_gs_ms_ns_lengths, e_gs_ms_ns_strides);
    Tensor<EDataType> e_gs_ms_ns_device_result(e_gs_ms_ns_lengths, e_gs_ms_ns_strides);

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b_gs_ns_ks: " << b_gs_ns_ks.mDesc << std::endl;
    std::cout << "d_gs_ms_ns: " << d_gs_ms_ns.mDesc << std::endl;
    std::cout << "e_gs_ms_ns: " << e_gs_ms_ns_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_gs_ms_ns.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_gs_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) *
                           e_gs_ms_ns_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b_device_buf.ToDevice(b_gs_ns_ks.mData.data());
    d_device_buf.ToDevice(d_gs_ms_ns.mData.data());

    // set zero
    e_device_buf.SetZero();

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // device operation
    auto op       = DeviceOpInstance{};
    auto invoker  = op.MakeInvoker();
    auto argument = op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                    b_device_buf.GetDeviceBuffer(),
                                    std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                                    e_device_buf.GetDeviceBuffer(),
                                    a_gs_ms_ks_lengths,
                                    a_gs_ms_ks_strides,
                                    b_gs_ns_ks_lengths,
                                    b_gs_ns_ks_strides,
                                    std::array<std::vector<ck::index_t>, 1>{d_gs_ms_ns_lengths},
                                    std::array<std::vector<ck::index_t>, 1>{d_gs_ms_ns_strides},
                                    e_gs_ms_ns_lengths,
                                    e_gs_ms_ns_strides,
                                    a_element_op,
                                    b_element_op,
                                    cde_element_op);

    if(!op.IsSupportedArgument(argument))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    ck::index_t M =
        ck::accumulate_n<ck::index_t>(e_gs_ms_ns_lengths.begin(), NumDimM, 1, std::multiplies<>{});

    ck::index_t N = ck::accumulate_n<ck::index_t>(
        e_gs_ms_ns_lengths.begin() + NumDimM, NumDimN, 1, std::multiplies<>{});

    ck::index_t K = ck::accumulate_n<ck::index_t>(
        a_gs_ms_ks_lengths.begin() + NumDimM, NumDimK, 1, std::multiplies<>{});

    std::size_t flop      = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                            sizeof(DDataType) * M * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op.GetTypeString() << std::endl;

    e_device_buf.FromDevice(e_gs_ms_ns_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_gs_ms_ns_host_result(e_gs_ms_ns_lengths, e_gs_ms_ns_strides);

        using ReferenceOpInstance = ReferenceContraction_G1_M3_N2_K1<NumDimG,
                                                                     NumDimM,
                                                                     NumDimN,
                                                                     NumDimK,
                                                                     ADataType,
                                                                     BDataType,
                                                                     CShuffleDataType,
                                                                     AccDataType,
                                                                     AElementOp,
                                                                     BElementOp,
                                                                     PassThrough>;

        auto ref_gemm    = ReferenceOpInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_gs_ms_ks,
                                                  b_gs_ns_ks,
                                                  c_gs_ms_ns_host_result,
                                                  a_element_op,
                                                  b_element_op,
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        for(size_t g0 = 0; g0 < e_gs_ms_ns_host_result.mDesc.GetLengths()[0]; ++g0)
        {
            for(size_t m0 = 0; m0 < e_gs_ms_ns_host_result.mDesc.GetLengths()[1]; ++m0)
            {
                for(size_t m1 = 0; m1 < e_gs_ms_ns_host_result.mDesc.GetLengths()[2]; ++m1)
                {
                    for(size_t m2 = 0; m2 < e_gs_ms_ns_host_result.mDesc.GetLengths()[3]; ++m2)
                    {
                        for(size_t n0 = 0; n0 < e_gs_ms_ns_host_result.mDesc.GetLengths()[4]; ++n0)
                        {
                            for(size_t n1 = 0; n1 < e_gs_ms_ns_host_result.mDesc.GetLengths()[5];
                                ++n1)
                            {
                                cde_element_op(e_gs_ms_ns_host_result(g0, m0, m1, m2, n0, n1),
                                               c_gs_ms_ns_host_result(g0, m0, m1, m2, n0, n1),
                                               d_gs_ms_ns(g0, m0, m1, m2, n0, n1));
                            }
                        }
                    }
                }
            }
        }

        return ck::utils::check_err(e_gs_ms_ns_device_result, e_gs_ms_ns_host_result) ? 0 : 1;
    }

    return 0;
}
