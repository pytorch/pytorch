// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/numeric.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F32;
using BDataType        = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = F32;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F32;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Bilinear;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
using DeviceOpInstanceKKNN = ck::tensor_operation::device::
        //#####################################| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //#####################################|        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //#####################################|        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //#####################################|        |        |        |      |      |        |         |           |      |             |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContractionMultipleD_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,     F32,      F32, DsDataType,   F32,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,               4>;

using DeviceOpInstanceKNNN = ck::tensor_operation::device::
        //#####################################| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //#####################################|        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //#####################################|        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //#####################################|        |        |        |      |      |        |         |           |      |             |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContractionMultipleD_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,     F32,      F32, DsDataType,   F32,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   256,   128,    16,   4,   1,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              1,         0,           1,           1,              S<1, 16, 1, 16>,               4>;

using DeviceOpInstanceMKNN = ck::tensor_operation::device::
        //#####################################| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //#####################################|        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //#####################################|        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //#####################################|        |        |        |      |      |        |         |           |      |             |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContractionMultipleD_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,     F32,      F32, DsDataType,   F32,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   256,   128,    16,   1,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              1,         0,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,               4>;

using DeviceOpInstanceMNNN = ck::tensor_operation::device::
        //#####################################| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //#####################################|        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //#####################################|        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //#####################################|        |        |        |      |      |        |         |           |      |             |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContractionMultipleD_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,     F32,      F32, DsDataType,   F32,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   256,   128,    16,   1,   1,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              1,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              1,         0,           1,           1,              S<1, 16, 1, 16>,               4>;
// clang-format on

using DeviceOpInstance = DeviceOpInstanceKKNN;

// hardcoded for NumDimM == NumDimN == NumDimK == 2
template <ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::enable_if_t<NumDimM == 2 && NumDimN == 2 && NumDimK == 2, bool> = false>
struct ReferenceContraction_M2_N2_K2 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_ms_ks,
                 const Tensor<BDataType>& b_ns_ks,
                 Tensor<EDataType>& e_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_ms_ks_{a_ms_ks},
              b_ns_ks_{b_ns_ks},
              e_ms_ns_{e_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
        }

        const Tensor<ADataType>& a_ms_ks_;
        const Tensor<BDataType>& b_ns_ks_;
        Tensor<EDataType>& e_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceContraction_M2_N2_K2::Argument;

        float Run(const Argument& arg)
        {
            auto f_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
                const int K0 = arg.a_ms_ks_.mDesc.GetLengths()[2];
                const int K1 = arg.a_ms_ks_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    for(int k1 = 0; k1 < K1; ++k1)
                    {
                        AccDataType v_a;
                        AccDataType v_b;

                        arg.a_element_op_(
                            v_a, ck::type_convert<const AccDataType>(arg.a_ms_ks_(m0, m1, k0, k1)));
                        arg.b_element_op_(
                            v_b, ck::type_convert<const AccDataType>(arg.b_ns_ks_(n0, n1, k0, k1)));

                        v_acc += v_a * v_b;
                    }
                }

                AccDataType v_c;

                arg.cde_element_op_(v_c, v_acc);

                arg.e_ms_ns_(m0, m1, n0, n1) = v_c;
            };

            make_ParallelTensorFunctor(f_ms_ns,
                                       arg.e_ms_ns_.mDesc.GetLengths()[0],
                                       arg.e_ms_ns_.mDesc.GetLengths()[1],
                                       arg.e_ms_ns_.mDesc.GetLengths()[2],
                                       arg.e_ms_ns_.mDesc.GetLengths()[3])(
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

    static auto MakeArgument(const Tensor<ADataType>& a_ms_ks,
                             const Tensor<BDataType>& b_ns_ks,
                             Tensor<EDataType>& e_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{a_ms_ks, b_ns_ks, e_ms_ns, a_element_op, b_element_op, cde_element_op};
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
        str << "ReferenceContraction_M2_N2_K2"
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

    // A[M0, M1, K0, K1]
    std::vector<ck::index_t> a_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a_ms_ks_strides{524288, 4096, 128, 1};
    // B[N0, N1, K0, K1]
    std::vector<ck::index_t> b_ns_ks_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b_ns_ks_strides{524288, 4096, 128, 1};
    // D[M0, M1, N0, N1]
    std::vector<ck::index_t> d_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> d_ms_ns_strides{524288, 4096, 128, 1};
    // E[M0, M1, N0, N1]
    std::vector<ck::index_t> e_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> e_ms_ns_strides{524288, 4096, 128, 1};

    float alpha = 1.f;
    float beta  = 1.f;

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
    else if(argc == 28)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        const ck::index_t M0 = std::stoi(argv[4]);
        const ck::index_t M1 = std::stoi(argv[5]);

        const ck::index_t N0 = std::stoi(argv[6]);
        const ck::index_t N1 = std::stoi(argv[7]);

        const ck::index_t K0 = std::stoi(argv[8]);
        const ck::index_t K1 = std::stoi(argv[9]);

        a_ms_ks_lengths = {M0, M1, K0, K1};
        a_ms_ks_strides = {
            std::stoi(argv[10]), std::stoi(argv[11]), std::stoi(argv[12]), std::stoi(argv[13])};

        b_ns_ks_lengths = {N0, N1, K0, K1};
        b_ns_ks_strides = {
            std::stoi(argv[14]), std::stoi(argv[15]), std::stoi(argv[16]), std::stoi(argv[17])};

        d_ms_ns_lengths = {M0, M1, N0, N1};
        d_ms_ns_strides = {
            std::stoi(argv[18]), std::stoi(argv[19]), std::stoi(argv[20]), std::stoi(argv[21])};

        e_ms_ns_lengths = {M0, M1, N0, N1};
        e_ms_ns_strides = {
            std::stoi(argv[22]), std::stoi(argv[23]), std::stoi(argv[24]), std::stoi(argv[25])};

        alpha = std::stof(argv[26]);
        beta  = std::stof(argv[27]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 7: M0, M1, N0, N1, K0, K1\n");
        printf("arg10 to 13: Stride_A_M0, Stride_A_M1, Stride_A_K0, Stride_A_K1\n");
        printf("arg14 to 17: Stride_B_N0, Stride_B_N1, Stride_B_K0, Stride_B_K1\n");
        printf("arg18 to 21: Stride_D_M0, Stride_D_M1, Stride_D_N0, Stride_D_N1\n");
        printf("arg22 to 25: Stride_E_M0, Stride_E_M1, Stride_E_N0, Stride_E_N1\n");
        printf("arg26 to 27: alpha, beta\n");
        exit(0);
    }

    Tensor<ADataType> a_ms_ks(a_ms_ks_lengths, a_ms_ks_strides);
    Tensor<BDataType> b_ns_ks(b_ns_ks_lengths, b_ns_ks_strides);
    Tensor<EDataType> d_ms_ns(d_ms_ns_lengths, d_ms_ns_strides);
    Tensor<EDataType> e_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides);
    Tensor<EDataType> e_ms_ns_device_result(e_ms_ns_lengths, e_ms_ns_strides);

    std::cout << "a_ms_ks: " << a_ms_ks.mDesc << std::endl;
    std::cout << "b_ns_ks: " << b_ns_ks.mDesc << std::endl;
    std::cout << "d_ms_ns: " << d_ms_ns.mDesc << std::endl;
    std::cout << "e_ms_ns: " << e_ms_ns_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_ns_ks.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        d_ms_ns.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_ns_ks.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        d_ms_ns.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d_ms_ns.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_ms_ns_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_ms_ks.mData.data());
    b_device_buf.ToDevice(b_ns_ks.mData.data());
    d_device_buf.ToDevice(d_ms_ns.mData.data());

    // set zero
    e_device_buf.SetZero();

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{alpha, beta};

    // device operation
    auto op       = DeviceOpInstance{};
    auto invoker  = op.MakeInvoker();
    auto argument = op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                    b_device_buf.GetDeviceBuffer(),
                                    std::array<const void*, 1>{d_device_buf.GetDeviceBuffer()},
                                    e_device_buf.GetDeviceBuffer(),
                                    a_ms_ks_lengths,
                                    a_ms_ks_strides,
                                    b_ns_ks_lengths,
                                    b_ns_ks_strides,
                                    std::array<std::vector<ck::index_t>, 1>{d_ms_ns_lengths},
                                    std::array<std::vector<ck::index_t>, 1>{d_ms_ns_strides},
                                    e_ms_ns_lengths,
                                    e_ms_ns_strides,
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
        ck::accumulate_n<ck::index_t>(e_ms_ns_lengths.begin(), NumDimM, 1, std::multiplies<>{});

    ck::index_t N = ck::accumulate_n<ck::index_t>(
        e_ms_ns_lengths.begin() + NumDimM, NumDimN, 1, std::multiplies<>{});

    ck::index_t K = ck::accumulate_n<ck::index_t>(
        a_ms_ks_lengths.begin() + NumDimM, NumDimK, 1, std::multiplies<>{});

    std::size_t flop      = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                            sizeof(DDataType) * M * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op.GetTypeString() << std::endl;

    e_device_buf.FromDevice(e_ms_ns_device_result.mData.data());

    if(do_verification)
    {
        Tensor<CShuffleDataType> c_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides);

        using ReferenceOpInstance = ReferenceContraction_M2_N2_K2<NumDimM,
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

        auto ref_argument = ref_gemm.MakeArgument(
            a_ms_ks, b_ns_ks, c_ms_ns_host_result, a_element_op, b_element_op, PassThrough{});

        ref_invoker.Run(ref_argument);

        for(size_t m0 = 0; m0 < e_ms_ns_host_result.mDesc.GetLengths()[0]; ++m0)
        {
            for(size_t m1 = 0; m1 < e_ms_ns_host_result.mDesc.GetLengths()[1]; ++m1)
            {
                for(size_t n0 = 0; n0 < e_ms_ns_host_result.mDesc.GetLengths()[2]; ++n0)
                {
                    for(size_t n1 = 0; n1 < e_ms_ns_host_result.mDesc.GetLengths()[3]; ++n1)
                    {
                        cde_element_op(e_ms_ns_host_result(m0, m1, n0, n1),
                                       c_ms_ns_host_result(m0, m1, n0, n1),
                                       d_ms_ns(m0, m1, n0, n1));
                    }
                }
            }
        }

        return ck::utils::check_err(e_ms_ns_device_result, e_ms_ns_host_result) ? 0 : 1;
    }

    return 0;
}
