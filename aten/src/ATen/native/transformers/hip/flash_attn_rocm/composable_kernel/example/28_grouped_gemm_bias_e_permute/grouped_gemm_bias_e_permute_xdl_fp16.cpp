// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_contraction_multiple_d_xdl_cshuffle.hpp"
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

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F16;
using DDataType        = F16;
using DsDataType       = ck::Tuple<DDataType>;
using EDataType        = F16;

static constexpr ck::index_t NumDimM = 3;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 1;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::Add;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr auto ABSpec = ck::tensor_operation::device::TensorSpecialization::Packed;
static constexpr auto DESpec = ck::tensor_operation::device::TensorSpecialization::Packed;

// clang-format off
using DeviceOpInstanceKKNN = ck::tensor_operation::device::
        //############################################| NumDimM| NumDimN| NumDimK| AData| BData| AccData| CShuffle|     DsData| EData|            A|           B|          CDE|           Gemm|              A|              B|             DE| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################################|        |        |        |  Type|  Type|    Type| DataType|       Type|  Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Spacialization| Spacialization| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################################|        |        |        |      |      |        |         |           |      |    Operation|   Operation|    Operation|               |               |               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################################|        |        |        |      |      |        |         |           |      |             |            |             |               |               |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedContractionMultipleD_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F16,   F16,     F32,      F16, DsDataType,   F16,   AElementOp,  BElementOp, CDEElementOp,       GemmSpec,         ABSpec,         ABSpec,         DESpec,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,              S<1, 32, 1, 4>,               8>;
// clang-format on

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
          ck::enable_if_t<NumDimM == 3 && NumDimN == 2 && NumDimK == 1, bool> = false>
struct ReferenceContraction_M3_N2_K1 : public ck::tensor_operation::device::BaseOperator
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
        using Argument = ReferenceContraction_M3_N2_K1::Argument;

        float Run(const Argument& arg)
        {
            auto f_ms_ns = [&](auto m0, auto m1, auto m2, auto n0, auto n1) {
                const int K0 = arg.a_ms_ks_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    AccDataType v_a;
                    AccDataType v_b;

                    arg.a_element_op_(
                        v_a, ck::type_convert<const AccDataType>(arg.a_ms_ks_(m0, m1, m2, k0)));
                    arg.b_element_op_(
                        v_b, ck::type_convert<const AccDataType>(arg.b_ns_ks_(n0, n1, k0)));

                    v_acc += v_a * v_b;
                }

                AccDataType v_c;

                arg.cde_element_op_(v_c, v_acc);

                arg.e_ms_ns_(m0, m1, m2, n0, n1) = v_c;
            };

            make_ParallelTensorFunctor(f_ms_ns,
                                       arg.e_ms_ns_.mDesc.GetLengths()[0],
                                       arg.e_ms_ns_.mDesc.GetLengths()[1],
                                       arg.e_ms_ns_.mDesc.GetLengths()[2],
                                       arg.e_ms_ns_.mDesc.GetLengths()[3],
                                       arg.e_ms_ns_.mDesc.GetLengths()[4])(
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
        str << "ReferenceContraction_M3_N2_K1"
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

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        exit(0);
    }

    std::size_t group_count = rand() % 16 + 1;

    // GEMM shape
    std::vector<ck::tensor_operation::device::ContractionDesc<1>> contraction_descs;
    std::vector<const void*> p_a, p_b;
    std::vector<std::array<const void*, 1>> p_ds;
    std::vector<void*> p_c;

    contraction_descs.reserve(group_count);

    for(std::size_t i = 0; i < group_count; i++)
    {
        int M0 = 4 * (rand() % 4 + 1);
        int M1 = 4 * (rand() % 4 + 1);
        int M2 = 256;

        int N0 = 4 * (rand() % 4 + 1);
        int N1 = 128;

        int K0 = 64 * (rand() % 4 + 1);

        // A[M0, M1, M2, K0]
        std::vector<ck::index_t> a_ms_ks_lengths{M0, M1, M2, K0};
        std::vector<ck::index_t> a_ms_ks_strides{M1 * M2 * K0, M2 * K0, K0, 1};
        // B[N0, N1, K0]
        std::vector<ck::index_t> b_ns_ks_lengths{N0, N1, K0};
        std::vector<ck::index_t> b_ns_ks_strides{N1 * K0, K0, 1};
#if 0
        // D[M0, N0, M1, N1, M2]
        std::vector<ck::index_t> d_ms_ns_lengths{M0, M1, M2, N0, N1};
        std::vector<ck::index_t> d_ms_ns_strides{0, 0, 0, N1, 1};
        // E[M0, N0, M1, N1, M2]
        std::vector<ck::index_t> e_ms_ns_lengths{M0, M1, M2, N0, N1};
        std::vector<ck::index_t> e_ms_ns_strides{N0 * M1 * N1 * M2, N1 * M2, 1, M1 * N1 * M2, M2};
#else
        // D[M0, N0, M1, N1, M2]
        std::vector<ck::index_t> d_ms_ns_lengths{M0, M1, M2, N0, N1};
        std::vector<ck::index_t> d_ms_ns_strides{0, 0, 0, N1, 1};
        // E[M0, N0, M1, N1, M2]
        std::vector<ck::index_t> e_ms_ns_lengths{M0, M1, M2, N0, N1};
        std::vector<ck::index_t> e_ms_ns_strides{M1 * M2 * N0 * N1, M2 * N0 * N1, N0 * N1, N1, 1};
#endif

        contraction_descs.push_back(
            ck::tensor_operation::device::ContractionDesc<1>{a_ms_ks_lengths,
                                                             a_ms_ks_strides,
                                                             b_ns_ks_lengths,
                                                             b_ns_ks_strides,
                                                             {d_ms_ns_lengths},
                                                             {d_ms_ns_strides},
                                                             e_ms_ns_lengths,
                                                             e_ms_ns_strides});
    }

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<DDataType>> d_tensors;
    std::vector<Tensor<EDataType>> e_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    d_tensors.reserve(group_count);
    e_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, d_tensors_device,
        e_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    d_tensors_device.reserve(group_count);
    e_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(std::size_t i = 0; i < contraction_descs.size(); i++)
    {
        const auto a_ms_ks_lengths = contraction_descs[i].a_ms_ks_lengths;
        const auto a_ms_ks_strides = contraction_descs[i].a_ms_ks_strides;

        const auto b_ns_ks_lengths = contraction_descs[i].b_ns_ks_lengths;
        const auto b_ns_ks_strides = contraction_descs[i].b_ns_ks_strides;

        const auto d_ms_ns_lengths = contraction_descs[i].ds_ms_ns_lengths[0];
        const auto d_ms_ns_strides = contraction_descs[i].ds_ms_ns_strides[0];

        const auto e_ms_ns_lengths = contraction_descs[i].e_ms_ns_lengths;
        const auto e_ms_ns_strides = contraction_descs[i].e_ms_ns_strides;

        Tensor<ADataType> a_ms_ks(a_ms_ks_lengths, a_ms_ks_strides);
        Tensor<BDataType> b_ns_ks(b_ns_ks_lengths, b_ns_ks_strides);
        Tensor<DDataType> d_ms_ns(d_ms_ns_lengths, d_ms_ns_strides);
        Tensor<EDataType> e_ms_ns_device_result(e_ms_ns_lengths, e_ms_ns_strides);

        ck::index_t M_ =
            ck::accumulate_n<ck::index_t>(e_ms_ns_lengths.begin(), NumDimM, 1, std::multiplies<>{});

        ck::index_t N_ = ck::accumulate_n<ck::index_t>(
            e_ms_ns_lengths.begin() + NumDimM, NumDimN, 1, std::multiplies<>{});

        ck::index_t K_ = ck::accumulate_n<ck::index_t>(
            a_ms_ks_lengths.begin() + NumDimM, NumDimK, 1, std::multiplies<>{});

        a_tensors.push_back(a_ms_ks);
        b_tensors.push_back(b_ns_ks);
        d_tensors.push_back(d_ms_ns);

        // e_host_tensors.push_back(e_ms_ns_host_result);
        e_device_tensors.push_back(e_ms_ns_device_result);

        flop += std::size_t(2) * M_ * K_ * N_;

        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(EDataType) * e_device_tensors[i].mDesc.GetElementSize();

        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_n_k: " << b_tensors[i].mDesc << " c_m_n: " << e_device_tensors[i].mDesc
                  << std::endl;

        switch(init_method)
        {
        case 0: break;
        case 1:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
            d_tensors[i].GenerateTensorValue(GeneratorTensor_2<DDataType>{-5, 5});
            break;
        case 2:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
            d_tensors[i].GenerateTensorValue(GeneratorTensor_3<DDataType>{-0.5, 0.5});
            break;
        default:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_1<ADataType>{});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_1<BDataType>{});
            d_tensors[i].GenerateTensorValue(GeneratorTensor_1<DDataType>{});
        }
    }

    for(std::size_t i = 0; i < contraction_descs.size(); i++)
    {
        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ADataType) * a_tensors[i].mDesc.GetElementSpaceSize()));
        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(BDataType) * b_tensors[i].mDesc.GetElementSpaceSize()));
        d_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(DDataType) * d_tensors[i].mDesc.GetElementSpaceSize()));
        e_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(EDataType) * e_device_tensors[i].mDesc.GetElementSpaceSize()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());
        d_tensors_device[i]->ToDevice(d_tensors[i].mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_ds.push_back({d_tensors_device[i]->GetDeviceBuffer()});
        p_c.push_back(e_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    auto gemm    = DeviceOpInstanceKKNN{};
    auto invoker = gemm.MakeInvoker();

    // do GEMM
    auto argument = gemm.MakeArgument(
        p_a, p_b, p_ds, p_c, contraction_descs, a_element_op, b_element_op, cde_element_op);

    DeviceMem contraction_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, contraction_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;

    if(do_verification)
    {
        for(std::size_t i = 0; i < group_count; i++)
        {
            const auto e_ms_ns_lengths = contraction_descs[i].e_ms_ns_lengths;
            const auto e_ms_ns_strides = contraction_descs[i].e_ms_ns_strides;

            Tensor<EDataType> c_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides);

            Tensor<EDataType> e_ms_ns_host_result(e_ms_ns_lengths, e_ms_ns_strides);

            e_tensors_device[i]->FromDevice(e_device_tensors[i].mData.data());

            using ReferenceOpInstance = ReferenceContraction_M3_N2_K1<NumDimM,
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

            auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                      b_tensors[i],
                                                      c_ms_ns_host_result,
                                                      a_element_op,
                                                      b_element_op,
                                                      PassThrough{});

            ref_invoker.Run(ref_argument);

            for(size_t m0 = 0; m0 < e_ms_ns_host_result.mDesc.GetLengths()[0]; ++m0)
            {
                for(size_t m1 = 0; m1 < e_ms_ns_host_result.mDesc.GetLengths()[1]; ++m1)
                {
                    for(size_t m2 = 0; m2 < e_ms_ns_host_result.mDesc.GetLengths()[2]; ++m2)
                    {
                        for(size_t n0 = 0; n0 < e_ms_ns_host_result.mDesc.GetLengths()[3]; ++n0)
                        {
                            for(size_t n1 = 0; n1 < e_ms_ns_host_result.mDesc.GetLengths()[4]; ++n1)
                            {
                                cde_element_op(e_ms_ns_host_result(m0, m1, m2, n0, n1),
                                               c_ms_ns_host_result(m0, m1, m2, n0, n1),
                                               d_tensors[i](m0, m1, m2, n0, n1));
                            }
                        }
                    }
                }
            }

            pass &= ck::utils::check_err(e_device_tensors[i], e_ms_ns_host_result);
        }
    }

    return pass ? 0 : 1;
}
