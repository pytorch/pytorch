// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;
using QsElementOp  = ck::Tuple<PassThrough>;
using RsElementOp  = ck::Tuple<PassThrough>;

// ReduceOp
using RsThreadReduceOp = ck::Tuple<ck::reduce::Max>;

using RsGlobalReduceOp =
    ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicMax>;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
template <ck::index_t NDimSpatial>
using DeviceInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleDMultipleR_Xdl_CShuffle
//######| NDimSpatial|              ALayout|              BLayout|              DELayout|              RLayout|           AData|           BData|     AccData|         CShuffle|     DsData|     EData|     ReduceAccData|     RsData|           A|           B|          CDE|          Qs|          Rs|           Thread|           Global|           Conv|          GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|    CDRThreadTransfer|                  CDE|    RThreadTransfer|
//######|            |                     |                     |                      |                     |            Type|            Type|        Type|         DataType|       Type|      Type|              Type|       Type| Elementwise| Elementwise|  Elementwise| Elementwise| Elementwise|           Reduce|           Reduce|            Fwd|Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|       ClusterLengths| ReduceThreadTransfer| DstScalarPerVector|
//######|            |                     |                     |                      |                     |                |                |            |                 |           |          |                  |           |   Operation|   Operation|    Operation|   Operation|   Operation|        Operation|        Operation| Specialization|              |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _MPerBlock_NPerBlock|      ScalarPerVector|         _MPerBlock|
//######|            |                     |                     |                      |                     |                |                |            |                 |           |          |                  |           |            |            |             |            |            |                 |                 |               |              |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                     |           _NPerBlock|                   |
#ifdef BUILD_INT4_EXAMPLE
        < NDimSpatial, ALayout<NDimSpatial>, BLayout<NDimSpatial>, DELayout<NDimSpatial>, RLayout<NDimSpatial>, KernelADataType, KernelBDataType, AccDataType, CShuffleDataType, DsDataType, EDataType, ReduceAccDataType, RsDataType,  AElementOp,  BElementOp, CDEElementOp, QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp,       ConvSpec,   GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<64, 4>,                    4,                  1>;
#else
        < NDimSpatial, ALayout<NDimSpatial>, BLayout<NDimSpatial>, DELayout<NDimSpatial>, RLayout<NDimSpatial>,       ADataType,       BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType, ReduceAccDataType, RsDataType,  AElementOp,  BElementOp, CDEElementOp, QsElementOp, RsElementOp, RsThreadReduceOp, RsGlobalReduceOp,       ConvSpec,   GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<64, 4>,                    4,                  1>;
#endif

template <ck::index_t NDimSpatial>
using HostInstance = ck::tensor_operation::host::ReferenceConvFwd
        <NDimSpatial, ADataType, BDataType, EDataType, AElementOp, BElementOp, PassThrough>;
// clang-format on

template <ck::index_t NDimSpatial>
bool run_convnd_fwd_max(const ck::utils::conv::ConvParam& problem_size,
                        const ExecutionConfig& config)
{
    static_assert(1 <= NDimSpatial && NDimSpatial <= 3, "Unsupported NDimSpatial");

#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
#endif

    const auto conv_input_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ALayout<NDimSpatial>>(
            problem_size);

    const auto conv_weight_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<BLayout<NDimSpatial>>(
            problem_size);

    const auto conv_output_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<DELayout<NDimSpatial>>(
            problem_size);

    const auto r0_desc = make_r0_host_tensor_descriptor(problem_size);

    Tensor<ADataType> conv_input(conv_input_g_n_c_wis_desc);
    Tensor<BDataType> conv_weight(conv_weight_g_k_c_xs_desc);
    Tensor<EDataType> conv_output_device(conv_output_g_n_k_wos_desc);
    Tensor<R0DataType> r0_device(r0_desc);

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-8, 7}(conv_input);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-8, 7}(conv_weight);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-5, 5}(conv_input);
        ck::utils::FillUniformDistribution<BDataType>{-5, 5}(conv_weight);
    }

    DeviceMem conv_input_device_buf(sizeof(ADataType) * conv_input.mDesc.GetElementSpaceSize());
    DeviceMem conv_weight_device_buf(sizeof(BDataType) * conv_weight.mDesc.GetElementSpaceSize());
    DeviceMem conv_output_device_buf(sizeof(EDataType) *
                                     conv_output_device.mDesc.GetElementSpaceSize());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_device.mDesc.GetElementSpaceSize());

#ifdef BUILD_INT4_EXAMPLE
    const Tensor<KernelADataType> conv_input_converted(conv_input);
    const Tensor<KernelBDataType> conv_weight_converted(conv_weight);

    conv_input_device_buf.ToDevice(conv_input_converted.mData.data());
    conv_weight_device_buf.ToDevice(conv_weight_converted.mData.data());
#else
    conv_input_device_buf.ToDevice(conv_input.mData.data());
    conv_weight_device_buf.ToDevice(conv_weight.mData.data());
#endif

    std::array<ck::index_t, NDimSpatial + 3> conv_input_g_n_c_wis_lengths{},
        conv_input_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> conv_weight_g_k_c_xs_lengths{},
        conv_weight_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> conv_output_g_n_k_wos_lengths{},
        conv_output_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 2> r0_lengths{}, r0_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{}, conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{}, input_right_pads{};

    unpack_host_tensor_descriptor(
        conv_input_g_n_c_wis_desc, conv_input_g_n_c_wis_lengths, conv_input_g_n_c_wis_strides);
    unpack_host_tensor_descriptor(
        conv_weight_g_k_c_xs_desc, conv_weight_g_k_c_xs_lengths, conv_weight_g_k_c_xs_strides);
    unpack_host_tensor_descriptor(
        conv_output_g_n_k_wos_desc, conv_output_g_n_k_wos_lengths, conv_output_g_n_k_wos_strides);
    unpack_host_tensor_descriptor(r0_desc, r0_lengths, r0_strides);

    ck::ranges::copy(problem_size.conv_filter_strides_, begin(conv_filter_strides));
    ck::ranges::copy(problem_size.conv_filter_dilations_, begin(conv_filter_dilations));
    ck::ranges::copy(problem_size.input_left_pads_, begin(input_left_pads));
    ck::ranges::copy(problem_size.input_right_pads_, begin(input_right_pads));

    // run Conv + Reduction on device
    auto conv     = DeviceInstance<NDimSpatial>{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(conv_input_device_buf.GetDeviceBuffer(),
                                      conv_weight_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      conv_output_device_buf.GetDeviceBuffer(),
                                      {r0_device_buf.GetDeviceBuffer()},
                                      conv_input_g_n_c_wis_lengths,
                                      conv_input_g_n_c_wis_strides,
                                      conv_weight_g_k_c_xs_lengths,
                                      conv_weight_g_k_c_xs_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{{}},
                                      conv_output_g_n_k_wos_lengths,
                                      conv_output_g_n_k_wos_strides,
                                      r0_lengths,
                                      r0_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      AElementOp{},
                                      BElementOp{},
                                      CDEElementOp{},
                                      QsElementOp{},
                                      RsElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        std::cerr << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return false;
    }

    const float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    const std::size_t flop      = problem_size.GetFlops();
    const std::size_t num_btype = problem_size.GetByte<ADataType, BDataType, EDataType>();

    const float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    const float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        Tensor<EDataType> conv_output_host(conv_output_g_n_k_wos_desc);

        // run Conv + Reduction on host
        auto ref_conv     = HostInstance<NDimSpatial>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(conv_input,
                                                  conv_weight,
                                                  conv_output_host,
                                                  problem_size.conv_filter_strides_,
                                                  problem_size.conv_filter_dilations_,
                                                  problem_size.input_left_pads_,
                                                  problem_size.input_right_pads_,
                                                  AElementOp{},
                                                  BElementOp{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        Tensor<R0DataType> r0_host(r0_device.mDesc);

        auto reduce0_op = RsThreadReduceOp{}[ck::Number<0>{}];

        auto& output_dims = conv_output_g_n_k_wos_desc.GetLengths();

        if constexpr(NDimSpatial == 1)
        {
            for(std::size_t g = 0; g < output_dims[0]; ++g)
            {
                for(std::size_t n = 0; n < output_dims[1]; ++n)
                {
                    for(std::size_t w = 0; w < output_dims[3]; ++w)
                    {
                        auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
                        for(std::size_t k = 0; k < output_dims[2]; ++k)
                        {

                            auto e_val =
                                ck::type_convert<ReduceAccDataType>(conv_output_host(g, n, k, w));
                            reduce0_op(reduce0_acc, e_val);
                        }
                        r0_host(g, n, w) = ck::type_convert<R0DataType>(reduce0_acc);
                    }
                }
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            for(std::size_t g = 0; g < output_dims[0]; ++g)
            {
                for(std::size_t n = 0; n < output_dims[1]; ++n)
                {
                    for(std::size_t h = 0; h < output_dims[3]; ++h)
                    {
                        for(std::size_t w = 0; w < output_dims[4]; ++w)
                        {
                            auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
                            for(std::size_t k = 0; k < output_dims[2]; ++k)
                            {

                                auto e_val = ck::type_convert<ReduceAccDataType>(
                                    conv_output_host(g, n, k, h, w));
                                reduce0_op(reduce0_acc, e_val);
                            }
                            r0_host(g, n, h, w) = ck::type_convert<R0DataType>(reduce0_acc);
                        }
                    }
                }
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            for(std::size_t g = 0; g < output_dims[0]; ++g)
            {
                for(std::size_t n = 0; n < output_dims[1]; ++n)
                {
                    for(std::size_t d = 0; d < output_dims[3]; ++d)
                    {
                        for(std::size_t h = 0; h < output_dims[4]; ++h)
                        {
                            for(std::size_t w = 0; w < output_dims[5]; ++w)
                            {
                                auto reduce0_acc = reduce0_op.GetIdentityValue<ReduceAccDataType>();
                                for(std::size_t k = 0; k < output_dims[2]; ++k)
                                {

                                    auto e_val = ck::type_convert<ReduceAccDataType>(
                                        conv_output_host(g, n, k, d, h, w));
                                    reduce0_op(reduce0_acc, e_val);
                                }
                                r0_host(g, n, d, h, w) = ck::type_convert<R0DataType>(reduce0_acc);
                            }
                        }
                    }
                }
            }
        }

        conv_output_device_buf.FromDevice(conv_output_device.mData.data());
        r0_device_buf.FromDevice(r0_device.mData.data());

        return ck::utils::check_err(conv_output_device,
                                    conv_output_host,
                                    "Error: incorrect results! (Matrix E)",
                                    1e-5f,
                                    1e-4f) &&
               ck::utils::check_err(
                   r0_device, r0_host, "Error: incorrect results! (Matrix R0)", 1e-5f, 1e-4f);
    }

    return true;
}

bool run_convnd_fwd_max_example(int argc, char* argv[])
{
    ck::utils::conv::ConvParam problem_size{
        2, 1, 128, 256, 192, {3, 3}, {71, 71}, {2, 2}, {1, 1}, {1, 1}, {1, 1}};
    ExecutionConfig config;

    if(!parse_cmd_args(argc, argv, problem_size, config))
    {
        return false;
    }

    switch(problem_size.num_dim_spatial_)
    {
    case 1: return run_convnd_fwd_max<1>(problem_size, config);
    case 2: return run_convnd_fwd_max<2>(problem_size, config);
    case 3: return run_convnd_fwd_max<3>(problem_size, config);
    }

    return false;
}
