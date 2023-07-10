#pragma once

struct ProblemSize final
{
    std::vector<ck::index_t> Ms;
    std::vector<ck::index_t> Ns;
    std::vector<ck::index_t> Ks;

    std::vector<ck::index_t> stride_As;
    std::vector<ck::index_t> stride_Bs;
    std::vector<ck::index_t> stride_Cs;

    ck::index_t group_count;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

bool run_grouped_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
    static_assert(sizeof(ADataType) == sizeof(KernelADataType));
    static_assert(sizeof(BDataType) == sizeof(KernelBDataType));
    static_assert(sizeof(EDataType) == sizeof(KernelEDataType));
#endif
    int group_count = problem_size.group_count;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        int M = problem_size.Ms[i];
        int N = problem_size.Ns[i];
        int K = problem_size.Ks[i];

        int stride_A = problem_size.stride_As[i];
        int stride_B = problem_size.stride_Bs[i];
        int stride_C = problem_size.stride_Cs[i];

        gemm_descs.push_back({M, N, K, stride_A, stride_B, stride_C, {}});
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

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<EDataType>> c_host_tensors;
#ifdef BUILD_INT4_EXAMPLE
    std::vector<Tensor<KernelEDataType>> c_device_tensors;
#else
    std::vector<Tensor<EDataType>> c_device_tensors;
#endif

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    c_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, c_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    std::size_t flop = 0, num_btype = 0;

    for(std::size_t i = 0; i < gemm_descs.size(); i++)
    {
        a_tensors.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].K_, gemm_descs[i].stride_A_, ALayout{})));
        b_tensors.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            gemm_descs[i].K_, gemm_descs[i].N_, gemm_descs[i].stride_B_, BLayout{})));
        c_host_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].N_, gemm_descs[i].stride_C_, ELayout{})));
#ifdef BUILD_INT4_EXAMPLE
        c_device_tensors.push_back(Tensor<KernelEDataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].N_, gemm_descs[i].stride_C_, ELayout{})));
#else
        c_device_tensors.push_back(Tensor<EDataType>(f_host_tensor_descriptor(
            gemm_descs[i].M_, gemm_descs[i].N_, gemm_descs[i].stride_C_, ELayout{})));
#endif
        std::cout << "gemm[" << i << "] a_m_k: " << a_tensors[i].mDesc
                  << " b_k_n: " << b_tensors[i].mDesc << " c_m_n: " << c_device_tensors[i].mDesc
                  << std::endl;

        flop += std::size_t(2) * gemm_descs[i].M_ * gemm_descs[i].K_ * gemm_descs[i].N_;
        num_btype += sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize() +
                     sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize() +
                     sizeof(EDataType) * c_device_tensors[i].mDesc.GetElementSize();

        switch(config.init_method)
        {
        case 0: break;
        case 1:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
            break;
        case 2:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
            break;
        default:
            a_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<0>{});
            b_tensors[i].GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        }
    }

    for(std::size_t i = 0; i < gemm_descs.size(); i++)
    {
        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ADataType) * a_tensors[i].mDesc.GetElementSpaceSize()));
        b_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(BDataType) * b_tensors[i].mDesc.GetElementSpaceSize()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(EDataType) * c_device_tensors[i].mDesc.GetElementSpaceSize()));

#ifdef BUILD_INT4_EXAMPLE
        const Tensor<KernelADataType> a_converted(a_tensors[i]);
        const Tensor<KernelBDataType> b_converted(b_tensors[i]);

        a_tensors_device[i]->ToDevice(a_converted.mData.data());
        b_tensors_device[i]->ToDevice(b_converted.mData.data());
#else
        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());
#endif

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CDEElementOp{};

    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();

    std::vector<std::array<const void*, 0>> p_Ds = {};

    // do GEMM
    auto argument = gemm.MakeArgument(
        p_a, p_b, p_Ds, p_c, gemm_descs, a_element_op, b_element_op, c_element_op);

    DeviceMem gemm_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, gemm_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass = true;
    if(config.do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                EDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CDEElementOp>;

        for(std::size_t i = 0; i < gemm_descs.size(); i++)
        {
            c_tensors_device[i]->FromDevice(c_device_tensors[i].mData.data());
            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                      b_tensors[i],
                                                      c_host_tensors[i],
                                                      a_element_op,
                                                      b_element_op,
                                                      c_element_op);

            ref_invoker.Run(ref_argument);

#ifdef BUILD_INT4_EXAMPLE
            const Tensor<EDataType> c_device_result_converted(c_device_tensors[i]);
            pass &= ck::utils::check_err(c_device_result_converted, c_host_tensors[i]);

#else
            pass &= ck::utils::check_err(c_device_tensors[i], c_host_tensors[i]);
#endif
        }
    }

    if(config.time_kernel)
    {
        float ave_time   = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
        float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }

    return pass;
}

bool run_grouped_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    problem_size.group_count = 16;

    for(int i = 0; i < problem_size.group_count; i++)
    {
        problem_size.Ms.push_back(256 + 256 * i);
        problem_size.Ns.push_back(128 + 128 * i);
        problem_size.Ks.push_back(128 + 64 * i);

        problem_size.stride_As.push_back(problem_size.Ks[i]);
        problem_size.stride_Bs.push_back(problem_size.Ks[i]);
        problem_size.stride_Cs.push_back(problem_size.Ns[i]);
    }

    if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        exit(0);
    }

    return run_grouped_gemm(problem_size, config);
}
