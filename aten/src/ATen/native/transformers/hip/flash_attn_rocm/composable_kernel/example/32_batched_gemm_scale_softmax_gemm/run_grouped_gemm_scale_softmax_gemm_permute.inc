// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    bool input_permute  = false;
    bool output_permute = true;

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
    else if(argc == 6)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        input_permute  = std::stoi(argv[4]);
        output_permute = std::stoi(argv[5]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 5: input / output permute\n");
        exit(0);
    }

    float alpha = 1; // scaling after 1st gemm

    std::size_t group_count = 7;

    // Problem descs
    std::vector<DeviceGemmInstance::ProblemDesc> problem_descs;
    std::vector<const void*> p_a;
    std::vector<const void*> p_b0;
    std::vector<const void*> p_b1;
    std::vector<void*> p_c;
    std::vector<std::vector<int>> g0_g1_m_n_k_o;

    std::vector<Tensor<ADataType>> a_tensors;
    std::vector<Tensor<B0DataType>> b0_tensors;
    std::vector<Tensor<B1DataType>> b1_tensors;
    std::vector<Tensor<CDataType>> c_tensors;

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;
    std::vector<DeviceMemPtr> a_tensors_device;
    std::vector<DeviceMemPtr> b0_tensors_device;
    std::vector<DeviceMemPtr> b1_tensors_device;
    std::vector<DeviceMemPtr> c_tensors_device;

    std::size_t flop = 0, num_byte = 0;

    std::cout << "group count " << group_count << ". printing first 4 groups\n";
    for(std::size_t i = 0; i < group_count; i++)
    {
        int M  = 128 * (rand() % 8 + 1);
        int N  = 128 * (rand() % 8 + 1);
        int K  = 40;
        int O  = 40 * (rand() % 2 + 1);
        int G0 = rand() % 3 + 1;
        int G1 = rand() % 5 + 1;

        g0_g1_m_n_k_o.push_back({G0, G1, M, N, K, O});

        std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
        std::vector<ck::index_t> a_gs_ms_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // A layout [G0, M, G1, K]
                : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // A layout [G0, G1, M, K]

        std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
        std::vector<ck::index_t> b0_gs_ns_ks_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // B0 layout [G0, N, G1, K]
                : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // B0 layout [G0, G1, N, K]

        std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
        std::vector<ck::index_t> b1_gs_os_ns_strides =
            input_permute
                ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // B1 layout [G0, N, G1, O]
                : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // B1 layout [G0, G1, N, O]

        std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
        std::vector<ck::index_t> c_gs_ms_os_strides =
            output_permute
                ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // C layout [G0, M, G1, O]
                : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // C layout [G0, G1, M, O]

        problem_descs.push_back({a_gs_ms_ks_lengths,
                                 a_gs_ms_ks_strides,
                                 b0_gs_ns_ks_lengths,
                                 b0_gs_ns_ks_strides,
                                 b1_gs_os_ns_lengths,
                                 b1_gs_os_ns_strides,
                                 c_gs_ms_os_lengths,
                                 c_gs_ms_os_strides,
                                 {},   // acc0_biases_gs_ms_ns_lengths
                                 {},   // acc0_biases_gs_ms_ns_strides
                                 {},   // acc1_biases_gs_ms_os_lengths
                                 {}}); // acc1_biases_gs_ms_os_strides

        // C_m_o = A_m_k * B0_k_n * B1_n_o
        Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
        Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
        Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
        Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);

        int Batch = G0 * G1;
        flop += (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * Batch;
        num_byte += (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                     sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                    Batch;

        if(i < 4)
        {
            std::cout << "a_gs_ms_ks[" << i << "]: " << a_gs_ms_ks.mDesc << ", "
                      << "b0_gs_ns_ks[" << i << "]: " << b0_gs_ns_ks.mDesc << ", "
                      << "b1_gs_os_ns[" << i << "]: " << b1_gs_os_ns.mDesc << ", "
                      << "c_gs_ms_os[" << i << "]: " << c_gs_ms_os_device_result.mDesc << std::endl;
        }

        switch(init_method)
        {
        case 0: break;
        case 1:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-2, 2});
            break;
        case 2:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
            break;
        case 3:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
            break;
        default:
            a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
            b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
            b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        }

        a_tensors.push_back(a_gs_ms_ks);
        b0_tensors.push_back(b0_gs_ns_ks);
        b1_tensors.push_back(b1_gs_os_ns);
        c_tensors.push_back(c_gs_ms_os_device_result);

        a_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize()));
        b0_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize()));
        b1_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_gs_ms_os_device_result.mDesc.GetElementSpaceSize()));

        a_tensors_device[i]->ToDevice(a_gs_ms_ks.mData.data());
        b0_tensors_device[i]->ToDevice(b0_gs_ns_ks.mData.data());
        b1_tensors_device[i]->ToDevice(b1_gs_os_ns.mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b0.push_back(b0_tensors_device[i]->GetDeviceBuffer());
        p_b1.push_back(b1_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(p_a,
                                      p_b0,
                                      p_b1,
                                      p_c,
                                      {}, // p_acc0_biases
                                      {}, // p_acc1_biases
                                      problem_descs,
                                      a_element_op,
                                      b0_element_op,
                                      acc0_element_op,
                                      b1_element_op,
                                      c_element_op);

    // specify workspace for problem_desc
    DeviceMem problem_desc_workspace(gemm.GetWorkSpaceSize(&argument));

    gemm.SetWorkSpacePointer(&argument, problem_desc_workspace.GetDeviceBuffer());

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    bool pass = true;
    if(do_verification)
    {
        for(std::size_t i = 0; i < group_count; i++)
        {
            const int& G0 = g0_g1_m_n_k_o[i][0];
            const int& G1 = g0_g1_m_n_k_o[i][1];
            const int& M  = g0_g1_m_n_k_o[i][2];
            const int& N  = g0_g1_m_n_k_o[i][3];
            const int& K  = g0_g1_m_n_k_o[i][4];
            const int& O  = g0_g1_m_n_k_o[i][5];

            const auto& c_gs_ms_os_lengths = problem_descs[i].c_gs_ms_os_lengths;
            const auto& c_gs_ms_os_strides = problem_descs[i].c_gs_ms_os_strides;

            const auto& a_gs_ms_ks         = a_tensors[i];
            const auto& b0_gs_ns_ks        = b0_tensors[i];
            const auto& b1_gs_os_ns        = b1_tensors[i];
            auto& c_gs_ms_os_device_result = c_tensors[i];
            auto& c_gs_ms_os_device_buf    = *c_tensors_device[i];

            c_gs_ms_os_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

            Tensor<ADataType> a_g_m_k({G0 * G1, M, K});
            Tensor<B0DataType> b0_g_k_n({G0 * G1, K, N});
            Tensor<B1DataType> b1_g_n_o({G0 * G1, N, O});
            Tensor<AccDataType> acc0_g_m_n({G0 * G1, M, N});        // scratch object after gemm0
            Tensor<ADataType> a1_g_m_n({G0 * G1, M, N});            // scratch object after softmax
            Tensor<CDataType> c_g_m_o_host_result({G0 * G1, M, O}); // scratch object after gemm1
            Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);

            // permute
            a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
                a_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
            });
            b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
                b0_g_k_n(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
            });
            b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
                b1_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
            });

            // gemm 0
            auto ref_gemm0          = ReferenceGemm0Instance{};
            auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
            auto ref_gemm0_argument = ref_gemm0.MakeArgument(
                a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

            ref_gemm0_invoker.Run(ref_gemm0_argument);

            // masking
            const auto mask = DeviceGemmInstance::C0MatrixMask(N);
            acc0_g_m_n.ForEach([&](auto& self, auto idx) {
                if(mask.IsMaskedElement(idx[1], idx[2]))
                    self(idx) = -ck::NumericLimits<float>::Infinity();
            });

            // softmax
            auto ref_softmax          = ReferenceSoftmaxInstance{};
            auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
            auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2});

            ref_softmax_invoker.Run(ref_softmax_argument);

            // gemm 1
            auto ref_gemm1          = ReferenceGemm1Instance{};
            auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
            auto ref_gemm1_argument = ref_gemm1.MakeArgument(a1_g_m_n,
                                                             b1_g_n_o,
                                                             c_g_m_o_host_result,
                                                             PassThrough{},
                                                             b1_element_op,
                                                             c_element_op);

            ref_gemm1_invoker.Run(ref_gemm1_argument);

            // permute
            c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
                const size_t& g0 = idx[0];
                const size_t& g1 = idx[1];

                const size_t g = g0 * G1 + g1;

                self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
            });

            bool pass_ =
                ck::utils::check_err(c_gs_ms_os_device_result.mData, c_gs_ms_os_host_result.mData);
            pass &= pass_;
        }
    }

    return pass ? 0 : 1;
}
