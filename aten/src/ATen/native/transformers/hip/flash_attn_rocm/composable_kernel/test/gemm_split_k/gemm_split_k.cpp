// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

#include "ck/library/utility/host_gemm.hpp"

enum struct GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
};

template <typename T>
static bool check_out(const Tensor<T>& ref, const Tensor<T>& result)
{
    float max_diff = 1e-6;

    for(std::size_t i = 0; i < ref.mData.size(); ++i)
    {
        float diff = std::abs(double(ref.mData[i]) - double(result.mData[i]));
        if(max_diff < diff)
        {
            return false;
        }
    }

    return true;
}

struct gemmArgs
{
    GemmMatrixLayout layout;
    int M;
    int N;
    int K;
    int StrideA;
    int StrideB;
    int StrideC;
    int KBatch;
};

int test_gemm(const gemmArgs& args)
{
    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    bool a_row_major, b_row_major, c_row_major;

    switch(args.layout)
    {
    case GemmMatrixLayout::MK_KN_MN:
        a_row_major = true;
        b_row_major = true;
        c_row_major = true;
        break;
    case GemmMatrixLayout::MK_NK_MN:
        a_row_major = true;
        b_row_major = false;
        c_row_major = true;
        break;
    case GemmMatrixLayout::KM_KN_MN:
        a_row_major = false;
        b_row_major = true;
        c_row_major = true;
        break;
    case GemmMatrixLayout::KM_NK_MN:
        a_row_major = false;
        b_row_major = false;
        c_row_major = true;
        break;
    default: printf("not supported layout"); return 1;
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, bool row_major) {
            using namespace ck::literals;

            if(row_major)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<float> a_m_k(f_host_tensor_descriptor(args.M, args.K, args.StrideA, a_row_major));
    Tensor<float> b_k_n(f_host_tensor_descriptor(args.K, args.N, args.StrideB, b_row_major));
    Tensor<float> c_m_n_host_result(
        f_host_tensor_descriptor(args.M, args.N, args.StrideC, c_row_major));
    Tensor<float> c_m_n_device_result(
        f_host_tensor_descriptor(args.M, args.N, args.StrideC, c_row_major));

    // init data
    std::size_t num_thread = 1;
    a_m_k.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5}, num_thread);
    b_k_n.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5}, num_thread);
    // set zero to c_device_buf
    c_m_n_device_result.GenerateTensorValue(GeneratorTensor_0<float>{}, num_thread);

    host_gemm_mk_kn_mn(a_m_k,
                       b_k_n,
                       c_m_n_host_result,
                       ck::tensor_operation::element_wise::PassThrough{},
                       ck::tensor_operation::element_wise::PassThrough{},
                       ck::tensor_operation::element_wise::PassThrough{});

    DeviceMem a_device_buf(sizeof(float) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(float) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(float) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    c_device_buf.ToDevice(c_m_n_device_result.mData.data());

    auto test = [&](auto a_layout, auto b_layout, auto c_layout) {
        bool success = false;

        using DeviceOp = ck::tensor_operation::device::DeviceGemmSplitK<decltype(a_layout),
                                                                        decltype(b_layout),
                                                                        decltype(c_layout),
                                                                        float,
                                                                        float,
                                                                        float,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        PassThrough>;

        const auto gemm_ptrs =
            ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
                DeviceOp>::GetInstances();

        for(auto& gemm_ptr : gemm_ptrs)
        {
            auto argument_ptr =
                gemm_ptr->MakeArgumentPointer(static_cast<float*>(a_device_buf.GetDeviceBuffer()),
                                              static_cast<float*>(b_device_buf.GetDeviceBuffer()),
                                              static_cast<float*>(c_device_buf.GetDeviceBuffer()),
                                              args.M,
                                              args.N,
                                              args.K,
                                              args.StrideA,
                                              args.StrideB,
                                              args.StrideC,
                                              ck::tensor_operation::element_wise::PassThrough{},
                                              ck::tensor_operation::element_wise::PassThrough{},
                                              ck::tensor_operation::element_wise::PassThrough{},
                                              args.KBatch);

            auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

            if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                invoker_ptr->Run(argument_ptr.get());

                c_device_buf.FromDevice(c_m_n_device_result.mData.data());

                if(!check_out(c_m_n_host_result, c_m_n_device_result))
                {
                    success = false;
                    break;
                }
                success = true;
            }
        }

        return success;
    };

    bool success = false;

    if(args.layout == GemmMatrixLayout::MK_KN_MN)
    {
        success = test(Row{}, Row{}, Row{});
    }
    else if(args.layout == GemmMatrixLayout::MK_NK_MN)
    {
        success = test(Row{}, Col{}, Row{});
    }
    else if(args.layout == GemmMatrixLayout::KM_KN_MN)
    {
        success = test(Col{}, Row{}, Row{});
    }
    else
    {
        success = test(Col{}, Col{}, Row{});
    }

    auto error_code = 0;
    if(success)
    {
        std::cout << "test split k : Pass" << std::endl;
    }
    else
    {
        std::cout << "test split k: Fail " << std::endl;
        error_code = -1; // test needs to report failure
    }
    return error_code;
}

int main(int argc, char* argv[])
{
    std::vector<gemmArgs> test_cases;
    if(argc == 1)
    {
        test_cases = {{GemmMatrixLayout::MK_KN_MN, 1024, 1024, 1024, 1024, 1024, 1024, 2},
                      {GemmMatrixLayout::MK_KN_MN, 1024, 1024, 1024, 1024, 1024, 1024, 8}};
    }
    else if(argc == 9)
    {
        const auto layout = static_cast<GemmMatrixLayout>(std::stoi(argv[1]));

        const int M = std::stoi(argv[2]);
        const int N = std::stoi(argv[3]);
        const int K = std::stoi(argv[4]);

        const int StrideA = std::stoi(argv[5]);
        const int StrideB = std::stoi(argv[6]);
        const int StrideC = std::stoi(argv[7]);
        const int KBatch  = std::stoi(argv[8]);
        test_cases        = {{layout, M, N, K, StrideA, StrideB, StrideC, KBatch}};
    }
    else
    {
        printf("arg1: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, m] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, m] * B[n, k] = C[m, n])\n");
        printf("arg2 to 7: M, N, K, StrideA, StrideB, StrideC KBatch\n");
        return -1;
    }
    bool error = false;
    for(const auto& kinder : test_cases)
    {
        error |= test_gemm(kinder);
    }
    return error ? 1 : 0;
}
