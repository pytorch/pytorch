// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iomanip>

#include "ck/ck.hpp"

#include "ck/library/tensor_operation_instance/gpu/elementwise_normalization.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_layernorm.hpp"

namespace ck {
namespace profiler {

template <typename HostTensorA, typename HostTensorB, typename HostTensorC, typename Functor>
void host_elementwise2D(HostTensorC& C,
                        const HostTensorA& A,
                        const HostTensorB& B,
                        const std::vector<std::size_t>& shape,
                        Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0, 0))>;

    for(std::size_t m = 0; m < shape[0]; ++m)
        for(std::size_t n = 0; n < shape[1]; ++n)
        {
            auto a_val  = A(m, n);
            auto b_val  = B(m, n);
            ctype c_val = 0;
            functor(c_val, a_val, b_val);
            C(m, n) = c_val;
        }
}

template <typename ADataType,
          typename BDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType>
bool profile_elementwise_layernorm_impl(int do_verification,
                                        int init_method,
                                        bool do_log,
                                        bool time_kernel,
                                        std::vector<index_t> length)
{
    using Add         = ck::tensor_operation::element_wise::Add;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    if(length.size() != 2)
        return false;

    index_t M      = length[0];
    index_t N      = length[1];
    index_t Stride = N;

    constexpr int Rank         = 2;
    constexpr int NumReduceDim = 1;

    std::vector<index_t> reduce_dim      = {1};
    std::vector<index_t> gammaBetaLength = {N};
    std::vector<index_t> gammaBetaStride = {0, 1};

    auto f_host_tensor_descriptor2d = [](std::size_t row, std::size_t col, std::size_t stride) {
        using namespace ck::literals;

        return HostTensorDescriptor({row, col}, {stride, 1_uz});
    };

    Tensor<ADataType> a(length);
    Tensor<BDataType> b(length);
    Tensor<GammaDataType> gamma(gammaBetaLength);
    Tensor<BetaDataType> beta(gammaBetaLength);
    Tensor<YDataType> y(length);
    Tensor<YDataType> host_y(length);

    switch(init_method)
    {
    case 0:
        a.GenerateTensorValue(GeneratorTensor_1<ADataType>{});
        b.GenerateTensorValue(GeneratorTensor_1<BDataType>{});
        gamma.GenerateTensorValue(GeneratorTensor_1<GammaDataType>{});
        beta.GenerateTensorValue(GeneratorTensor_1<BetaDataType>{});
        break;
    case 1:
        a.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        gamma.GenerateTensorValue(GeneratorTensor_2<GammaDataType>{-5, 5});
        beta.GenerateTensorValue(GeneratorTensor_2<BetaDataType>{-5, 5});
        break;
    default:
        a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0, 1});
        b.GenerateTensorValue(GeneratorTensor_3<BDataType>{0, 1});
        gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{-0.5, 0.5});
        beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{-0.5, 0.5});
    }

    DeviceMem a_dev(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_dev(sizeof(ADataType) * b.mDesc.GetElementSpaceSize());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpaceSize());

    a_dev.ToDevice(a.mData.data());
    b_dev.ToDevice(b.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    std::array<const void*, 2> input = {a_dev.GetDeviceBuffer(), b_dev.GetDeviceBuffer()};

    // add device normalization instances
    using DeviceOp = ck::tensor_operation::device::DeviceElementwiseNormalization<
        ck::Tuple<ADataType, BDataType>,
        GammaDataType,
        BetaDataType,
        AccDataType,
        YDataType,
        Add,
        PassThrough,
        2,
        1>;

    // get device op instances
    const auto instance_ptrs =
        ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

    std::cout << "found " << instance_ptrs.size() << " instances" << std::endl;

    std::string best_instance_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_gb_per_sec = 0;

    if(do_verification)
    {
        using XDataType             = ADataType;
        std::vector<std::size_t> mn = {static_cast<unsigned long>(M),
                                       static_cast<unsigned long>(N)};
        Tensor<XDataType> x(f_host_tensor_descriptor2d(M, N, Stride));
        host_elementwise2D<Tensor<ADataType>, Tensor<BDataType>, Tensor<XDataType>, Add>(
            x, a, b, mn, Add{});

        using ReferenceInstance = ck::tensor_operation::host::ReferenceLayernorm<XDataType,
                                                                                 GammaDataType,
                                                                                 BetaDataType,
                                                                                 YDataType,
                                                                                 AccDataType,
                                                                                 PassThrough,
                                                                                 Rank,
                                                                                 NumReduceDim>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(x, gamma, beta, host_y, PassThrough{}, {M, N}, {1}, 1e-4);
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);
    }

    int num_kernel = 0;

    for(auto& inst_ptr : instance_ptrs)
    {
        auto argument_ptr = inst_ptr->MakeArgumentPointer(
            length,
            {
                std::vector<ck::index_t>{a.mDesc.GetStrides().begin(), a.mDesc.GetStrides().end()},
                std::vector<ck::index_t>{b.mDesc.GetStrides().begin(), b.mDesc.GetStrides().end()},
            },
            gammaBetaStride,
            gammaBetaStride,
            std::vector<ck::index_t>{y.mDesc.GetStrides().begin(), y.mDesc.GetStrides().end()},
            reduce_dim,
            1e-4,
            input,
            gamma_dev.GetDeviceBuffer(),
            beta_dev.GetDeviceBuffer(),
            y_dev.GetDeviceBuffer(),
            Add{},
            PassThrough{});

        if(inst_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            ++num_kernel;
        }
        else
        {
            continue;
        }

        auto invoker_ptr = inst_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes = a.mDesc.GetElementSize() * sizeof(ADataType) +
                                b.mDesc.GetElementSize() * sizeof(BDataType) +
                                gamma.mDesc.GetElementSize() * sizeof(GammaDataType) +
                                beta.mDesc.GetElementSize() * sizeof(BetaDataType) +
                                y.mDesc.GetElementSize() * sizeof(YDataType);

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        if(time_kernel)
            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
                      << inst_ptr->GetTypeString() << std::endl;

        if(avg_time < best_avg_time)
        {
            best_instance_name = inst_ptr->GetTypeString();
            best_avg_time      = avg_time;
            best_gb_per_sec    = gb_per_sec;
        }

        if(do_verification)
        {
            y_dev.FromDevice(y.mData.data());

            bool pass =
                ck::utils::check_err(y.mData, host_y.mData, "Error: Incorrect results", 1e-3, 1e-3);

            if(do_log)
            {
                LogRangeAsType<float>(std::cout << "a  : ", a.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "b  : ", b.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "host_y  : ", host_y.mData, ",") << std::endl;
                LogRangeAsType<float>(std::cout << "y  : ", y.mData, ",") << std::endl;
            }

            if(!pass)
            {
                std::cout << inst_ptr->GetTypeString() << " failed verification: ";
                LogRange(std::cout << "lengths = [", length, ", ") << "]." << std::endl;
                return false;
            }
            else
            {
                if(time_kernel)
                    std::cout << "pass" << std::endl;
            }
        }
    }

    if(time_kernel)
    {
        LogRange(std::cout << "length = ", length, ",") << ", ";
        std::cout << "num_kernel = " << num_kernel << ", best perf = " << best_avg_time << " ms, "
                  << best_gb_per_sec << " GB/s, " << best_instance_name << std::endl;
    }

    if(num_kernel == 0)
    {
        std::cout << "Error: No kernel is tested" << std::endl;
        return false;
    }

    return true;
}

} // namespace profiler
} // namespace ck
