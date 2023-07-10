// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ABDataType = F16;
using CDataType  = F16;

using Add = ck::tensor_operation::element_wise::Add;

using DeviceElementwiseAddInstance =
    ck::tensor_operation::device::DeviceElementwise<ck::Tuple<ABDataType, ABDataType>,
                                                    ck::Tuple<CDataType>,
                                                    Add,
                                                    3,
                                                    8,
                                                    ck::Sequence<1, 8>,
                                                    ck::Sequence<8>>;

template <typename HostTensorA, typename HostTensorB, typename HostTensorC, typename Functor>
void host_broadcast3D_am_bmnk(HostTensorC& C,
                              const HostTensorA& A,
                              const HostTensorB& B,
                              const std::vector<std::size_t>& shape,
                              Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0, 0))>;

    for(std::size_t m = 0; m < shape[0]; ++m)
        for(std::size_t n = 0; n < shape[1]; ++n)
            for(std::size_t k = 0; k < shape[2]; ++k)
            {
                auto a_val  = A(m);
                auto b_val  = B(m, n, k);
                ctype c_val = 0;
                functor(c_val, a_val, b_val);
                C(m, n, k) = c_val;
            }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    std::vector<std::size_t> mnk = {4, 16, 32};
    ck::index_t M                = mnk[0];

    Tensor<ABDataType> a_m({M});
    Tensor<ABDataType> b_m_n_k(mnk);
    Tensor<CDataType> c_m_n_k(mnk);

    a_m.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});
    b_m_n_k.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});

    DeviceMem a_m_device_buf(sizeof(ABDataType) * a_m.mDesc.GetElementSpaceSize());
    DeviceMem b_m_n_k_device_buf(sizeof(ABDataType) * b_m_n_k.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_k_device_buf(sizeof(CDataType) * c_m_n_k.mDesc.GetElementSpaceSize());

    a_m_device_buf.ToDevice(a_m.mData.data());
    b_m_n_k_device_buf.ToDevice(b_m_n_k.mData.data());

    std::array<const void*, 2> input = {a_m_device_buf.GetDeviceBuffer(),
                                        b_m_n_k_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {c_m_n_k_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 3> abc_lengths;
    std::array<ck::index_t, 3> a_strides = {1, 0, 0};
    std::array<ck::index_t, 3> b_strides;
    std::array<ck::index_t, 3> c_strides;

    ck::ranges::copy(mnk, abc_lengths.begin());
    ck::ranges::copy(b_m_n_k.mDesc.GetStrides(), b_strides.begin());
    ck::ranges::copy(c_m_n_k.mDesc.GetStrides(), c_strides.begin());

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument     = broadcastAdd.MakeArgumentPointer(
        abc_lengths, {a_strides, b_strides}, {c_strides}, input, output, Add{});

    if(!broadcastAdd.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    auto broadcastAdd_invoker_ptr = broadcastAdd.MakeInvokerPointer();
    float ave_time =
        broadcastAdd_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_m_n_k_device_buf.FromDevice(c_m_n_k.mData.data());
        Tensor<CDataType> host_c_m_n_k(mnk);

        host_broadcast3D_am_bmnk<Tensor<ABDataType>, Tensor<ABDataType>, Tensor<CDataType>, Add>(
            host_c_m_n_k, a_m, b_m_n_k, mnk, Add{});

        pass &=
            ck::utils::check_err(c_m_n_k, host_c_m_n_k, "Error: Incorrect results c", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
