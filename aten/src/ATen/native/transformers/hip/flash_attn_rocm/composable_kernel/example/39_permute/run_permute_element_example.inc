// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_permute_element(const Problem& problem)
{
    const auto& input_shape = problem.shape;
    const auto& input_axes  = problem.axes;

    const auto output_shape = transpose(input_shape, input_axes);

    Tensor<InDataType> input_tensor(input_shape);
    Tensor<OutDataType> output_tensor(output_shape);

    ck::utils::FillUniformDistribution<InDataType>{-1.f, 1.f}(input_tensor);

    DeviceMem input_device_buf(input_tensor.GetElementSpaceSizeInBytes());
    DeviceMem output_device_buf(output_tensor.GetElementSpaceSizeInBytes());

    using std::data;
    input_device_buf.ToDevice(data(input_tensor));

    static_assert(std::is_default_constructible_v<DevicePermuteInstance>);

    auto permute  = DevicePermuteInstance{};
    auto argument = permute.MakeArgument(to_array(input_shape),
                                         to_array(input_tensor.GetStrides()),
                                         to_array(output_shape),
                                         to_array(output_tensor.GetStrides()),
                                         input_device_buf.GetDeviceBuffer(),
                                         output_device_buf.GetDeviceBuffer(),
                                         PassThrough{});

    if(!permute.IsSupportedArgument(argument))
    {
        std::cerr << "The runtime parameters seems not supported by the device instance, exiting!"
                  << std::endl;
        return false;
    };

    auto invoker   = permute.MakeInvoker();
    float ave_time = invoker.Run(argument, StreamConfig{nullptr, true});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    output_device_buf.FromDevice(data(output_tensor));

    Tensor<OutDataType> output_tensor_host(output_shape);
    if(!host_permute(input_tensor, input_axes, PassThrough{}, output_tensor_host))
    {
        return false;
    }

    return ck::utils::check_err(output_tensor.AsSpan<const OutDataType>(),
                                output_tensor_host.AsSpan<const OutDataType>(),
                                "Error: incorrect results in output tensor",
                                1e-6,
                                1e-6);
}

bool run_permute_element_example(const Problem::Shape& shape, const Problem::Axes& axes)
{
    return run_permute_element(Problem{shape, axes});
}
