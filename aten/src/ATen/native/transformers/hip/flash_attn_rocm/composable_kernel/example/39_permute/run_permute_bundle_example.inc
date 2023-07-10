// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_permute_bundle(const Problem& problem)
{
    const auto& input_bundle_shape = problem.shape;
    const auto& input_bundle_axes  = problem.axes;

    const auto output_bundle_shape = transpose(input_bundle_shape, input_bundle_axes);

    Tensor<BundleType> input_bundle_tensor(input_bundle_shape);
    Tensor<BundleType> output_bundle_tensor(output_bundle_shape);

    // initialize tensor by assigning DataType values
    ck::utils::FillUniformDistribution<DataType>{-1.f, 1.f}(input_bundle_tensor.AsSpan<DataType>());

    DeviceMem input_device_buf(input_bundle_tensor.GetElementSpaceSizeInBytes());
    DeviceMem output_device_buf(output_bundle_tensor.GetElementSpaceSizeInBytes());

    using std::data;
    input_device_buf.ToDevice(data(input_bundle_tensor));

    static_assert(std::is_default_constructible_v<DevicePermuteInstance>);

    auto permute  = DevicePermuteInstance{};
    auto argument = permute.MakeArgument(to_array(input_bundle_shape),
                                         to_array(input_bundle_tensor.GetStrides()),
                                         to_array(output_bundle_shape),
                                         to_array(output_bundle_tensor.GetStrides()),
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

    output_device_buf.FromDevice(data(output_bundle_tensor));

    constexpr std::size_t NumElemsInBundle = sizeof(BundleType) / sizeof(DataType);

    // extend tensor shape from [N, H, W] to [N, H, W, NumElemsInBundle]
    //               axes  from [0, 2, 1] to [0, 2, 1, 3]
    const auto input_shape = extend_shape(input_bundle_shape, NumElemsInBundle);
    const auto input_axes  = extend_axes(input_bundle_axes);

    using std::begin;

    Tensor<DataType> input_tensor(input_shape);
    ck::ranges::copy(input_bundle_tensor.AsSpan<const DataType>(), begin(input_tensor));

    Tensor<DataType> output_tensor(transpose(input_shape, input_axes));
    if(!host_permute(input_tensor, input_axes, PassThrough{}, output_tensor))
    {
        return false;
    }

    return ck::utils::check_err(output_bundle_tensor.AsSpan<const DataType>(),
                                output_tensor.AsSpan<const DataType>(),
                                "Error: incorrect results in output tensor",
                                1e-6,
                                1e-6);
}

bool run_permute_bundle_example(const Problem::Shape& shape, const Problem::Axes& axes)
{
    return run_permute_bundle(Problem{shape, axes});
}
