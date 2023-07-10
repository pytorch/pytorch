// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_reduction.hpp"

#include "reduce_example_common.hpp"

template <typename InOutDataType,
          typename AccDataType,
          ck::ReduceTensorOp ReduceOpId,
          ck::index_t Rank,
          ck::index_t NumReduceDim,
          bool PropagateNan>
int reduce_multiblock_atomic_add_impl(bool do_verification,
                                      int init_method,
                                      bool time_kernel,
                                      const std::vector<size_t>& inLengths,
                                      const std::array<int, NumReduceDim>& reduceDims,
                                      float alpha,
                                      float beta)

{
    using namespace ck;
    using namespace ck::tensor_operation::device;

    constexpr index_t NumOutDim = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

    constexpr bool op_support_atomic_add =
        (ReduceOpId == ReduceTensorOp::ADD || ReduceOpId == ReduceTensorOp::AVG);

    constexpr bool invalid_reduce_1 = !op_support_atomic_add;
    constexpr bool invalid_reduce_2 =
        !(std::is_same<InOutDataType, float>::value || std::is_same<InOutDataType, double>::value);

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2);

    if(invalid_reduce)
    {
        std::cerr << "The reduction setting is invalid, exiting!" << std::endl;
        return (-1);
    };

    using ReduceOperation = typename reduce_binary_operator<ReduceOpId>::opType;
    using InElementwiseOperation =
        typename reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
    using AccElementwiseOperation =
        typename reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

    using DeviceReduceInstance =
        ck::tensor_operation::device::DeviceReduceMultiBlock<InOutDataType,
                                                             AccDataType,
                                                             InOutDataType,
                                                             Rank,
                                                             NumReduceDim,
                                                             ReduceOperation,
                                                             InElementwiseOperation,
                                                             AccElementwiseOperation,
                                                             InMemoryDataOperationEnum::AtomicAdd,
                                                             PropagateNan,
                                                             false,
                                                             false, // HaveIndexInputIfOutputIndex
                                                             256,
                                                             4,
                                                             64,
                                                             1,
                                                             1,
                                                             0,
                                                             1,
                                                             1>;

    Tensor<InOutDataType> in(inLengths);

    std::vector<size_t> outLengths;

    auto invariantDims = get_invariant_dims<Rank, NumReduceDim>(reduceDims);

    if(invariantDims.empty())
        outLengths.push_back(1);
    else
        for(auto dim : invariantDims)
            outLengths.push_back(inLengths[dim]);

    Tensor<InOutDataType> out_ref(outLengths);
    Tensor<InOutDataType> out(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t invariant_total_length = out.mDesc.GetElementSize();
    size_t reduce_total_length    = in.mDesc.GetElementSize() / invariant_total_length;

    std::size_t num_thread = 1;

    if(do_verification)
    {
        switch(init_method)
        {
        case 0: break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            break;
        case 2:
            in.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0},
                                            num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_ref.mDesc.GetElementSpaceSize(); i++)
                out.mData[i] = out_ref.mData[i];
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InOutDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(InOutDataType) * out.mDesc.GetElementSpaceSize());

    in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    InElementwiseOperation in_elementwise_op;
    AccElementwiseOperation acc_elementwise_op;

    std::tie(in_elementwise_op, acc_elementwise_op) =
        reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
            static_cast<int32_t>(reduce_total_length));

    if(do_verification)
    {
        ReductionHost<InOutDataType,
                      AccDataType,
                      InOutDataType,
                      ReduceOperation,
                      InElementwiseOperation,
                      AccElementwiseOperation,
                      Rank,
                      NumReduceDim,
                      PropagateNan,
                      false>
            hostReduce(in.mDesc, out_ref.mDesc, invariantDims, reduceDims);

        hostReduce.Run(alpha,
                       in.mData.data(),
                       beta,
                       out_ref.mData.data(),
                       nullptr,
                       in_elementwise_op,
                       acc_elementwise_op);
    };

    std::array<index_t, Rank> arrInLengths;
    std::array<index_t, Rank> arrInStrides;
    std::array<index_t, NumOutDim> arrOutLengths;
    std::array<index_t, NumOutDim> arrOutStrides;

    ck::ranges::copy(inLengths, arrInLengths.begin());
    ck::ranges::copy(inStrides, arrInStrides.begin());
    ck::ranges::copy(outLengths, arrOutLengths.begin());
    ck::ranges::copy(outStrides, arrOutStrides.begin());

    auto reduce = DeviceReduceInstance{};

    auto argument_ptr = reduce.MakeArgumentPointer(arrInLengths,
                                                   arrInStrides,
                                                   arrOutLengths,
                                                   arrOutStrides,
                                                   reduceDims,
                                                   alpha,
                                                   beta,
                                                   in_dev.GetDeviceBuffer(),
                                                   nullptr,
                                                   out_dev.GetDeviceBuffer(),
                                                   nullptr,
                                                   in_elementwise_op,
                                                   acc_elementwise_op);

    if(!reduce.IsSupportedArgument(argument_ptr.get()))
    {
        std::cerr
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;

        return (-2);
    };

    std::string reduce_name = reduce.GetTypeString();

    auto invoker_ptr = reduce.MakeInvokerPointer();

    float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InOutDataType) +
                            invariant_total_length * sizeof(InOutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        out_dev.FromDevice(out.mData.data());
        pass = pass && ck::utils::check_err(out, out_ref);
    };

    return (pass ? 0 : 1);
}
