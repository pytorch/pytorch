// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_reduction.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <index_t Rank,
          index_t NumReduceDim,
          ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool UseIndex>
struct ReduceDescription
{
    static constexpr index_t Rank_              = Rank;
    static constexpr index_t NumReduceDim_      = NumReduceDim;
    static constexpr ReduceTensorOp ReduceOpId_ = ReduceOpId;
    static constexpr bool PropagateNan_         = PropagateNan;
    static constexpr bool UseIndex_             = UseIndex;
};

using reduce_description_instances =
    std::tuple<ReduceDescription<4, 3, ReduceTensorOp::ADD, false, false>, // for ADD
               ReduceDescription<4, 4, ReduceTensorOp::ADD, false, false>,
               ReduceDescription<4, 1, ReduceTensorOp::ADD, false, false>,
               ReduceDescription<2, 1, ReduceTensorOp::ADD, false, false>,

               ReduceDescription<4, 3, ReduceTensorOp::AVG, false, false>, // for AVG
               ReduceDescription<4, 4, ReduceTensorOp::AVG, false, false>,
               ReduceDescription<4, 1, ReduceTensorOp::AVG, false, false>,
               ReduceDescription<2, 1, ReduceTensorOp::AVG, false, false>,

               ReduceDescription<4, 3, ReduceTensorOp::NORM2, false, false>, // for NORM2
               ReduceDescription<4, 4, ReduceTensorOp::NORM2, false, false>,
               ReduceDescription<4, 1, ReduceTensorOp::NORM2, false, false>,
               ReduceDescription<2, 1, ReduceTensorOp::NORM2, false, false>,

               ReduceDescription<4, 3, ReduceTensorOp::MIN, false, false>, // for MIN
               ReduceDescription<4, 4, ReduceTensorOp::MIN, false, false>,
               ReduceDescription<4, 1, ReduceTensorOp::MIN, false, false>,
               ReduceDescription<2, 1, ReduceTensorOp::MIN, false, false>,
               ReduceDescription<4, 3, ReduceTensorOp::MAX, false, false>, // for MAX
               ReduceDescription<4, 4, ReduceTensorOp::MAX, false, false>,
               ReduceDescription<4, 1, ReduceTensorOp::MAX, false, false>,
               ReduceDescription<2, 1, ReduceTensorOp::MAX, false, false>,
               ReduceDescription<4, 3, ReduceTensorOp::AMAX, false, false>, // for AMAX
               ReduceDescription<4, 4, ReduceTensorOp::AMAX, false, false>,
               ReduceDescription<4, 1, ReduceTensorOp::AMAX, false, false>,
               ReduceDescription<2, 1, ReduceTensorOp::AMAX, false, false>,

               ReduceDescription<4, 3, ReduceTensorOp::MIN, false, true>, // for MIN
               ReduceDescription<4, 4, ReduceTensorOp::MIN, false, true>,
               ReduceDescription<4, 1, ReduceTensorOp::MIN, false, true>,
               ReduceDescription<2, 1, ReduceTensorOp::MIN, false, true>,
               ReduceDescription<4, 3, ReduceTensorOp::MAX, false, true>, // for MAX
               ReduceDescription<4, 4, ReduceTensorOp::MAX, false, true>,
               ReduceDescription<4, 1, ReduceTensorOp::MAX, false, true>,
               ReduceDescription<2, 1, ReduceTensorOp::MAX, false, true>,
               ReduceDescription<4, 3, ReduceTensorOp::AMAX, false, true>, // for AMAX
               ReduceDescription<4, 4, ReduceTensorOp::AMAX, false, true>,
               ReduceDescription<4, 1, ReduceTensorOp::AMAX, false, true>,
               ReduceDescription<2, 1, ReduceTensorOp::AMAX, false, true>>;

template <typename DescriptionType>
bool description_match(const DescriptionType& description,
                       int Rank,
                       const std::vector<int>& reduceDims,
                       ReduceTensorOp ReduceOpId,
                       bool PropagateNan,
                       bool UseIndex)
{
    if(description.Rank_ != Rank || description.ReduceOpId_ != ReduceOpId ||
       description.PropagateNan_ != PropagateNan || description.UseIndex_ != UseIndex)
        return (false);

    if(DescriptionType::NumReduceDim_ != reduceDims.size())
        return (false);

    bool result = true;

    return (result);
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <int Rank, int NumReduceDim>
static inline std::array<int, Rank - NumReduceDim>
get_invariant_dims(const std::array<int, NumReduceDim>& reduceDims)
{
    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    std::array<int, Rank - NumReduceDim> invariantDims;

    // collect invariant dimensions
    int dim = 0;
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            invariantDims[dim] = i;
            dim++;
        };

    return invariantDims;
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          int NumReduceDim,
          ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool UseIndex>
bool profile_reduce_impl_impl(bool do_verification,
                              int init_method,
                              bool do_dumpout,
                              bool time_kernel,
                              const std::vector<size_t>& inLengths,
                              const std::array<int, NumReduceDim>& reduceDims,
                              float alpha,
                              float beta)
{
    using namespace ck::tensor_operation::device;
    using namespace ck::tensor_operation::device::instance;
    using ck::host_common::dumpBufferToFile;

    constexpr index_t NumOutDim = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

    constexpr bool op_support_indices =
        (ReduceOpId == ReduceTensorOp::MIN || ReduceOpId == ReduceTensorOp::MAX ||
         ReduceOpId == ReduceTensorOp::AMAX);

    constexpr bool OutputIndex = (op_support_indices && UseIndex);

    constexpr bool out_support_atomic_add = std::is_same<OutDataType, float>::value;
    constexpr bool op_support_atomic_add =
        !op_support_indices && ReduceOpId != ReduceTensorOp::NORM2;
    constexpr bool use_atomic_add = (out_support_atomic_add && op_support_atomic_add);

    // 1) If InDataType is half_t, must use half_t as AccDataType for indexable reduction operations
    // 2) If InDataType is half_t, must use float as AccDataType for non-indexable reduction
    // operations
    constexpr bool invalid_reduce_1 =
        std::is_same<InDataType, half_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, float>::value) ||
         (op_support_indices && !std::is_same<AccDataType, half_t>::value));

    // 1) If InDataType is float, must use float as AccDataType for indexable reduction operations
    constexpr bool invalid_reduce_2 =
        std::is_same<InDataType, float>::value &&
        (op_support_indices && !std::is_same<AccDataType, float>::value);

    // 1) The indices can only be used when the reduction operation is indexable
    constexpr bool invalid_reduce_3 = (!op_support_indices && UseIndex);

    // 1) If InDataType is int8_t, must use int8_t as AccDataType for indexable reduction operations
    // 2) If InDataType is int8_t, must use int32_t as AccDataType for non-indexable reduction
    // operations
    constexpr bool invalid_reduce_4 =
        std::is_same<InDataType, int8_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, int32_t>::value) ||
         (op_support_indices && !std::is_same<AccDataType, int8_t>::value));

    // 1) If InDataType is int8_t, the supported operation must be either indexable operations or
    // ADD/AVG
    constexpr bool invalid_reduce_5 = std::is_same<InDataType, int8_t>::value &&
                                      (!op_support_indices && ReduceOpId != ReduceTensorOp::ADD &&
                                       ReduceOpId != ReduceTensorOp::AVG);

    // 1) If InDataType is bhalf_t, must use float as AccDataType for all reduction operations
    constexpr bool invalid_reduce_6 =
        std::is_same<InDataType, bhalf_t>::value && !std::is_same<AccDataType, float>::value;

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2 || invalid_reduce_3 ||
                                     invalid_reduce_4 || invalid_reduce_5 || invalid_reduce_6);

    bool pass = true;

    if constexpr(!invalid_reduce)
    {
        Tensor<InDataType> in(inLengths);

        std::vector<size_t> outLengths;

        const auto invariantDims = get_invariant_dims<Rank, NumReduceDim>(reduceDims);

        if(reduceDims.size() == Rank)
            outLengths.push_back(1);
        else
            for(auto dim : invariantDims)
                outLengths.push_back(inLengths[dim]);

        Tensor<OutDataType> out_ref(outLengths);
        Tensor<OutDataType> out(outLengths);
        Tensor<int32_t> out_indices_ref(outLengths);
        Tensor<int32_t> out_indices(outLengths);

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
                in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
                break;
            case 2:
                in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
                break;
            default:
                in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0},
                                                num_thread);
            }

            if(beta != 0.0f)
                for(size_t i = 0; i < out_ref.mDesc.GetElementSpaceSize(); i++)
                    out.mData[i] = out_ref.mData[i];
        };

        // these buffers are usually provided by the user application
        DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
        DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

        in_dev.ToDevice(in.mData.data());

        if(beta != 0.0f)
            out_dev.ToDevice(out.mData.data());

        size_t indicesSizeInBytes = OutputIndex ? out.mDesc.GetElementSize() * sizeof(int) : 0;

        DeviceMem out_indices_dev(indicesSizeInBytes);

        float best_avg_time   = 0;
        float best_gb_per_sec = 0;

        using InElementwiseOperation =
            typename reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
        using AccElementwiseOperation =
            typename reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

        using ReduceOperation = typename reduce_binary_operator<ReduceOpId>::opType;

        InElementwiseOperation in_elementwise_op;
        AccElementwiseOperation acc_elementwise_op;

        std::tie(in_elementwise_op, acc_elementwise_op) =
            reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
                static_cast<int32_t>(reduce_total_length));

        using DeviceReduceInstPtr =
            DeviceReducePtr<Rank, NumReduceDim, InElementwiseOperation, AccElementwiseOperation>;

        std::vector<DeviceReduceInstPtr> reduce_ptrs;

        add_device_reduce_instance_threadwise<InDataType,
                                              AccDataType,
                                              OutDataType,
                                              Rank,
                                              NumReduceDim,
                                              ReduceOperation,
                                              InElementwiseOperation,
                                              AccElementwiseOperation,
                                              PropagateNan,
                                              UseIndex>(reduce_ptrs);

        add_device_reduce_instance_blockwise<InDataType,
                                             AccDataType,
                                             OutDataType,
                                             Rank,
                                             NumReduceDim,
                                             ReduceOperation,
                                             InElementwiseOperation,
                                             AccElementwiseOperation,
                                             PropagateNan,
                                             UseIndex>(reduce_ptrs);

        if constexpr(use_atomic_add)
        {
            add_device_reduce_instance_multiblock_atomic_add<InDataType,
                                                             AccDataType,
                                                             OutDataType,
                                                             Rank,
                                                             NumReduceDim,
                                                             ReduceOperation,
                                                             InElementwiseOperation,
                                                             AccElementwiseOperation,
                                                             PropagateNan,
                                                             UseIndex>(reduce_ptrs);
        }

        if(reduce_ptrs.empty())
        {
            throw std::runtime_error("Wrong! No device REDUCE instance found");
        };

        if(do_verification)
        {
            ReductionHost<InDataType,
                          AccDataType,
                          OutDataType,
                          ReduceOperation,
                          InElementwiseOperation,
                          AccElementwiseOperation,
                          Rank,
                          NumReduceDim,
                          PropagateNan,
                          OutputIndex>
                hostReduce(in.mDesc, out_ref.mDesc, invariantDims, reduceDims);

            hostReduce.Run(alpha,
                           in.mData.data(),
                           beta,
                           out_ref.mData.data(),
                           out_indices_ref.mData.data(),
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

        for(auto& reduce_ptr : reduce_ptrs)
        {
            auto argument_ptr = reduce_ptr->MakeArgumentPointer(arrInLengths,
                                                                arrInStrides,
                                                                arrOutLengths,
                                                                arrOutStrides,
                                                                reduceDims,
                                                                alpha,
                                                                beta,
                                                                in_dev.GetDeviceBuffer(),
                                                                nullptr,
                                                                out_dev.GetDeviceBuffer(),
                                                                out_indices_dev.GetDeviceBuffer(),
                                                                in_elementwise_op,
                                                                acc_elementwise_op);

            if(!reduce_ptr->IsSupportedArgument(argument_ptr.get()))
                continue;

            std::string reduce_name = reduce_ptr->GetTypeString();

            auto invoker_ptr = reduce_ptr->MakeInvokerPointer();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t num_bytes =
                invariant_total_length * reduce_total_length * sizeof(InDataType) +
                invariant_total_length * sizeof(OutDataType);

            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            if(time_kernel)
                std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, "
                          << reduce_name << std::endl;

            if(gb_per_sec > best_gb_per_sec)
            {
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                bool single_pass;

                out_dev.FromDevice(out.mData.data());
                single_pass = ck::utils::check_err(out, out_ref);

                if(OutputIndex)
                {
                    out_indices_dev.FromDevice(out_indices.mData.data());
                    single_pass = single_pass && ck::utils::check_err(out_indices, out_indices_ref);
                };

                if(!single_pass)
                {
                    std::cout << "Fail Info: " << reduce_ptr->GetTypeString() << std::endl;
                }

                pass = pass && single_pass;
            };

            if(do_dumpout)
            {
                dumpBufferToFile("dump_in.bin", in.mData.data(), in.mDesc.GetElementSize());
                dumpBufferToFile("dump_out.bin", out.mData.data(), out.mDesc.GetElementSize());
                dumpBufferToFile(
                    "dump_out_host.bin", out_ref.mData.data(), out_ref.mDesc.GetElementSize());
                if(OutputIndex)
                {
                    dumpBufferToFile("dump_indices.bin",
                                     out_indices.mData.data(),
                                     out_indices.mDesc.GetElementSize());
                    dumpBufferToFile("dump_indices_host.bin",
                                     out_indices_ref.mData.data(),
                                     out_indices_ref.mDesc.GetElementSize());
                };
            };
        };

        if(time_kernel)
            std::cout << "Best Perf: " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s"
                      << std::endl;
    }
    else
    {
        std::cout << "The requested reduction operation is not supported, please check !!!"
                  << std::endl;
    };

    return pass;
};

template <typename InDataType, typename AccDataType, typename OutDataType>
bool profile_reduce_impl(bool do_verification,
                         int init_method,
                         bool do_dumpout,
                         bool time_kernel,
                         const std::vector<size_t>& inLengths,
                         const std::vector<int>& reduceDims,
                         ReduceTensorOp ReduceOpId,
                         bool PropagateNan,
                         bool UseIndex,
                         float alpha,
                         float beta)
{
    bool matched = false;
    bool pass    = true;

    using tuple_of_description_instances =
        tensor_operation::device::instance::reduce_description_instances;

    const auto tuple_object = tuple_of_description_instances{};

    static_for<0, std::tuple_size<tuple_of_description_instances>::value, 1>{}([&](auto i) {
        if(matched)
            return;

        using descType = remove_cvref_t<decltype(std::get<i>(tuple_object))>;

        if(!description_match(
               descType{}, inLengths.size(), reduceDims, ReduceOpId, PropagateNan, UseIndex))
            return;

        std::array<ck::index_t, descType::NumReduceDim_> arrReduceDims;

        ck::ranges::copy(reduceDims, arrReduceDims.begin());

        pass = pass && profile_reduce_impl_impl<InDataType,
                                                AccDataType,
                                                OutDataType,
                                                descType::Rank_,
                                                descType::NumReduceDim_,
                                                static_cast<ReduceTensorOp>(descType::ReduceOpId_),
                                                descType::PropagateNan_,
                                                descType::UseIndex_>(do_verification,
                                                                     init_method,
                                                                     do_dumpout,
                                                                     time_kernel,
                                                                     inLengths,
                                                                     arrReduceDims,
                                                                     alpha,
                                                                     beta);

        matched = true;
    });

    return pass;
};

} // namespace profiler
} // namespace ck
