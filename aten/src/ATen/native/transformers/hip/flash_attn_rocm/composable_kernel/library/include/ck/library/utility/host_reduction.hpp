// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <array>
#include <functional>

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <int NDim>
static void get_all_indexes(const std::array<size_t, NDim>& dimLengths,
                            std::vector<std::array<size_t, NDim>>& indexes)
{
    static_assert(NDim >= 1, "NDim >= 1 is required to use this function!");

    if constexpr(NDim == 1)
    {
        for(size_t i = 0; i < dimLengths[0]; i++)
        {
            std::array<size_t, 1> index{i};

            indexes.push_back(index);
        };
    }
    else
    {
        std::array<size_t, NDim - 1> partial_dim_lengths;

        for(int i = 0; i < NDim - 1; i++)
            partial_dim_lengths[i] = dimLengths[i + 1];

        std::vector<std::array<size_t, NDim - 1>> partial_indexes;

        get_all_indexes<NDim - 1>(partial_dim_lengths, partial_indexes);

        for(size_t i = 0; i < dimLengths[0]; i++)
            for(const auto& index : partial_indexes)
            {
                std::array<size_t, NDim> extIndex;

                extIndex[0] = i;

                for(int k = 0; k < NDim - 1; k++)
                    extIndex[k + 1] = index[k];

                indexes.push_back(extIndex);
            };
    };
};

template <int NDim>
static size_t get_offset_from_index(const std::array<size_t, NDim>& strides,
                                    const std::array<size_t, NDim>& index)
{
    size_t offset = 0;

    for(int i = 0; i < NDim; i++)
        offset += strides[i] * index[i];

    return (offset);
};

template <int NDim>
static size_t get_offset_from_index(const std::vector<size_t>& strides,
                                    const std::array<size_t, NDim>& index)
{
    size_t offset = 0;

    for(int i = 0; i < NDim; i++)
        offset += strides[i] * index[i];

    return (offset);
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          typename ReduceOperation,
          typename InElementwiseOperation,
          typename AccElementwiseOperation,
          int Rank,
          int NumReduceDim,
          bool PropagateNan,
          bool OutputIndex>
struct ReductionHost
{
    using IndexDataType = int32_t;

    static constexpr int NumInvariantDim = Rank - NumReduceDim;

    std::vector<size_t> outStrides;

    IndexDataType divider;

    std::array<size_t, NumReduceDim> reduceLengths;
    std::array<size_t, NumReduceDim> reduceStrides;
    std::array<size_t, NumInvariantDim> invariantLengths;
    std::array<size_t, NumInvariantDim> invariantStrides;

    std::vector<std::array<size_t, NumReduceDim>> reduce_dim_indexes;
    std::vector<std::array<size_t, NumInvariantDim>> invariant_dim_indexes;

    ReductionHost(HostTensorDescriptor& inDesc,
                  HostTensorDescriptor& outDesc,
                  const std::array<int, NumInvariantDim> invariantDims,
                  const std::array<int, NumReduceDim> reduceDims)
    {
        // this->outLengths = to_int_vector(outDesc.GetLengths());
        this->outStrides = outDesc.GetStrides();

        int product = 1;

        for(int i = 0; i < NumReduceDim; i++)
        {
            reduceLengths[i] = inDesc.GetLengths()[reduceDims[i]];
            reduceStrides[i] = inDesc.GetStrides()[reduceDims[i]];
            product *= inDesc.GetLengths()[reduceDims[i]];
        };

        divider = product;

        for(int i = 0; i < NumInvariantDim; i++)
        {
            invariantLengths[i] = inDesc.GetLengths()[invariantDims[i]];
            invariantStrides[i] = inDesc.GetStrides()[invariantDims[i]];
        };

        reduce_dim_indexes.clear();
        get_all_indexes<NumReduceDim>(reduceLengths, reduce_dim_indexes);

        if constexpr(NumInvariantDim > 0)
        {
            invariant_dim_indexes.clear();
            get_all_indexes<NumInvariantDim>(invariantLengths, invariant_dim_indexes);
        };
    };

    void Run(float alpha,
             const InDataType* in_data,
             float beta,
             OutDataType* out_data,
             IndexDataType* out_indices,
             InElementwiseOperation in_elementwise_op,
             AccElementwiseOperation acc_elementwise_op)
    {
        if constexpr(OutputIndex)
        {
            RunImpl_with_index(
                alpha, in_data, beta, out_data, out_indices, in_elementwise_op, acc_elementwise_op);
        }
        else
        {
            RunImpl_no_index(alpha, in_data, beta, out_data, in_elementwise_op, acc_elementwise_op);
        };
    };

    void RunImpl_with_index(float alpha,
                            const InDataType* in_data,
                            float beta,
                            OutDataType* out_data,
                            IndexDataType* out_indices,
                            InElementwiseOperation in_elementwise_op,
                            AccElementwiseOperation acc_elementwise_op)
    {
        using ck::float_equal_one;
        using ck::float_equal_zero;
        using ck::type_convert;

        using Accumulation = ck::detail::AccumulateWithIndexAndNanCheck<PropagateNan,
                                                                        ReduceOperation,
                                                                        AccDataType,
                                                                        IndexDataType>;

        if constexpr(NumInvariantDim == 0)
        {
            AccDataType accuVal     = ReduceOperation::template GetIdentityValue<AccDataType>();
            IndexDataType accuIndex = 0;

            for(std::size_t i = 0; i < reduce_dim_indexes.size(); i++)
            {
                auto offset_reduce =
                    get_offset_from_index<NumReduceDim>(reduceStrides, reduce_dim_indexes[i]);

                auto currVal = type_convert<AccDataType>(in_data[offset_reduce]);

                in_elementwise_op(currVal, currVal);

                auto currIndex = static_cast<IndexDataType>(i);

                Accumulation::Calculate(accuVal, currVal, accuIndex, currIndex);
            };

            acc_elementwise_op(accuVal, accuVal);

            if(!float_equal_one{}(alpha))
                accuVal *= type_convert<AccDataType>(alpha);

            if(!float_equal_zero{}(beta))
                accuVal += type_convert<AccDataType>(out_data[0]) * type_convert<AccDataType>(beta);

            out_data[0]    = type_convert<OutDataType>(accuVal);
            out_indices[0] = accuIndex;
        }
        else
        {
            auto thread_reduce_func = [&](auto invariant_index) {
                AccDataType accuVal     = ReduceOperation::template GetIdentityValue<AccDataType>();
                IndexDataType accuIndex = 0;

                auto offset_invariant =
                    get_offset_from_index<NumInvariantDim>(invariantStrides, invariant_index);

                for(std::size_t i = 0; i < reduce_dim_indexes.size(); i++)
                {
                    auto offset_reduce =
                        get_offset_from_index<NumReduceDim>(reduceStrides, reduce_dim_indexes[i]);

                    auto currVal =
                        type_convert<AccDataType>(in_data[offset_invariant + offset_reduce]);

                    in_elementwise_op(currVal, currVal);

                    auto currIndex = static_cast<IndexDataType>(i);

                    Accumulation::Calculate(accuVal, currVal, accuIndex, currIndex);
                };

                acc_elementwise_op(accuVal, accuVal);

                if(!float_equal_one{}(alpha))
                    accuVal *= type_convert<AccDataType>(alpha);

                auto dst_offset =
                    get_offset_from_index<NumInvariantDim>(outStrides, invariant_index);

                if(!float_equal_zero{}(beta))
                    accuVal += type_convert<AccDataType>(out_data[dst_offset]) *
                               type_convert<AccDataType>(beta);

                out_data[dst_offset]    = type_convert<OutDataType>(accuVal);
                out_indices[dst_offset] = accuIndex;
            };

            std::size_t num_thread = 1;
            std::size_t work_per_thread =
                (invariant_dim_indexes.size() + num_thread - 1) / num_thread;

            std::vector<joinable_thread> threads(num_thread);

            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t iw_begin = it * work_per_thread;
                std::size_t iw_end =
                    std::min((it + 1) * work_per_thread, invariant_dim_indexes.size());

                auto f = [=] {
                    for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                    {
                        thread_reduce_func(invariant_dim_indexes[iw]);
                    }
                };

                threads[it] = joinable_thread(f);
            }
        };
    };

    void RunImpl_no_index(float alpha,
                          const InDataType* in_data,
                          float beta,
                          OutDataType* out_data,
                          InElementwiseOperation in_elementwise_op,
                          AccElementwiseOperation acc_elementwise_op)
    {
        using ck::float_equal_one;
        using ck::float_equal_zero;
        using ck::type_convert;

        using Accumulation =
            ck::detail::AccumulateWithNanCheck<PropagateNan, ReduceOperation, AccDataType>;

        if constexpr(NumInvariantDim == 0)
        {
            AccDataType accuVal = ReduceOperation::template GetIdentityValue<AccDataType>();

            for(const auto& reduce_index : reduce_dim_indexes)
            {
                auto offset_reduce =
                    get_offset_from_index<NumReduceDim>(reduceStrides, reduce_index);

                auto currVal = type_convert<AccDataType>(in_data[offset_reduce]);

                in_elementwise_op(currVal, currVal);

                Accumulation::Calculate(accuVal, currVal);
            };

            acc_elementwise_op(accuVal, accuVal);

            if(!float_equal_one{}(alpha))
                accuVal *= type_convert<AccDataType>(alpha);

            if(!float_equal_zero{}(beta))
                accuVal += type_convert<AccDataType>(out_data[0]) * type_convert<AccDataType>(beta);

            out_data[0] = type_convert<OutDataType>(accuVal);
        }
        else
        {
            auto thread_reduce_func = [&](auto invariant_index) {
                AccDataType accuVal = ReduceOperation::template GetIdentityValue<AccDataType>();

                auto offset_invariant =
                    get_offset_from_index<NumInvariantDim>(invariantStrides, invariant_index);

                for(const auto& reduce_index : reduce_dim_indexes)
                {
                    auto offset_reduce =
                        get_offset_from_index<NumReduceDim>(reduceStrides, reduce_index);

                    auto currVal =
                        type_convert<AccDataType>(in_data[offset_invariant + offset_reduce]);

                    in_elementwise_op(currVal, currVal);

                    Accumulation::Calculate(accuVal, currVal);
                };

                acc_elementwise_op(accuVal, accuVal);

                if(!float_equal_one{}(alpha))
                    accuVal *= type_convert<AccDataType>(alpha);

                auto dst_offset =
                    get_offset_from_index<NumInvariantDim>(outStrides, invariant_index);

                if(!float_equal_zero{}(beta))
                    accuVal += type_convert<AccDataType>(out_data[dst_offset]) *
                               type_convert<AccDataType>(beta);

                out_data[dst_offset] = type_convert<OutDataType>(accuVal);
            };

            std::size_t num_thread = 1;
            std::size_t work_per_thread =
                (invariant_dim_indexes.size() + num_thread - 1) / num_thread;

            std::vector<joinable_thread> threads(num_thread);

            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t iw_begin = it * work_per_thread;
                std::size_t iw_end =
                    std::min((it + 1) * work_per_thread, invariant_dim_indexes.size());

                auto f = [=] {
                    for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                    {
                        thread_reduce_func(invariant_dim_indexes[iw]);
                    }
                };

                threads[it] = joinable_thread(f);
            }
        };
    };
};
