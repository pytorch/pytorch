// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename InDataType, typename OutDataType, typename AccDataType>
struct ReferenceSoftmax : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<InDataType>& in,
                 Tensor<OutDataType>& out,
                 AccDataType alpha,
                 AccDataType beta,
                 const std::vector<index_t> sm_reduce_dims,
                 Tensor<AccDataType>* sm_stats_ptr = nullptr)
            : in_(in),
              out_(out),
              alpha_(alpha),
              beta_(beta),
              sm_reduce_dims_(sm_reduce_dims),
              sm_stats_ptr_(sm_stats_ptr)
        {
            for(size_t i = 0; i < in.mDesc.GetNumOfDimension(); i++)
            {
                if(std::find(sm_reduce_dims.begin(), sm_reduce_dims.end(), i) ==
                   sm_reduce_dims.end())
                {
                    sm_stats_dims_.push_back(i);
                }
            }

            for(index_t dim : sm_stats_dims_)
            {
                sm_stats_lengths_.push_back(in_.mDesc.GetLengths()[dim]);
            }
            // max and sum reduction with final reduced values of dim=0 is a scalar so give it
            // appropriate lengths of {1}
            if(sm_stats_dims_.size() == 0)
            {
                sm_stats_lengths_.push_back(1);
            }
        }

        const Tensor<InDataType>& in_;
        Tensor<OutDataType>& out_;
        AccDataType alpha_;
        AccDataType beta_;
        std::vector<index_t> sm_reduce_dims_;
        std::vector<index_t> sm_stats_dims_; // dim after internal max/sum reduction
        std::vector<size_t> sm_stats_lengths_;
        Tensor<AccDataType>* sm_stats_ptr_; // max + ln(sum)
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            Tensor<AccDataType> reduce_max(arg.sm_stats_lengths_);
            reduce_max.GenerateTensorValue(
                GeneratorTensor_1<AccDataType>{std::numeric_limits<AccDataType>::lowest()});
            Tensor<AccDataType> reduce_sum(arg.sm_stats_lengths_);
            reduce_sum.GenerateTensorValue(GeneratorTensor_1<AccDataType>{0});

            // when final reduced values is of dim=0, the index will be transformed into empty
            // std::vector which is actually a valid input for Tensor::operator(std::vector) and
            // internally accesses 0'th element
            auto to_sm_stats_idx = [&](auto idx) {
                std::vector<size_t> sm_scalar_idx;
                for(index_t dim : arg.sm_stats_dims_)
                {
                    sm_scalar_idx.push_back(idx[dim]);
                }
                return sm_scalar_idx;
            };

            arg.in_.ForEach([&](auto& self, auto idx) {
                reduce_max(to_sm_stats_idx(idx)) = std::max(
                    reduce_max(to_sm_stats_idx(idx)), ck::type_convert<AccDataType>(self(idx)));
            });

            Tensor<AccDataType> in_stable(arg.in_.mDesc);
            in_stable.ForEach([&](auto& self, auto idx) {
                // numerator = exp(x - max(x))
                self(idx) = std::exp(ck::type_convert<AccDataType>(arg.in_(idx)) -
                                     reduce_max(to_sm_stats_idx(idx)));
            });

            in_stable.ForEach([&](auto& self, auto idx) {
                // denominator = sum(exp(x - max(x)))
                reduce_sum(to_sm_stats_idx(idx)) += self(idx);
            });

            if(arg.sm_stats_ptr_)
            {
                arg.sm_stats_ptr_->ForEach([&](auto& self, auto idx) {
                    self(idx) = reduce_max(idx) + std::log(reduce_sum(idx));
                });
            }

            arg.out_.ForEach([&](auto& self, auto idx) {
                AccDataType temp_result =
                    arg.alpha_ * in_stable(idx) / reduce_sum(to_sm_stats_idx(idx)) +
                    arg.beta_ * self(idx);
                self(idx) = ck::type_convert<OutDataType>(temp_result);
            });

            return 0;
        }

        float RunWithPreCalcStats(const Argument& arg)
        {
            if(arg.sm_stats_lengths_ != arg.sm_stats_ptr_[0].GetLengths())
            {
                throw std::runtime_error(
                    "softmax stats shape must match shape after softmax sum reduction op");
            }
            // when final reduced values is of dim=0, the index will be transformed into empty
            // std::vector which is actually a valid input for Tensor::operator(std::vector) and
            // internally accesses 0'th element
            auto to_sm_stats_idx = [&](auto idx) {
                std::vector<size_t> sm_scalar_idx;
                for(index_t dim : arg.sm_stats_dims_)
                {
                    sm_scalar_idx.push_back(idx[dim]);
                }
                return sm_scalar_idx;
            };

            // each element in stats corresponds to max + log(sum) after reduction
            // exp(x - max) / sum = exp(x - max) / exp(log(sum)) = exp(x - (max + log(sum)))
            arg.out_.ForEach([&](auto& self, auto idx) {
                self(idx) = arg.alpha_ * std::exp(ck::type_convert<AccDataType>(arg.in_(idx)) -
                                                  ck::type_convert<AccDataType>(
                                                      arg.sm_stats_ptr_[0](to_sm_stats_idx(idx)))) +
                            arg.beta_ * self(idx);
            });

            return 0;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<InDataType>& in,
                             Tensor<OutDataType>& out,
                             AccDataType alpha,
                             AccDataType beta,
                             const std::vector<index_t> sm_reduce_dims,
                             Tensor<AccDataType>* stats = nullptr)
    {
        return Argument{in, out, alpha, beta, sm_reduce_dims, stats};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceSoftmax"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
