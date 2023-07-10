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

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename AccDataType,
          typename AccElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
struct ReferenceLayernorm : public device::BaseOperator
{
    // TODO - support generic layernorm
    static_assert((Rank == 2 && NumReduceDim == 1), "Only support 2D version so far");

    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<XDataType>& x_m_n,
                 const Tensor<GammaDataType>& gamma_n,
                 const Tensor<BetaDataType>& beta_n,
                 Tensor<YDataType>& y_m_n,
                 AccElementwiseOperation acc_elementwise_op,
                 const std::vector<index_t> lengths,
                 const std::vector<index_t> reduceDims,
                 AccDataType epsilon)
            : x_m_n_(x_m_n),
              gamma_n_(gamma_n),
              beta_n_(beta_n),
              y_m_n_(y_m_n),
              acc_elementwise_op_(acc_elementwise_op),
              lengths_(lengths),
              reduceDims_(reduceDims),
              epsilon_(epsilon)
        {
        }

        const Tensor<XDataType> x_m_n_;
        const Tensor<XDataType> gamma_n_;
        const Tensor<XDataType> beta_n_;
        Tensor<YDataType>& y_m_n_;
        AccElementwiseOperation acc_elementwise_op_;
        std::vector<index_t> lengths_;
        std::vector<index_t> reduceDims_;
        AccDataType epsilon_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            int M = arg.lengths_[0];
            int N = arg.lengths_[1];

            Tensor<AccDataType> mean({M});
            Tensor<AccDataType> var({M});

            for(int m = 0; m < M; ++m)
            {
                mean(m) = 0;
                var(m)  = 0;

                for(int n = 0; n < N; ++n)
                {
                    auto x_val = ck::type_convert<AccDataType>(arg.x_m_n_(m, n));
                    mean(m) += x_val;
                    var(m) += x_val * x_val;
                }

                mean(m) = mean(m) / N;
                var(m)  = (var(m) / N) - (mean(m) * mean(m));
            }

            for(int m = 0; m < M; ++m)
            {
                for(int n = 0; n < N; ++n)
                {
                    auto x_val = ck::type_convert<AccDataType>(arg.x_m_n_(m, n));
                    auto y_val = (x_val - mean(m)) / sqrt(var(m) + arg.epsilon_);
                    y_val      = (y_val * arg.gamma_n_(n)) + arg.beta_n_(n);
                    arg.acc_elementwise_op_(y_val, y_val);
                    arg.y_m_n_(m, n) = ck::type_convert<YDataType>(y_val);
                }
            }

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

    bool IsSupportedArgument(const device::BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        // TODO - support generic layernorm
        if(p_arg_->lengths_.size() != 2)
            return false;

        if(p_arg_->reduceDims_.size() != 1)
            return false;

        if(p_arg_->reduceDims_[0] != 1)
            return false;

        return true;
    }

    static auto MakeArgument(const Tensor<XDataType>& x_m_n,
                             const Tensor<GammaDataType>& gamma_n,
                             const Tensor<BetaDataType>& beta_n,
                             Tensor<YDataType>& y_m_n,
                             AccElementwiseOperation acc_elementwise_op,
                             const std::vector<index_t> lengths,
                             const std::vector<index_t> reduceDims,
                             AccDataType epsilon)
    {
        return Argument{
            x_m_n, gamma_n, beta_n, y_m_n, acc_elementwise_op, lengths, reduceDims, epsilon};
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
        str << "ReferenceLayernorm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
