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
          typename AccElementwiseOperation>
struct ReferenceGroupnorm : public device::BaseOperator
{
    // x = [N, H, W, G, C]
    // y = [N, H, W, G, C]
    // reduce dim [H, W, C], mean, var = [N, G]
    // gamma, beta = [G, C]
    // beta: [G, C]
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<XDataType>& x,
                 const Tensor<GammaDataType>& gamma,
                 const Tensor<BetaDataType>& beta,
                 Tensor<YDataType>& y,
                 AccElementwiseOperation acc_elementwise_op,
                 const std::vector<index_t> lengths,
                 AccDataType epsilon)
            : x_(x),
              gamma_(gamma),
              beta_(beta),
              y_(y),
              acc_elementwise_op_(acc_elementwise_op),
              lengths_(lengths),
              epsilon_(epsilon)
        {
        }

        const Tensor<XDataType> x_;
        const Tensor<XDataType> gamma_;
        const Tensor<XDataType> beta_;
        Tensor<YDataType>& y_;
        AccElementwiseOperation acc_elementwise_op_;
        std::vector<index_t> lengths_;
        AccDataType epsilon_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            int N = arg.lengths_[0];
            int H = arg.lengths_[1];
            int W = arg.lengths_[2];
            int G = arg.lengths_[3];
            int C = arg.lengths_[4];

            Tensor<AccDataType> mean({N, G});
            Tensor<AccDataType> var({N, G});

            // Compute mean & var in [H, W, C] by Welford Algorithm
            // TODO - parallel for each HWC
            // TODO - address calculation
            for(int n = 0; n < N; ++n)
            {
                for(int g = 0; g < G; ++g)
                {
                    AccDataType mean_val = type_convert<AccDataType>(0.0f);
                    AccDataType var_val  = type_convert<AccDataType>(0.0f);
                    int32_t curr_count   = 0;

                    for(int h = 0; h < H; ++h)
                    {
                        for(int w = 0; w < W; ++w)
                        {
                            for(int c = 0; c < C; ++c)
                            {
                                curr_count++;
                                AccDataType x = type_convert<AccDataType>(arg.x_(n, h, w, g, c));
                                AccDataType delta = x - mean_val;
                                mean_val += delta / curr_count;
                                AccDataType delta2 = x - mean_val;
                                var_val += delta * delta2;
                            }
                        }
                    }

                    mean(n, g) = mean_val;
                    var(n, g)  = var_val / curr_count;
                }
            }

            // Normalization
            for(int n = 0; n < N; ++n)
            {
                for(int h = 0; h < H; ++h)
                {
                    for(int w = 0; w < W; ++w)
                    {
                        for(int g = 0; g < G; ++g)
                        {
                            for(int c = 0; c < C; ++c)
                            {
                                AccDataType x = type_convert<AccDataType>(arg.x_(n, h, w, g, c));
                                AccDataType gamma    = type_convert<AccDataType>(arg.gamma_(g, c));
                                AccDataType beta     = type_convert<AccDataType>(arg.beta_(g, c));
                                AccDataType mean_val = type_convert<AccDataType>(mean(n, g));
                                AccDataType var_val  = type_convert<AccDataType>(var(n, g));
                                AccDataType y        = gamma * (x - mean_val) /
                                                    ck::math::sqrt(arg.epsilon_ + var_val) +
                                                beta;
                                arg.acc_elementwise_op_(y, y);
                                arg.y_(n, h, w, g, c) = type_convert<YDataType>(y);
                            }
                        }
                    }
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
        if(p_arg_->lengths_.size() != 5)
            return false;

        return true;
    }

    static auto MakeArgument(const Tensor<XDataType>& x,
                             const Tensor<GammaDataType>& gamma,
                             const Tensor<BetaDataType>& beta,
                             Tensor<YDataType>& y,
                             AccElementwiseOperation acc_elementwise_op,
                             const std::vector<index_t> lengths,
                             AccDataType epsilon)
    {
        return Argument{x, gamma, beta, y, acc_elementwise_op, lengths, epsilon};
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
