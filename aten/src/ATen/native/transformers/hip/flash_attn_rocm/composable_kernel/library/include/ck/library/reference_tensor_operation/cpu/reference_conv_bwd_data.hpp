// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

// input descriptor in [G, N, C, Do, Ho, Wo] order
// weight descriptor in [G, K, C, Z, Y, X] order
// output descriptor in [G, N, K, Di, Hi, Wi] order
// phyiscal layout is irrelavent
template <ck::index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename std::enable_if<NDimSpatial >= 1 && NDimSpatial <= 3, bool>::type = false>
struct ReferenceConvBwdData : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(Tensor<InDataType>& input,
                 const Tensor<WeiDataType>& weight,
                 const Tensor<OutDataType>& output,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : input_{input},
              weight_{weight},
              output_{output},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        Tensor<InDataType>& input_;
        const Tensor<WeiDataType>& weight_;
        const Tensor<OutDataType>& output_;

        std::vector<index_t> conv_strides_;
        std::vector<index_t> conv_dilations_;
        std::vector<index_t> in_left_pads_;
        std::vector<index_t> in_right_pads_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceConvBwdData::Argument;

        float Run(const Argument& arg)
        {
            if(!(arg.input_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.weight_.GetNumOfDimension() == NDimSpatial + 3 &&
                 arg.output_.GetNumOfDimension() == NDimSpatial + 3))
            {
                throw std::runtime_error("wrong! inconsistent dimension");
            }

            if constexpr(NDimSpatial == 1)
            {
                auto f_ncw = [&](auto g, auto n, auto c, auto wi) {
                    std::size_t K  = arg.weight_.GetLengths()[1];
                    std::size_t X  = arg.weight_.GetLengths()[3];
                    std::size_t Wo = arg.output_.GetLengths()[3];

                    float v_acc = 0;

                    for(std::size_t x = 0; x < X; ++x)
                    {
                        auto w_tmp = static_cast<ck::long_index_t>(wi) +
                                     static_cast<ck::long_index_t>(arg.in_left_pads_[0]) -
                                     static_cast<ck::long_index_t>(x * arg.conv_dilations_[0]);

                        if(w_tmp % arg.conv_strides_[0] == 0)
                        {
                            auto wo = static_cast<ck::long_index_t>(w_tmp) /
                                      static_cast<ck::long_index_t>(arg.conv_strides_[0]);

                            if(wo >= 0 && ck::type_convert<std::size_t>(wo) < Wo)
                            {
                                for(std::size_t k = 0; k < K; ++k)
                                {
                                    float v_out = 0;
                                    float v_wei = 0;

                                    arg.out_element_op_(
                                        v_out, ck::type_convert<float>(arg.output_(g, n, k, wo)));

                                    arg.wei_element_op_(
                                        v_wei, ck::type_convert<float>(arg.weight_(g, k, c, x)));

                                    v_acc += v_out * v_wei;
                                }
                            }
                        }
                    }

                    float v_in;

                    arg.in_element_op_(v_in, v_acc);

                    arg.input_(g, n, c, wi) = ck::type_convert<InDataType>(v_acc);
                };

                make_ParallelTensorFunctor(f_ncw,
                                           arg.input_.GetLengths()[0],
                                           arg.input_.GetLengths()[1],
                                           arg.input_.GetLengths()[2],
                                           arg.input_.GetLengths()[3])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 2)
            {
                auto f_nchw = [&](auto g, auto n, auto c, auto hi, auto wi) {
                    std::size_t K = arg.weight_.GetLengths()[1];
                    std::size_t Y = arg.weight_.GetLengths()[3];
                    std::size_t X = arg.weight_.GetLengths()[4];

                    std::size_t Ho = arg.output_.GetLengths()[3];
                    std::size_t Wo = arg.output_.GetLengths()[4];

                    float v_acc = 0;

                    for(std::size_t y = 0; y < Y; ++y)
                    {
                        auto h_tmp = static_cast<ck::long_index_t>(hi) +
                                     static_cast<ck::long_index_t>(arg.in_left_pads_[0]) -
                                     static_cast<ck::long_index_t>(y * arg.conv_dilations_[0]);
                        if(h_tmp % arg.conv_strides_[0] == 0)
                        {
                            auto ho = static_cast<ck::long_index_t>(h_tmp) /
                                      static_cast<ck::long_index_t>(arg.conv_strides_[0]);
                            if(ho >= 0 && ck::type_convert<std::size_t>(ho) < Ho)
                            {
                                for(std::size_t x = 0; x < X; ++x)
                                {
                                    auto w_tmp =
                                        static_cast<ck::long_index_t>(wi) +
                                        static_cast<ck::long_index_t>(arg.in_left_pads_[1]) -
                                        static_cast<ck::long_index_t>(x * arg.conv_dilations_[1]);
                                    if(w_tmp % arg.conv_strides_[1] == 0)
                                    {
                                        auto wo =
                                            static_cast<ck::long_index_t>(w_tmp) /
                                            static_cast<ck::long_index_t>(arg.conv_strides_[1]);
                                        if(wo >= 0 && ck::type_convert<std::size_t>(wo) < Wo)
                                        {
                                            for(std::size_t k = 0; k < K; ++k)
                                            {
                                                float v_out = 0;
                                                float v_wei = 0;

                                                arg.out_element_op_(
                                                    v_out,
                                                    ck::type_convert<float>(
                                                        arg.output_(g, n, k, ho, wo)));

                                                arg.wei_element_op_(
                                                    v_wei,
                                                    ck::type_convert<float>(
                                                        arg.weight_(g, k, c, y, x)));

                                                v_acc += v_out * v_wei;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    float v_in;

                    arg.in_element_op_(v_in, v_acc);

                    arg.input_(g, n, c, hi, wi) = ck::type_convert<InDataType>(v_acc);
                };

                make_ParallelTensorFunctor(f_nchw,
                                           arg.input_.GetLengths()[0],
                                           arg.input_.GetLengths()[1],
                                           arg.input_.GetLengths()[2],
                                           arg.input_.GetLengths()[3],
                                           arg.input_.GetLengths()[4])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 3)
            {
                auto f_ncdhw = [&](auto g, auto n, auto c, auto di, auto hi, auto wi) {
                    std::size_t K = arg.weight_.GetLengths()[1];
                    std::size_t Z = arg.weight_.GetLengths()[3];
                    std::size_t Y = arg.weight_.GetLengths()[4];
                    std::size_t X = arg.weight_.GetLengths()[5];

                    std::size_t Do = arg.output_.GetLengths()[3];
                    std::size_t Ho = arg.output_.GetLengths()[4];
                    std::size_t Wo = arg.output_.GetLengths()[5];

                    float v_acc = 0;

                    for(std::size_t z = 0; z < Z; ++z)
                    {
                        auto d_tmp = static_cast<ck::long_index_t>(di) +
                                     static_cast<ck::long_index_t>(arg.in_left_pads_[0]) -
                                     static_cast<ck::long_index_t>(z * arg.conv_dilations_[0]);
                        if(d_tmp % arg.conv_strides_[0] == 0)
                        {
                            auto do_ = static_cast<ck::long_index_t>(d_tmp) /
                                       static_cast<ck::long_index_t>(arg.conv_strides_[0]);
                            if(do_ >= 0 && ck::type_convert<std::size_t>(do_) < Do)
                            {
                                for(std::size_t y = 0; y < Y; ++y)
                                {
                                    auto h_tmp =
                                        static_cast<ck::long_index_t>(hi) +
                                        static_cast<ck::long_index_t>(arg.in_left_pads_[1]) -
                                        static_cast<ck::long_index_t>(y * arg.conv_dilations_[1]);
                                    if(h_tmp % arg.conv_strides_[1] == 0)
                                    {
                                        auto ho =
                                            static_cast<ck::long_index_t>(h_tmp) /
                                            static_cast<ck::long_index_t>(arg.conv_strides_[1]);
                                        if(ho >= 0 && ck::type_convert<std::size_t>(ho) < Ho)
                                        {
                                            for(std::size_t x = 0; x < X; ++x)
                                            {
                                                auto w_tmp = static_cast<ck::long_index_t>(wi) +
                                                             static_cast<ck::long_index_t>(
                                                                 arg.in_left_pads_[2]) -
                                                             static_cast<ck::long_index_t>(
                                                                 x * arg.conv_dilations_[2]);

                                                if(w_tmp % arg.conv_strides_[2] == 0)
                                                {
                                                    auto wo = static_cast<ck::long_index_t>(w_tmp) /
                                                              static_cast<ck::long_index_t>(
                                                                  arg.conv_strides_[2]);
                                                    if(wo >= 0 &&
                                                       ck::type_convert<std::size_t>(wo) < Wo)
                                                    {
                                                        for(std::size_t k = 0; k < K; ++k)
                                                        {
                                                            float v_out = 0;
                                                            float v_wei = 0;

                                                            arg.out_element_op_(
                                                                v_out,
                                                                ck::type_convert<float>(arg.output_(
                                                                    g, n, k, do_, ho, wo)));

                                                            arg.wei_element_op_(
                                                                v_wei,
                                                                ck::type_convert<float>(
                                                                    arg.weight_(g, k, c, z, y, x)));

                                                            v_acc += v_out * v_wei;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    float v_in;

                    arg.in_element_op_(v_in, v_acc);

                    arg.input_(g, n, c, di, hi, wi) = ck::type_convert<InDataType>(v_acc);
                };

                make_ParallelTensorFunctor(f_ncdhw,
                                           arg.input_.GetLengths()[0],
                                           arg.input_.GetLengths()[1],
                                           arg.input_.GetLengths()[2],
                                           arg.input_.GetLengths()[3],
                                           arg.input_.GetLengths()[4],
                                           arg.input_.GetLengths()[5])(
                    std::thread::hardware_concurrency());

                return 0;
            }
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

    static auto MakeArgument(Tensor<InDataType>& input,
                             const Tensor<WeiDataType>& weight,
                             const Tensor<OutDataType>& output,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{input,
                        weight,
                        output,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
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
        str << "ReferenceConvBwdData"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
