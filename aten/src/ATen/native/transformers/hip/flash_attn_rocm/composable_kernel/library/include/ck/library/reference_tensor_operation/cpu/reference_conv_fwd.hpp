// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <type_traits>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

//
// @brief      Reference implementation for forward convolution.
//
// @paragraph
//             Tensor descriptor in GNCHW/GKCXY/GNKHW dimensional order
//             Supports both GNCHW/NGCHW as well as GNHWC/NHWGC physical layout
//             as long as dimensions in tensor descriptor is in GNCHW order
//
// @tparam     InDataType               Input tensor data type.
// @tparam     WeiDataType              Weights tensor data type.
// @tparam     OutDataType              Output tensor data type.
// @tparam     InElementwiseOperation   Functor for input tensor elementwise
//                                      operation.
// @tparam     WeiElementwiseOperation  Functor for weights tensor elementwise
//                                      operation.
// @tparam     NDimSpatial  Number of spatial dimensions.
//
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
struct ReferenceConvFwd : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<InDataType>& input,
                 const Tensor<WeiDataType>& weight,
                 Tensor<OutDataType>& output,
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

        const Tensor<InDataType>& input_;
        const Tensor<WeiDataType>& weight_;
        Tensor<OutDataType>& output_;

        std::vector<index_t> conv_strides_;
        std::vector<index_t> conv_dilations_;
        std::vector<index_t> in_left_pads_;
        std::vector<index_t> in_right_pads_;

        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceConvFwd::Argument;

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
                auto func = [&](auto g, auto n, auto k, auto wo) {
                    float v_acc = 0;

                    for(std::size_t c = 0; c < arg.weight_.GetLengths()[2]; ++c)
                    {
                        for(std::size_t x = 0; x < arg.weight_.GetLengths()[3]; ++x)
                        {
                            auto wi = static_cast<ck::long_index_t>(wo * arg.conv_strides_[0]) +
                                      static_cast<ck::long_index_t>(x * arg.conv_dilations_[0]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[0]);

                            if(wi >= 0 &&
                               ck::type_convert<std::size_t>(wi) < arg.input_.GetLengths()[3])
                            {
                                float v_in;
                                float v_wei;

                                arg.in_element_op_(
                                    v_in, ck::type_convert<float>(arg.input_(g, n, c, wi)));

                                arg.wei_element_op_(
                                    v_wei, ck::type_convert<float>(arg.weight_(g, k, c, x)));

                                v_acc += v_in * v_wei;
                            }
                        }
                    }

                    float v_out;

                    arg.out_element_op_(v_out, v_acc);

                    arg.output_(g, n, k, wo) = ck::type_convert<OutDataType>(v_out);
                };

                make_ParallelTensorFunctor(func,
                                           arg.output_.GetLengths()[0],
                                           arg.output_.GetLengths()[1],
                                           arg.output_.GetLengths()[2],
                                           arg.output_.GetLengths()[3])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 2)
            {
                auto func = [&](auto g, auto n, auto k, auto ho, auto wo) {
                    float v_acc = 0;

                    for(std::size_t c = 0; c < arg.weight_.GetLengths()[2]; ++c)
                    {
                        for(std::size_t y = 0; y < arg.weight_.GetLengths()[3]; ++y)
                        {
                            auto hi = static_cast<ck::long_index_t>(ho * arg.conv_strides_[0]) +
                                      static_cast<ck::long_index_t>(y * arg.conv_dilations_[0]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[0]);

                            for(std::size_t x = 0; x < arg.weight_.GetLengths()[4]; ++x)
                            {
                                auto wi =
                                    static_cast<ck::long_index_t>(wo * arg.conv_strides_[1]) +
                                    static_cast<ck::long_index_t>(x * arg.conv_dilations_[1]) -
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[1]);

                                if(hi >= 0 &&
                                   ck::type_convert<std::size_t>(hi) < arg.input_.GetLengths()[3] &&
                                   wi >= 0 &&
                                   ck::type_convert<std::size_t>(wi) < arg.input_.GetLengths()[4])
                                {
                                    float v_in;
                                    float v_wei;

                                    arg.in_element_op_(
                                        v_in, ck::type_convert<float>(arg.input_(g, n, c, hi, wi)));

                                    arg.wei_element_op_(
                                        v_wei, ck::type_convert<float>(arg.weight_(g, k, c, y, x)));

                                    v_acc += v_in * v_wei;
                                }
                            }
                        }
                    }

                    float v_out;

                    arg.out_element_op_(v_out, v_acc);

                    arg.output_(g, n, k, ho, wo) = ck::type_convert<OutDataType>(v_out);
                };

                make_ParallelTensorFunctor(func,
                                           arg.output_.GetLengths()[0],
                                           arg.output_.GetLengths()[1],
                                           arg.output_.GetLengths()[2],
                                           arg.output_.GetLengths()[3],
                                           arg.output_.GetLengths()[4])(
                    std::thread::hardware_concurrency());

                return 0;
            }
            else if constexpr(NDimSpatial == 3)
            {
                auto func = [&](auto g, auto n, auto k, auto d_o, auto ho, auto wo) {
                    float v_acc = 0;

                    for(std::size_t c = 0; c < arg.weight_.GetLengths()[2]; ++c)
                    {
                        for(std::size_t z = 0; z < arg.weight_.GetLengths()[3]; ++z)
                        {
                            auto di = static_cast<ck::long_index_t>(d_o * arg.conv_strides_[0]) +
                                      static_cast<ck::long_index_t>(z * arg.conv_dilations_[0]) -
                                      static_cast<ck::long_index_t>(arg.in_left_pads_[0]);
                            for(std::size_t y = 0; y < arg.weight_.GetLengths()[4]; ++y)
                            {
                                auto hi =
                                    static_cast<ck::long_index_t>(ho * arg.conv_strides_[1]) +
                                    static_cast<ck::long_index_t>(y * arg.conv_dilations_[1]) -
                                    static_cast<ck::long_index_t>(arg.in_left_pads_[1]);
                                for(std::size_t x = 0; x < arg.weight_.GetLengths()[5]; ++x)
                                {
                                    auto wi =
                                        static_cast<ck::long_index_t>(wo * arg.conv_strides_[2]) +
                                        static_cast<ck::long_index_t>(x * arg.conv_dilations_[2]) -
                                        static_cast<ck::long_index_t>(arg.in_left_pads_[2]);
                                    if(di >= 0 &&
                                       ck::type_convert<std::size_t>(di) <
                                           arg.input_.GetLengths()[3] &&
                                       hi >= 0 &&
                                       ck::type_convert<std::size_t>(hi) <
                                           arg.input_.GetLengths()[4] &&
                                       wi >= 0 &&
                                       ck::type_convert<std::size_t>(wi) <
                                           arg.input_.GetLengths()[5])
                                    {
                                        float v_in;
                                        float v_wei;

                                        arg.in_element_op_(v_in,
                                                           ck::type_convert<float>(
                                                               arg.input_(g, n, c, di, hi, wi)));

                                        arg.wei_element_op_(
                                            v_wei,
                                            ck::type_convert<float>(arg.weight_(g, k, c, z, y, x)));

                                        v_acc += v_in * v_wei;
                                    }
                                }
                            }
                        }
                    }

                    float v_out;

                    arg.out_element_op_(v_out, v_acc);

                    arg.output_(g, n, k, d_o, ho, wo) = ck::type_convert<OutDataType>(v_out);
                };

                make_ParallelTensorFunctor(func,
                                           arg.output_.GetLengths()[0],
                                           arg.output_.GetLengths()[1],
                                           arg.output_.GetLengths()[2],
                                           arg.output_.GetLengths()[3],
                                           arg.output_.GetLengths()[4],
                                           arg.output_.GetLengths()[5])(
                    std::thread::hardware_concurrency());

                return 0;
            }
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /*stream_config*/ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override
    {
        return NDimSpatial >= 1 && NDimSpatial <= 3;
    }

    static auto MakeArgument(const Tensor<InDataType>& input,
                             const Tensor<WeiDataType>& weight,
                             Tensor<OutDataType>& output,
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
        str << "ReferenceConvFwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
