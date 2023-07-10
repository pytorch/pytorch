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

// out[N, Ho, Wo, K] =
//     activate(in[N, Hi, Wi, C] * wei[K, Y, X, C] + bias[K]) + residual[N, Ho, Wo, K]
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct ReferenceConvFwd_Bias_Activation_Add : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<InDataType>& in_n_c_hi_wi,
                 const Tensor<WeiDataType>& wei_k_c_y_x,
                 Tensor<OutDataType>& out_n_k_ho_wo,
                 const Tensor<OutDataType>& bias_k,
                 const Tensor<OutDataType>& resi_n_k_ho_wo,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : in_n_c_hi_wi_{in_n_c_hi_wi},
              wei_k_c_y_x_{wei_k_c_y_x},
              out_n_k_ho_wo_{out_n_k_ho_wo},
              bias_k_{bias_k},
              resi_n_k_ho_wo_{resi_n_k_ho_wo},
              conv_strides_{conv_filter_strides},
              conv_dilations_{conv_filter_dilations},
              in_left_pads_{input_left_pads},
              in_right_pads_{input_right_pads},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
        }

        const Tensor<InDataType>& in_n_c_hi_wi_;
        const Tensor<WeiDataType>& wei_k_c_y_x_;
        Tensor<OutDataType>& out_n_k_ho_wo_;
        const Tensor<OutDataType>& bias_k_;
        const Tensor<OutDataType>& resi_n_k_ho_wo_;

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
        using Argument = ReferenceConvFwd_Bias_Activation_Add::Argument;

        float Run(const Argument& arg)
        {
            auto f_nchw = [&](auto n, auto k, auto ho, auto wo) {
                float v_acc = 0;

                for(std::size_t c = 0; c < arg.wei_k_c_y_x_.mDesc.GetLengths()[1]; ++c)
                {
                    for(std::size_t y = 0; y < arg.wei_k_c_y_x_.mDesc.GetLengths()[2]; ++y)
                    {
                        auto hi = ck::type_convert<ck::long_index_t>(ho * arg.conv_strides_[0]) +
                                  ck::type_convert<ck::long_index_t>(y * arg.conv_dilations_[0]) -
                                  ck::type_convert<ck::long_index_t>(arg.in_left_pads_[0]);
                        for(std::size_t x = 0; x < arg.wei_k_c_y_x_.mDesc.GetLengths()[3]; ++x)
                        {
                            auto wi =
                                ck::type_convert<ck::long_index_t>(wo * arg.conv_strides_[1]) +
                                ck::type_convert<ck::long_index_t>(x * arg.conv_dilations_[1]) -
                                ck::type_convert<ck::long_index_t>(arg.in_left_pads_[1]);
                            if(hi >= 0 &&
                               ck::type_convert<std::size_t>(hi) <
                                   arg.in_n_c_hi_wi_.mDesc.GetLengths()[2] &&
                               wi >= 0 &&
                               ck::type_convert<std::size_t>(wi) <
                                   arg.in_n_c_hi_wi_.mDesc.GetLengths()[3])
                            {
                                float v_in;
                                float v_wei;

                                arg.in_element_op_(
                                    v_in,
                                    static_cast<const float>(arg.in_n_c_hi_wi_(n, c, hi, wi)));
                                arg.wei_element_op_(
                                    v_wei, static_cast<const float>(arg.wei_k_c_y_x_(k, c, y, x)));

                                v_acc += v_in * v_wei;
                            }
                        }
                    }
                }

                float v_out;

                arg.out_element_op_(v_out,
                                    v_acc,
                                    static_cast<const float>(arg.bias_k_(k)),
                                    static_cast<const float>(arg.resi_n_k_ho_wo_(n, k, ho, wo)));

                arg.out_n_k_ho_wo_(n, k, ho, wo) = v_out;
            };

            make_ParallelTensorFunctor(f_nchw,
                                       arg.out_n_k_ho_wo_.mDesc.GetLengths()[0],
                                       arg.out_n_k_ho_wo_.mDesc.GetLengths()[1],
                                       arg.out_n_k_ho_wo_.mDesc.GetLengths()[2],
                                       arg.out_n_k_ho_wo_.mDesc.GetLengths()[3])(
                std::thread::hardware_concurrency());
            return 0;
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

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<InDataType>& in_n_c_hi_wi,
                             const Tensor<WeiDataType>& wei_k_c_y_x,
                             Tensor<OutDataType>& out_n_k_ho_wo,
                             const Tensor<OutDataType>& bias_k,
                             const Tensor<OutDataType>& resi_n_k_ho_wo,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{in_n_c_hi_wi,
                        wei_k_c_y_x,
                        out_n_k_ho_wo,
                        bias_k,
                        resi_n_k_ho_wo,
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
        str << "ReferenceConvFwd_Bias_Activation_Add"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
