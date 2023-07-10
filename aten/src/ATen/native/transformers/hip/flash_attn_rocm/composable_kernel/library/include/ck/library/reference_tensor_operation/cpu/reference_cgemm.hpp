// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

// FIXME: support arbitrary elementwise operation for A/B/C
template <
    typename ADataType,
    typename BDataType,
    typename CDataType,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
    enable_if_t<
        is_same_v<AElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<BElementwiseOperation, ck::tensor_operation::element_wise::PassThrough> &&
            is_same_v<CElementwiseOperation, ck::tensor_operation::element_wise::PassThrough>,
        bool> = false>
struct ReferenceCGemm : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_m_k_real,
                 const Tensor<ADataType>& a_m_k_imag,
                 const Tensor<BDataType>& b_k_n_real,
                 const Tensor<BDataType>& b_k_n_imag,
                 Tensor<CDataType>& c_m_n_real,
                 Tensor<CDataType>& c_m_n_imag,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_m_k_real_{a_m_k_real},
              a_m_k_imag_{a_m_k_imag},
              b_k_n_real_{b_k_n_real},
              b_k_n_imag_{b_k_n_imag},
              c_m_n_real_{c_m_n_real},
              c_m_n_imag_{c_m_n_imag},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_m_k_real_;
        const Tensor<ADataType>& a_m_k_imag_;
        const Tensor<BDataType>& b_k_n_real_;
        const Tensor<BDataType>& b_k_n_imag_;
        Tensor<CDataType>& c_m_n_real_;
        Tensor<CDataType>& c_m_n_imag_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceCGemm::Argument;

        float Run(const Argument& arg)
        {
            const std::size_t K = arg.a_m_k_real_.mDesc.GetLengths()[1];

            if(K != arg.a_m_k_imag_.mDesc.GetLengths()[1])
            {
                throw std::runtime_error("wrong! Incompatible real and imag sizes in CGEMM");
            }

            auto f_mk_kn_mn_real = [&](auto m, auto n) {
                float v_c_real = 0;

                for(std::size_t k = 0; k < K; ++k)
                {
                    float v_a_real = ck::type_convert<float>(arg.a_m_k_real_(m, k));
                    float v_a_imag = ck::type_convert<float>(arg.a_m_k_imag_(m, k));
                    float v_b_real = ck::type_convert<float>(arg.b_k_n_real_(k, n));
                    float v_b_imag = ck::type_convert<float>(arg.b_k_n_imag_(k, n));

                    v_c_real += v_a_real * v_b_real - v_a_imag * v_b_imag;
                }

                arg.c_m_n_real_(m, n) = ck::type_convert<CDataType>(v_c_real);
            };

            auto f_mk_kn_mn_imag = [&](auto m, auto n) {
                float v_c_imag = 0;

                for(std::size_t k = 0; k < K; ++k)
                {
                    float v_a_real = ck::type_convert<float>(arg.a_m_k_real_(m, k));
                    float v_a_imag = ck::type_convert<float>(arg.a_m_k_imag_(m, k));
                    float v_b_real = ck::type_convert<float>(arg.b_k_n_real_(k, n));
                    float v_b_imag = ck::type_convert<float>(arg.b_k_n_imag_(k, n));

                    v_c_imag += v_a_real * v_b_imag + v_a_imag * v_b_real;
                }

                arg.c_m_n_imag_(m, n) = ck::type_convert<CDataType>(v_c_imag);
            };

            make_ParallelTensorFunctor(f_mk_kn_mn_real,
                                       arg.c_m_n_real_.mDesc.GetLengths()[0],
                                       arg.c_m_n_real_.mDesc.GetLengths()[1])(
                std::thread::hardware_concurrency());
            make_ParallelTensorFunctor(f_mk_kn_mn_imag,
                                       arg.c_m_n_imag_.mDesc.GetLengths()[0],
                                       arg.c_m_n_imag_.mDesc.GetLengths()[1])(
                std::thread::hardware_concurrency());

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

    static auto MakeArgument(const Tensor<ADataType>& a_m_k_real,
                             const Tensor<ADataType>& a_m_k_imag,
                             const Tensor<BDataType>& b_k_n_real,
                             const Tensor<BDataType>& b_k_n_imag,
                             Tensor<CDataType>& c_m_n_real,
                             Tensor<CDataType>& c_m_n_imag,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{a_m_k_real,
                        a_m_k_imag,
                        b_k_n_real,
                        b_k_n_imag,
                        c_m_n_real,
                        c_m_n_imag,
                        a_element_op,
                        b_element_op,
                        c_element_op};
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
        str << "ReferenceCGemm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
