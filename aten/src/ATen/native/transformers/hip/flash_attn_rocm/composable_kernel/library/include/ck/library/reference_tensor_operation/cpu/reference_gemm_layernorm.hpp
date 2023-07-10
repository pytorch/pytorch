// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

// D = Layernorm(acc_element_op(A * B + broadcast(bias)) + add) * broadcast(gamma) + broadcast(beta)
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename C0DataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename CElementwiseOperation>
struct ReferenceGemmLayernorm : public device::BaseOperator
{
    using ReferenceGemmInstance = ReferenceGemm<ADataType,
                                                BDataType,
                                                AccDataType,
                                                AccDataType,
                                                AElementwiseOperation,
                                                BElementwiseOperation,
                                                element_wise::PassThrough>;

    template <typename InDataType, typename OutDataType, typename ComputeDataType>
    static void RunLayernorm(Tensor<OutDataType>& result,
                             const Tensor<ComputeDataType>& acc, // MxN
                             const Tensor<InDataType>& gamma,    // 1xN
                             const Tensor<InDataType>& beta,     // 1xN
                             const InDataType epsilon = 1e-5)
    {
        assert(acc.mDesc.GetLengths()[1] == gamma.mDesc.GetLengths()[0] &&
               acc.mDesc.GetLengths()[1] == beta.mDesc.GetLengths()[0]);

        size_t M = acc.mDesc.GetLengths()[0];
        size_t N = acc.mDesc.GetLengths()[1];

        Tensor<ComputeDataType> avg_acc_sq({M});
        Tensor<ComputeDataType> avg_acc({M});
        Tensor<ComputeDataType> acc_layernorm(acc);

        // reduce N dim
        for(size_t i = 0; i < M; i++)
        {
            ComputeDataType sum_acc_sq = 0;
            ComputeDataType sum_acc    = 0;
            for(size_t j = 0; j < N; j++)
            {
                sum_acc_sq += acc_layernorm(i, j) * acc_layernorm(i, j);
                sum_acc += acc_layernorm(i, j);
            }
            avg_acc_sq(i) = sum_acc_sq / N;
            avg_acc(i)    = sum_acc / N;
        }

        // normalize
        acc_layernorm.ForEach([&](auto& self, auto idx) {
            self(idx[0], idx[1]) =
                (self(idx[0], idx[1]) - avg_acc(idx[0])) /
                sqrt(avg_acc_sq(idx[0]) - avg_acc(idx[0]) * avg_acc(idx[0]) + epsilon);
        });

        // affine
        acc_layernorm.ForEach([&](auto& self, auto idx) {
            self(idx[0], idx[1]) = self(idx[0], idx[1]) * gamma(idx[1]) + beta(idx[1]);
        });

        // cast
        result = acc_layernorm.template CopyAsType<OutDataType>();
    }

    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_m_k,
                 const Tensor<BDataType>& b_k_n,
                 Tensor<CDataType>& c_m_n,
                 const Tensor<C0DataType>& c0_n_bias,  // 1xN
                 const Tensor<C0DataType>& c0_m_n_add, // MxN
                 const Tensor<C0DataType>& c0_n_gamma, // 1xN
                 const Tensor<C0DataType>& c0_n_beta,  // 1xN
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 AccElementwiseOperation acc_element_op,
                 CElementwiseOperation c_element_op,
                 const CDataType epsilon = 1e-5)
            : a_m_k_{a_m_k},
              b_k_n_{b_k_n},
              c_m_n_{c_m_n},
              c0_n_bias_{c0_n_bias},
              c0_m_n_add_{c0_m_n_add},
              c0_n_gamma_{c0_n_gamma},
              c0_n_beta_{c0_n_beta},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              acc_element_op_{acc_element_op},
              c_element_op_{c_element_op},
              epsilon_{epsilon}
        {
        }

        const Tensor<ADataType>& a_m_k_;
        const Tensor<BDataType>& b_k_n_;
        Tensor<CDataType>& c_m_n_;
        const Tensor<C0DataType>& c0_n_bias_;
        const Tensor<C0DataType>& c0_m_n_add_;
        const Tensor<C0DataType>& c0_n_gamma_;
        const Tensor<C0DataType>& c0_n_beta_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        AccElementwiseOperation acc_element_op_;
        CElementwiseOperation c_element_op_;

        const CDataType epsilon_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        // using Argument = ReferenceGemm::Argument;

        float Run(const Argument& arg)
        {
            Tensor<AccDataType> acc_m_n(arg.c_m_n_.mDesc);
            acc_m_n.GenerateTensorValue(GeneratorTensor_1<AccDataType>{0});

            auto ref_gemm     = ReferenceGemmInstance{};
            auto ref_invoker  = ref_gemm.MakeInvoker();
            auto ref_argument = ref_gemm.MakeArgument(arg.a_m_k_,
                                                      arg.b_k_n_,
                                                      acc_m_n,
                                                      arg.a_element_op_,
                                                      arg.b_element_op_,
                                                      element_wise::PassThrough{});

            // gemm
            ref_invoker.Run(ref_argument);

            // activation(acc + bias)
            acc_m_n.ForEach([&](auto& self, auto idx) {
                AccDataType out;
                arg.acc_element_op_(out, acc_m_n(idx[0], idx[1]) + arg.c0_n_bias_(idx[1]));
                self(idx[0], idx[1]) = out;
            });

            // add from other layers
            acc_m_n.ForEach([&](auto& self, auto idx) {
                self(idx[0], idx[1]) += arg.c0_m_n_add_(idx[0], idx[1]);
            });

            // layernorm
            RunLayernorm(arg.c_m_n_, acc_m_n, arg.c0_n_gamma_, arg.c0_n_beta_);

            // elementwise op
            arg.c_m_n_.ForEach([&](auto& self, auto idx) {
                arg.c_element_op_(self(idx[0], idx[1]), self(idx[0], idx[1]));
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

    static auto MakeArgument(const Tensor<ADataType>& a_m_k,
                             const Tensor<BDataType>& b_k_n,
                             Tensor<CDataType>& c_m_n,
                             const Tensor<C0DataType>& c0_n_bias,  // 1xN
                             const Tensor<C0DataType>& c0_m_n_add, // 1xN
                             const Tensor<C0DataType>& c0_n_gamma, // 1xN
                             const Tensor<C0DataType>& c0_n_beta,  // 1xN
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             AccElementwiseOperation acc_element_op,
                             CElementwiseOperation c_element_op,
                             const CDataType epsilon = 1e-5)
    {
        return Argument{a_m_k,
                        b_k_n,
                        c_m_n,
                        c0_n_bias,
                        c0_m_n_add,
                        c0_n_gamma,
                        c0_n_beta,
                        a_element_op,
                        b_element_op,
                        acc_element_op,
                        c_element_op,
                        epsilon};
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
        str << "ReferenceGemmLayernorm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
