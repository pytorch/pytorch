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

template <typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutType>
struct ReferenceSparseEmbedding3ForwardLayernorm : public device::BaseOperator
{
    struct Argument : public device::BaseArgument
    {
        Argument(Tensor<OutType>& output,
                 const Tensor<EmbType>& emb_a,
                 const Tensor<EmbType>& emb_b,
                 const Tensor<EmbType>& emb_c,
                 const Tensor<IndexType>& index_a,
                 const Tensor<IndexType>& index_b,
                 const Tensor<IndexType>& index_c,
                 const Tensor<GammaDataType>& gamma,
                 const Tensor<BetaDataType>& beta,
                 ck::index_t NumRows,
                 ck::index_t EmbeddingDim,
                 ck::index_t IndexLength,
                 AccDataType epsilon)
            : output_(output),
              emb_a_(emb_a),
              emb_b_(emb_b),
              emb_c_(emb_c),
              index_a_(index_a),
              index_b_(index_b),
              index_c_(index_c),
              gamma_(gamma),
              beta_(beta),
              NumRows_(NumRows),
              EmbeddingDim_(EmbeddingDim),
              IndexLength_(IndexLength),
              epsilon_(epsilon)
        {
        }
        Tensor<OutType>& output_;
        const Tensor<EmbType> emb_a_;
        const Tensor<EmbType> emb_b_;
        const Tensor<EmbType> emb_c_;
        const Tensor<IndexType> index_a_;
        const Tensor<IndexType> index_b_;
        const Tensor<IndexType> index_c_;
        const Tensor<GammaDataType> gamma_;
        const Tensor<BetaDataType> beta_;
        ck::index_t NumRows_;
        ck::index_t EmbeddingDim_;
        ck::index_t IndexLength_;
        AccDataType epsilon_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            ck::index_t D = arg.EmbeddingDim_;
            ck::index_t L = arg.IndexLength_;
            ck::index_t E = arg.NumRows_;

            Tensor<AccDataType> accumulator({L, D});

            Tensor<AccDataType> mean({L});
            Tensor<AccDataType> var({L});

            accumulator.SetZero();

            auto f_emb_per_row = [&](auto idx) {
                IndexType idx_a = arg.index_a_(idx);
                IndexType idx_b = arg.index_b_(idx);
                IndexType idx_c = arg.index_c_(idx);

                if(!((idx_a < E) && (idx_b < E) && (idx_c < E)))
                {
                    throw(std::runtime_error("wrong! out of range"));
                }

                for(auto d = 0; d < D; d++)
                {
                    auto v_a = ck::type_convert<AccDataType>(arg.emb_a_(idx_a, d));
                    auto v_b = ck::type_convert<AccDataType>(arg.emb_b_(idx_b, d));
                    auto v_c = ck::type_convert<AccDataType>(arg.emb_c_(idx_c, d));

                    accumulator(idx, d) += v_a + v_b + v_c;
                }
            };
            make_ParallelTensorFunctor(f_emb_per_row, L)(std::thread::hardware_concurrency());

            // layernorm
            for(auto idx = 0; idx < L; ++idx)
            {
                mean(idx) = 0;
                var(idx)  = 0;

                for(auto d = 0; d < D; ++d)
                {
                    auto x_val = accumulator(idx, d);
                    mean(idx) += x_val;
                    var(idx) += x_val * x_val;
                }

                mean(idx) = mean(idx) / D;
                var(idx)  = (var(idx) / D) - (mean(idx) * mean(idx));
            }

            for(auto idx = 0; idx < L; ++idx)
            {
                for(auto d = 0; d < D; ++d)
                {
                    auto x_val          = accumulator(idx, d);
                    auto y_val          = (x_val - mean(idx)) / sqrt(var(idx) + arg.epsilon_);
                    y_val               = (y_val * arg.gamma_(d)) + arg.beta_(d);
                    arg.output_(idx, d) = ck::type_convert<OutType>(y_val);
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

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(Tensor<OutType>& output,
                             const Tensor<EmbType>& emb_a,
                             const Tensor<EmbType>& emb_b,
                             const Tensor<EmbType>& emb_c,
                             const Tensor<IndexType>& index_a,
                             const Tensor<IndexType>& index_b,
                             const Tensor<IndexType>& index_c,
                             const Tensor<GammaDataType>& gamma,
                             const Tensor<BetaDataType>& beta,
                             ck::index_t NumRows,
                             ck::index_t EmbeddingDim,
                             ck::index_t IndexLength,
                             AccDataType epsilon)
    {
        return Argument(output,
                        emb_a,
                        emb_b,
                        emb_c,
                        index_a,
                        index_b,
                        index_c,
                        gamma,
                        beta,
                        NumRows,
                        EmbeddingDim,
                        IndexLength,
                        epsilon);
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
        str << "ReferenceSparseEmbedding3ForwardLayernorm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
