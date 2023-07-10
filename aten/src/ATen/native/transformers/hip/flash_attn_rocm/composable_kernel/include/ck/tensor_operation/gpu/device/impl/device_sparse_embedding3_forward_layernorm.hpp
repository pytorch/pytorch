// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_sparse_embedding3_forward_layernorm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutType,
          ck::index_t BlockSize,
          ck::index_t DimClusterSize,
          ck::index_t RowClusterSize,
          ck::index_t DimPerBlock,
          ck::index_t RowPerBlock,
          ck::index_t DimThreadSize,
          ck::index_t RowVectorSize>
struct DeviceSparseEmbedding3ForwardLayernorm : public BaseOperator
{

    static auto MakeOutputDescriptor(const index_t index_length, const index_t rows)
    {
        return make_naive_tensor_descriptor_packed(make_tuple(index_length, rows));
    }

    struct Argument : public BaseArgument
    {
        Argument(OutType* p_out,
                 const EmbType* p_emb_a,
                 const EmbType* p_emb_b,
                 const EmbType* p_emb_c,
                 const IndexType* p_index_a,
                 const IndexType* p_index_b,
                 const IndexType* p_index_c,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 const ck::index_t NumRows,
                 const ck::index_t EmbeddingDim,
                 const ck::index_t IndexLength,
                 const AccDataType epsilon)
            : p_out_(p_out),
              p_emb_a_(p_emb_a),
              p_emb_b_(p_emb_b),
              p_emb_c_(p_emb_c),
              p_index_a_(p_index_a),
              p_index_b_(p_index_b),
              p_index_c_(p_index_c),
              p_gamma_(p_gamma),
              p_beta_(p_beta),
              NumRows_(NumRows),
              EmbeddingDim_(EmbeddingDim),
              IndexLength_(IndexLength),
              epsilon_(epsilon)
        {
            grid_size_ = (IndexLength + DimClusterSize - 1) / DimClusterSize;
        }

        OutType* p_out_;
        const EmbType* p_emb_a_;
        const EmbType* p_emb_b_;
        const EmbType* p_emb_c_;
        const IndexType* p_index_a_;
        const IndexType* p_index_b_;
        const IndexType* p_index_c_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        ck::index_t NumRows_;
        ck::index_t EmbeddingDim_;
        ck::index_t IndexLength_;
        AccDataType epsilon_;

        size_t grid_size_;
    };

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(void* p_out,
                                                              const void* p_emb_a,
                                                              const void* p_emb_b,
                                                              const void* p_emb_c,
                                                              const void* p_index_a,
                                                              const void* p_index_b,
                                                              const void* p_index_c,
                                                              const void* p_gamma,
                                                              const void* p_beta,
                                                              ck::index_t NumRows,
                                                              ck::index_t EmbeddingDim,
                                                              ck::index_t IndexLength,
                                                              const AccDataType epsilon)
    {
        return std::make_unique<Argument>(reinterpret_cast<OutType*>(p_out),
                                          reinterpret_cast<const EmbType*>(p_emb_a),
                                          reinterpret_cast<const EmbType*>(p_emb_b),
                                          reinterpret_cast<const EmbType*>(p_emb_c),
                                          reinterpret_cast<const IndexType*>(p_index_a),
                                          reinterpret_cast<const IndexType*>(p_index_b),
                                          reinterpret_cast<const IndexType*>(p_index_c),
                                          reinterpret_cast<const GammaDataType*>(p_gamma),
                                          reinterpret_cast<const BetaDataType*>(p_beta),
                                          NumRows,
                                          EmbeddingDim,
                                          IndexLength,
                                          epsilon);
    }

    using GridwiseSparseEmbedding =
        GridwiseSparseEmbedding3ForwardLayernorm<EmbType,
                                                 IndexType,
                                                 GammaDataType,
                                                 BetaDataType,
                                                 AccDataType,
                                                 OutType,
                                                 decltype(MakeOutputDescriptor(1, 1)),
                                                 BlockSize,
                                                 DimClusterSize,
                                                 RowClusterSize,
                                                 DimPerBlock,
                                                 RowPerBlock,
                                                 DimThreadSize,
                                                 RowVectorSize>;

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            auto out_desc = MakeOutputDescriptor(arg.IndexLength_, arg.EmbeddingDim_);
            const auto kernel_main =
                kernel_sparse_embedding3_forward_layernorm<GridwiseSparseEmbedding,
                                                           EmbType,
                                                           IndexType,
                                                           GammaDataType,
                                                           BetaDataType,
                                                           AccDataType,
                                                           OutType,
                                                           decltype(out_desc)>;
            float avg_time = 0;
            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.grid_size_),
                                               dim3(BlockSize),
                                               0,
                                               arg.p_out_,
                                               arg.p_emb_a_,
                                               arg.p_emb_b_,
                                               arg.p_emb_c_,
                                               arg.p_index_a_,
                                               arg.p_index_b_,
                                               arg.p_index_c_,
                                               arg.p_gamma_,
                                               arg.p_beta_,
                                               out_desc,
                                               arg.epsilon_);

            return (avg_time);
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    static bool IsSupportedArgument(const Argument* p_arg)
    {
        return (RowPerBlock == p_arg->EmbeddingDim_) && (p_arg->NumRows_ % DimPerBlock == 0);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(dynamic_cast<const Argument*>(p_arg));
    }

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceSparseEmbedding3ForwardLayernorm_"<< BlockSize << "_" <<
            DimClusterSize << "x" << RowClusterSize << "_" <<
            DimPerBlock << "x" << RowPerBlock << "_" <<
            DimThreadSize << "x" << RowVectorSize;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
