// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "include/ck/utility/data_type.hpp"
#include "profiler/profile_softmax_impl.hpp"

namespace ck {

template <typename Range>
std::string serialize_range(const Range& range)
{
    std::stringstream ss;
    for(auto& r : range)
    {
        ss << r << ", ";
    }
    std::string str = ss.str();
    return std::string(str.begin(), str.end() - 2);
}

template <typename Tuple>
class TestSoftmax : public ::testing::Test
{
    protected:
    using InDataType              = std::tuple_element_t<0, Tuple>;
    using AccDataType             = std::tuple_element_t<1, Tuple>;
    using OutDataType             = std::tuple_element_t<2, Tuple>;
    static constexpr index_t Rank = std::tuple_element_t<3, Tuple>{}.value;

    public:
    std::vector<std::vector<index_t>> in_lengths_ = {{2, 128, 1024}, {4, 16, 8448}, {128, 128, 64}};
    std::vector<std::vector<AccDataType>> scales_ = {{2, 0}, {0, 2}, {2, 2}};
    bool bench_                                   = false; // measure kernel performance
    bool verify_                                  = true;

    void SetUp() override
    {
        if constexpr(Rank == 4)
        {
            in_lengths_ = std::vector<std::vector<index_t>>{
                {1, 2, 128, 1024}, {2, 4, 16, 8448}, {1, 128, 128, 64}};
        }
    }

    void RunSingle(std::vector<index_t> in_length,
                   std::vector<index_t> reduce_dims,
                   AccDataType alpha,
                   AccDataType beta)
    {
        int init_method = 1; // integer value initialization
        bool log        = false;
        std::vector<ck::index_t> strides; // intenionally empty, to get packed layout.
        bool pass = ck::profiler::profile_softmax_impl<InDataType, AccDataType, OutDataType, Rank>(
            verify_, init_method, log, bench_, in_length, strides, reduce_dims, alpha, beta);
        EXPECT_TRUE(pass);
    }

    void Run(std::vector<index_t> reduce_dims = {})
    {
        if(reduce_dims.empty())
        {
            reduce_dims.push_back(Rank - 1);
        }

        for(auto in_length : this->in_lengths_)
        {
            for(auto scale : this->scales_)
            {
                this->RunSingle(in_length, reduce_dims, scale[0], scale[1]);
            }
        }
    }
};

template <index_t Rank,
          index_t NumReduceDim,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InSrcVectorDim,
          index_t InSrcVectorSize,
          index_t OutDstVectorSize>
struct DeviceSoftmaxInstanceWrapper
{
    using F16  = half_t;
    using F32  = float;
    using Pass = tensor_operation::element_wise::PassThrough;

    using InDataType   = F16;
    using AccDataType  = F32;
    using OutDataType  = F16;
    using InElementOp  = Pass;
    using AccElementOp = Pass;

    using DeviceSoftmaxInstance = tensor_operation::device::DeviceSoftmaxImpl<InDataType,
                                                                              AccDataType,
                                                                              OutDataType,
                                                                              InElementOp,
                                                                              AccElementOp,
                                                                              Rank,
                                                                              NumReduceDim,
                                                                              BlockSize,
                                                                              MThreadClusterSize,
                                                                              KThreadClusterSize,
                                                                              MThreadSliceSize,
                                                                              KThreadSliceSize,
                                                                              InSrcVectorDim,
                                                                              InSrcVectorSize,
                                                                              OutDstVectorSize>;

    bool IsSupported(const std::vector<index_t> in_lengths,
                     const std::vector<index_t> in_strides,
                     const std::vector<index_t> reduce_dims) const
    {
        auto softmax  = DeviceSoftmaxInstance{};
        auto argument = softmax.MakeArgument(in_lengths,
                                             in_strides,
                                             reduce_dims,
                                             1,       // alpha
                                             1,       // beta
                                             nullptr, // in_dev
                                             nullptr, // in_out
                                             Pass{},  // in elementwise op
                                             Pass{}); // acc elementwise op
        return softmax.IsSupportedArgument(argument);
    }
};

} // namespace ck
