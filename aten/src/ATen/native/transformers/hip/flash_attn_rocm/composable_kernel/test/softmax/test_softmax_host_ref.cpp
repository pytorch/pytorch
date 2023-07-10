// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

#include "gtest/gtest.h"

using namespace ck;

TEST(ReferenceSoftmax, Run)
{
    Tensor<float> x({2, 2});
    Tensor<float> y({2, 2});
    x.GenerateTensorValue(GeneratorTensor_Diagonal<float>{});

    using ReferenceSoftmax = tensor_operation::host::ReferenceSoftmax<float, float, float>;

    float alpha = 1.f;
    float beta  = 0.f;

    auto ref_softmax         = ReferenceSoftmax{};
    auto ref_softmax_invoker = ref_softmax.MakeInvoker();

    auto ref_softmax_argument = ref_softmax.MakeArgument(x, y, alpha, beta, {1});
    ref_softmax_invoker.Run(ref_softmax_argument);
    EXPECT_TRUE((utils::check_err(
        y.mData, std::vector<float>{0.73105858f, 0.268941421f, 0.26894142f, 0.73105858f})));
}

TEST(ReferenceSoftmax, RunWithCalculatedStats)
{
    // >>> x = np.eye(4)
    // >>> m = np.max(np.exp(x), axis=1, keepdims=True)
    // >>> l = np.sum(np.exp(x - np.tile(m, (1,4))), axis=1, keepdims=True)
    // >>> m + np.log(l)
    // array([[1.74366838],
    //        [1.74366838],
    //        [1.74366838],
    //        [1.74366838]])
    Tensor<float> x({4, 4});
    Tensor<float> y({4, 4});
    Tensor<float> stats({4});
    x.GenerateTensorValue(GeneratorTensor_Diagonal<float>{});

    using ReferenceSoftmax = tensor_operation::host::ReferenceSoftmax<float, float, float>;

    float alpha = 1.f;
    float beta  = 0.f;

    auto ref_softmax         = ReferenceSoftmax{};
    auto ref_softmax_invoker = ref_softmax.MakeInvoker();

    {
        auto ref_softmax_argument = ref_softmax.MakeArgument(x, y, alpha, beta, {1}, &stats);
        ref_softmax_invoker.Run(ref_softmax_argument);
        EXPECT_TRUE((utils::check_err(
            stats.mData, std::vector<float>{1.74366838f, 1.74366838f, 1.74366838f, 1.74366838f})));
    }

    {
        Tensor<float> yy({4, 4});
        auto ref_softmax_argument = ref_softmax.MakeArgument(x, yy, alpha, beta, {1}, &stats);
        ref_softmax_invoker.RunWithPreCalcStats(ref_softmax_argument);
        EXPECT_TRUE((utils::check_err(y.mData, yy.mData)));
    }
}
