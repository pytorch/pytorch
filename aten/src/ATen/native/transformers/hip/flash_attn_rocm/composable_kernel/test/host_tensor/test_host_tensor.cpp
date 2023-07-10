// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/library/utility/host_tensor.hpp"

using namespace ck;

TEST(HostTensorTranspose, TestBadArugment)
{
    Tensor<float> tensor({13, 7});

    EXPECT_THROW(tensor.Transpose({0}), std::runtime_error);
    EXPECT_THROW(tensor.Transpose({0, 1, 2}), std::runtime_error);
}

TEST(HostTensorTranspose, Test2D)
{
    std::vector<size_t> lengths  = {13, 7};
    std::vector<size_t> tlengths = {7, 13};

    Tensor<float> tensor(lengths);
    tensor(0, 0) = 0.f;
    tensor(3, 4) = 34.f;

    EXPECT_EQ(tensor.GetLengths(), lengths);
    EXPECT_EQ(tensor(0, 0), 0.f);
    EXPECT_EQ(tensor(3, 4), 34.f);
    EXPECT_EQ(tensor(4, 3), 0.f);

    EXPECT_EQ(tensor.Transpose().GetLengths(), tlengths);
    EXPECT_EQ(tensor.Transpose()(0, 0), 0.f);
    EXPECT_EQ(tensor.Transpose()(4, 3), 34.f);
    EXPECT_EQ(tensor.Transpose()(3, 4), 0.f);
}

TEST(HostTensorTranspose, Test3D)
{
    std::vector<size_t> lengths  = {13, 7, 5};
    std::vector<size_t> tlengths = {5, 7, 13};

    Tensor<float> tensor(lengths);
    tensor(0, 0, 0) = 0.f;
    tensor(3, 4, 2) = 342.f;

    EXPECT_EQ(tensor.GetLengths(), lengths);
    EXPECT_EQ(tensor(0, 0, 0), 0.f);
    EXPECT_EQ(tensor(3, 4, 2), 342.f);
    EXPECT_EQ(tensor(4, 3, 2), 0.f);

    EXPECT_EQ(tensor.Transpose().GetLengths(), tlengths);
    EXPECT_EQ(tensor.Transpose()(0, 0, 0), 0.f);
    EXPECT_EQ(tensor.Transpose()(2, 4, 3), 342.f);
    EXPECT_EQ(tensor.Transpose()(2, 3, 4), 0.f);
}

TEST(HostTensorTranspose, Test3D_021)
{
    std::vector<size_t> lengths  = {13, 7, 5};
    std::vector<size_t> tlengths = {13, 5, 7};

    Tensor<float> tensor(lengths);
    tensor(0, 0, 0) = 0.f;
    tensor(3, 4, 2) = 342.f;

    EXPECT_EQ(tensor.GetLengths(), lengths);
    EXPECT_EQ(tensor(0, 0, 0), 0.f);
    EXPECT_EQ(tensor(3, 4, 2), 342.f);
    EXPECT_EQ(tensor(4, 3, 2), 0.f);

    // transpose last two dimensions
    EXPECT_EQ(tensor.Transpose({0, 2, 1}).GetLengths(), tlengths);
    EXPECT_EQ(tensor.Transpose({0, 2, 1})(0, 0, 0), 0.f);
    EXPECT_EQ(tensor.Transpose({0, 2, 1})(2, 4, 3), 0.f);
    EXPECT_EQ(tensor.Transpose({0, 2, 1})(3, 2, 4), 342.f);
    EXPECT_EQ(tensor.Transpose({0, 2, 1})(2, 3, 4), 0.f);

    // transpose last two dimensions back again
    EXPECT_EQ(tensor.Transpose({0, 2, 1}).Transpose({0, 2, 1}).GetLengths(), lengths);
    EXPECT_EQ(tensor.Transpose({0, 2, 1}).Transpose({0, 2, 1})(3, 4, 2), 342.f);
}

TEST(HostTensorTranspose, TestNonpacked2D)
{
    std::vector<size_t> lengths  = {13, 7};
    std::vector<size_t> strides  = {100, 1};
    std::vector<size_t> tlengths = {7, 13};

    Tensor<float> tensor(lengths, strides);
    tensor(0, 0) = 0.f;
    tensor(3, 4) = 34.f;

    EXPECT_EQ(tensor.GetLengths(), lengths);
    EXPECT_EQ(tensor(0, 0), 0.f);
    EXPECT_EQ(tensor(3, 4), 34.f);
    EXPECT_EQ(tensor(4, 3), 0.f);

    EXPECT_EQ(tensor.Transpose().GetLengths(), tlengths);
    EXPECT_EQ(tensor.Transpose()(0, 0), 0.f);
    EXPECT_EQ(tensor.Transpose()(4, 3), 34.f);
    EXPECT_EQ(tensor.Transpose()(3, 4), 0.f);
}
