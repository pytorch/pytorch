// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
// FIXME: can it be replaced with ck::Tuple?
#include <tuple>

namespace ck {

// The templated struct reduce_binary_operator maps the enum Ids of binary operators to their
// respective functor classes.
// The boolean member "indexable" are also provided in reduce_binary_operactor for
// easier checking by the upper-layer codes in the kernels.

template <ReduceTensorOp Op>
struct reduce_binary_operator;

template <>
struct reduce_binary_operator<ReduceTensorOp::ADD>
{
    using opType = reduce::Add;

    static constexpr bool indexable = false;
};

template <>
struct reduce_binary_operator<ReduceTensorOp::MUL>
{
    using opType = reduce::Mul;

    static constexpr bool indexable = false;
};

template <>
struct reduce_binary_operator<ReduceTensorOp::MIN>
{
    using opType = reduce::Min;

    static constexpr bool indexable = true;
};

template <>
struct reduce_binary_operator<ReduceTensorOp::MAX>
{
    using opType = reduce::Max;

    static constexpr bool indexable = true;
};

template <>
struct reduce_binary_operator<ReduceTensorOp::AMAX>
{
    using opType = reduce::AMax;

    static constexpr bool indexable = true;
};

template <>
struct reduce_binary_operator<ReduceTensorOp::AVG>
{
    using opType = reduce::Add;

    static constexpr bool indexable = false;
};

template <>
struct reduce_binary_operator<ReduceTensorOp::NORM1>
{
    using opType = reduce::Add;

    static constexpr bool indexable = false;
};

template <>
struct reduce_binary_operator<ReduceTensorOp::NORM2>
{
    using opType = reduce::Add;

    static constexpr bool indexable = false;
};

// The templated struct reduce_unary_operator maps the enum Ids of Reduce operators to two unary
// functor classes.
// The two unary functors are called before and afer the Reduction is executed respectively
template <ReduceTensorOp Op, bool IsFirstReduce, bool IsLastReduce>
struct reduce_unary_operator
{
    using InElementwiseOperation  = tensor_operation::element_wise::PassThrough;
    using AccElementwiseOperation = tensor_operation::element_wise::PassThrough;

    static std::tuple<InElementwiseOperation, AccElementwiseOperation>
    GetElementwiseOperator(int32_t reduceLength)
    {
        (void)reduceLength;
        return std::make_tuple(InElementwiseOperation{}, AccElementwiseOperation{});
    };
};

template <bool IsFirstReduce>
struct reduce_unary_operator<ReduceTensorOp::AVG, IsFirstReduce, true>
{
    using InElementwiseOperation  = tensor_operation::element_wise::PassThrough;
    using AccElementwiseOperation = tensor_operation::element_wise::UnaryDivide;

    static std::tuple<InElementwiseOperation, AccElementwiseOperation>
    GetElementwiseOperator(int32_t reduceLength)
    {
        return std::make_tuple(InElementwiseOperation{}, AccElementwiseOperation{reduceLength});
    };
};

template <bool IsLastReduce>
struct reduce_unary_operator<ReduceTensorOp::NORM1, true, IsLastReduce>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnaryAbs;
    using AccElementwiseOperation = tensor_operation::element_wise::PassThrough;

    static std::tuple<InElementwiseOperation, AccElementwiseOperation>
    GetElementwiseOperator(int32_t reduceLength)
    {
        (void)reduceLength;
        return std::make_tuple(InElementwiseOperation{}, AccElementwiseOperation{});
    };
};

template <bool IsLastReduce>
struct reduce_unary_operator<ReduceTensorOp::AMAX, true, IsLastReduce>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnaryAbs;
    using AccElementwiseOperation = tensor_operation::element_wise::PassThrough;

    static std::tuple<InElementwiseOperation, AccElementwiseOperation>
    GetElementwiseOperator(int32_t reduceLength)
    {
        (void)reduceLength;
        return std::make_tuple(InElementwiseOperation{}, AccElementwiseOperation{});
    };
};

template <>
struct reduce_unary_operator<ReduceTensorOp::NORM2, true, false>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnarySquare;
    using AccElementwiseOperation = tensor_operation::element_wise::PassThrough;

    static std::tuple<InElementwiseOperation, AccElementwiseOperation>
    GetElementwiseOperator(int32_t reduceLength)
    {
        (void)reduceLength;
        return std::make_tuple(InElementwiseOperation{}, AccElementwiseOperation{});
    };
};

template <>
struct reduce_unary_operator<ReduceTensorOp::NORM2, true, true>
{
    using InElementwiseOperation  = tensor_operation::element_wise::UnarySquare;
    using AccElementwiseOperation = tensor_operation::element_wise::UnarySqrt;

    static std::tuple<InElementwiseOperation, AccElementwiseOperation>
    GetElementwiseOperator(int32_t reduceLength)
    {
        (void)reduceLength;
        return std::make_tuple(InElementwiseOperation{}, AccElementwiseOperation{});
    };
};

template <>
struct reduce_unary_operator<ReduceTensorOp::NORM2, false, true>
{
    using InElementwiseOperation  = tensor_operation::element_wise::PassThrough;
    using AccElementwiseOperation = tensor_operation::element_wise::UnarySqrt;

    static std::tuple<InElementwiseOperation, AccElementwiseOperation>
    GetElementwiseOperator(int32_t reduceLength)
    {
        (void)reduceLength;
        return std::make_tuple(InElementwiseOperation{}, AccElementwiseOperation{});
    };
};

} // namespace ck
