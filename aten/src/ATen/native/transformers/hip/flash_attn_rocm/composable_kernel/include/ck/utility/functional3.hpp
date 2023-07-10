// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/functional.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/multi_index.hpp"

namespace ck {

namespace detail {

// RemainLengths: Sequence<...>
// Orders: Sequence<...>
template <class RemainLengths, class Orders>
struct static_ford_impl
{
    __host__ __device__ constexpr static_ford_impl()
    {
        static_assert(RemainLengths::GetSize() > 0, "wrong! should not get here");
    }

    // F signature: F(Sequence<...>)
    // CurrentOrderedId: Sequence<...>
    template <class F, class CurrentOrderedId>
    __host__ __device__ constexpr void operator()(F f, CurrentOrderedId) const
    {
        static_for<0, RemainLengths::Front(), 1>{}([=](auto I) {
            static_ford_impl<decltype(RemainLengths::PopFront()), Orders>{}(
                f, CurrentOrderedId::PushBack(I));
        });
    }
};

template <class Orders>
struct static_ford_impl<Sequence<>, Orders>
{
    // F signature: F(Sequence<...>)
    // OrderedId: Sequence<...>
    template <class F, class OrderedId>
    __host__ __device__ constexpr void operator()(F f, OrderedId) const
    {
        // retrive unordered Id
        f(OrderedId::ReorderGivenOld2New(Orders{}));
    }
};

// RemainLengths: Sequence<...>
// Orders: Sequence<...>
template <class RemainLengths, class Orders>
struct ford_impl
{
    __host__ __device__ constexpr ford_impl()
    {
        static_assert(RemainLengths::GetSize() > 0, "wrong! should not get here");
    }

    // F signature: F(Array<...> multi_id)
    // CurrentOrderdId: Array<...>
    template <class F, class CurrentOrderedId>
    __host__ __device__ constexpr void operator()(F f, CurrentOrderedId current_ordered_id) const
    {
        for(index_t i = 0; i < RemainLengths::Front(); ++i)
        {
            ford_impl<decltype(RemainLengths::PopFront()), Orders>{}(
                f, container_push_back(current_ordered_id, i));
        }
    }
};

template <class Orders>
struct ford_impl<Sequence<>, Orders>
{
    // F signature: F(Array<...> multi_id)
    // CurrentOrderdId: Array<...>
    template <class F, class CurrentOrderedId>
    __host__ __device__ constexpr void operator()(F f, CurrentOrderedId current_ordered_id) const
    {
        // retrive unordered Id
        f(container_reorder_given_old2new(current_ordered_id, Orders{}));
    }
};

} // namespace detail

// Lengths is Sequence<...>, it is the length of each dimension for
// N-dimensional loop
// Orders is Sequence<...>, it is the order of dimension in which static_ford
// will loop over each
// dimension
template <class Lengths,
          class Orders = typename arithmetic_sequence_gen<0, Lengths::GetSize(), 1>::type>
struct static_ford
{
    __host__ __device__ constexpr static_ford()
    {
        static_assert(Lengths::GetSize() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::GetSize() == Orders::GetSize(), "wrong! inconsistent size");
    }

    // F signature: F(Sequence<...> multi_id)
    // multi_id is the unordered multi-index
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        constexpr auto ordered_lengths = Lengths::ReorderGivenNew2Old(Orders{});
        detail::static_ford_impl<decltype(ordered_lengths), Orders>{}(f, Sequence<>{});
    }
};

// Lengths is Sequence<...>, it is the length of each dimension for
// N-dimensional loop
// Orders is Sequence<...>, it is the order of dimension in which ford will loop
// over each
// dimension
template <class Lengths,
          class Orders = typename arithmetic_sequence_gen<0, Lengths::GetSize(), 1>::type>
struct ford
{
    __host__ __device__ constexpr ford()
    {
        static_assert(Lengths::GetSize() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::GetSize() == Orders::GetSize(), "wrong! inconsistent size");
    }

    // F signature: F(Array<...> multi_id)
    // multi_id is the unordered multi-index
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        constexpr auto ordered_lengths = Lengths::ReorderGivenNew2Old(Orders{});

        for(index_t i = 0; i < ordered_lengths.Front(); ++i)
        {
            detail::ford_impl<decltype(ordered_lengths.PopFront()), Orders>{}(f,
                                                                              make_multi_index(i));
        }
    }
};

} // namespace ck
