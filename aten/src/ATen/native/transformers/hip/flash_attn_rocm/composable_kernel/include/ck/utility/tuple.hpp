// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/integral_constant.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/enable_if.hpp"

namespace ck {

namespace detail {

template <index_t>
struct TupleElementKey
{
    __host__ __device__ constexpr TupleElementKey() = default;
};

template <typename Key, typename Data>
struct TupleElementKeyData
{
    using DataType = Data;

#if 0 // workaround compiler complaint about implicitly-deleted default constructor
    __host__ __device__ constexpr TupleElementKeyData() = default;
#else
    __host__ __device__ constexpr TupleElementKeyData() : mData{} {}
#endif

    template <typename T,
              typename enable_if<!is_same<remove_cvref_t<T>, TupleElementKeyData>::value,
                                 bool>::type = false>
    __host__ __device__ constexpr TupleElementKeyData(T&& v) : mData(std::forward<T>(v))
    {
    }

    DataType mData;
};

// for read access of tuple element
template <typename Key, typename Data>
__host__ __device__ constexpr const Data&
get_tuple_element_data_reference(const TupleElementKeyData<Key, Data>& x)
{
    return static_cast<const Data&>(x.mData);
}

// for write access of tuple element
template <typename Key, typename Data>
__host__ __device__ constexpr Data&
get_tuple_element_data_reference(TupleElementKeyData<Key, Data>& x)
{
    return x.mData;
}

// TODO: not sure the use of reference is correct
template <typename Key, typename Data>
__host__ __device__ constexpr Data&&
get_tuple_element_data_reference(TupleElementKeyData<Key, Data>&& x)
{
    return static_cast<Data&&>(x.mData);
}

// for infering type of tuple element
template <typename Key, typename Data>
__host__ __device__ constexpr Data get_tuple_element_data(const TupleElementKeyData<Key, Data>& x)
{
    return std::forward(x.mData);
}

template <typename Indices, typename... Xs>
struct TupleImpl;

template <index_t... Is, typename... Xs>
struct TupleImpl<Sequence<Is...>, Xs...> : TupleElementKeyData<TupleElementKey<Is>, Xs>...
{
    __host__ __device__ constexpr TupleImpl() = default;

    template <typename Y,
              typename enable_if<sizeof...(Is) == 1 && sizeof...(Xs) == 1 &&
                                     !is_same<remove_cvref_t<Y>, TupleImpl>::value,
                                 bool>::type = false>
    __host__ __device__ constexpr TupleImpl(Y&& y)
        : TupleElementKeyData<TupleElementKey<Is>, Xs>(std::forward<Y>(y))...
    {
    }

    template <typename... Ys, typename enable_if<sizeof...(Ys) >= 2, bool>::type = false>
    __host__ __device__ constexpr TupleImpl(Ys&&... ys)
        : TupleElementKeyData<TupleElementKey<Is>, Xs>(std::forward<Ys>(ys))...
    {
        static_assert(sizeof...(Is) == sizeof...(Xs) && sizeof...(Is) == sizeof...(Ys),
                      "wrong! inconsistent size");
    }

    __host__ __device__ static constexpr index_t Size() { return sizeof...(Xs); }

    template <index_t I>
    __host__ __device__ constexpr const auto& GetElementDataByKey(TupleElementKey<I>) const
    {
        return get_tuple_element_data_reference<TupleElementKey<I>>(*this);
    }

    template <index_t I>
    __host__ __device__ constexpr auto& GetElementDataByKey(TupleElementKey<I>)
    {
        return get_tuple_element_data_reference<TupleElementKey<I>>(*this);
    }
};

} // namespace detail

template <typename... Xs>
struct Tuple : detail::TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>
{
    using base =
        detail::TupleImpl<typename arithmetic_sequence_gen<0, sizeof...(Xs), 1>::type, Xs...>;

    __host__ __device__ constexpr Tuple() = default;

    template <typename Y,
              typename enable_if<sizeof...(Xs) == 1 && !is_same<remove_cvref_t<Y>, Tuple>::value,
                                 bool>::type = false>
    __host__ __device__ constexpr Tuple(Y&& y) : base(std::forward<Y>(y))
    {
    }

    template <typename... Ys,
              typename enable_if<sizeof...(Ys) == sizeof...(Xs) && sizeof...(Ys) >= 2, bool>::type =
                  false>
    __host__ __device__ constexpr Tuple(Ys&&... ys) : base(std::forward<Ys>(ys)...)
    {
    }

    __host__ __device__ static constexpr index_t Size() { return sizeof...(Xs); }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I>) const
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return base::GetElementDataByKey(detail::TupleElementKey<I>{});
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I>)
    {
        static_assert(I < base::Size(), "wrong! out of range");
        return base::GetElementDataByKey(detail::TupleElementKey<I>{});
    }

    // read access
    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    // write access
    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
    {
        return At(i);
    }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }
};

template <>
struct Tuple<>
{
    __host__ __device__ constexpr Tuple() = default;

    __host__ __device__ static constexpr index_t Size() { return 0; }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T&)
    {
        return *this;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }
};

template <index_t I, typename TTuple>
struct tuple_element
{
    // type should keep the cv/ref qualifier of original tuple element
    using type = decltype(detail::get_tuple_element_data<detail::TupleElementKey<I>>(TTuple{}));
};

template <index_t I, typename TTuple>
using tuple_element_t = typename tuple_element<I, TTuple>::type;

template <typename... Xs>
__host__ __device__ constexpr auto make_tuple(Xs&&... xs)
{
    return Tuple<remove_cvref_t<Xs>...>(std::forward<Xs>(xs)...);
}

// https://en.cppreference.com/w/cpp/utility/tuple/tie
template <typename... Args>
constexpr Tuple<Args&...> tie(Args&... args) noexcept
{
    return {args...};
}

} // namespace ck
