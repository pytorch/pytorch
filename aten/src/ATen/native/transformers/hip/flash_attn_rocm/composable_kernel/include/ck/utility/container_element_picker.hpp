// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_CONTAINER_ELEMENT_PICKER_HPP
#define CK_CONTAINER_ELEMENT_PICKER_HPP

#include "functional2.hpp"
#include "sequence.hpp"

namespace ck {

// Arr: Array or StaticallyIndexedArray
// Picks: Sequence<...>
template <typename Arr, typename Picks>
struct ContainerElementPicker
{
    using type = ContainerElementPicker;
#if 0
    using data_type = typename Arr::data_type;
#endif

    __host__ __device__ constexpr ContainerElementPicker() = delete;

    __host__ __device__ constexpr ContainerElementPicker(Arr& array) : mArray{array}
    {
        constexpr index_t imax =
            reduce_on_sequence(Picks{}, math::maximize<index_t>{}, Number<0>{});

        static_assert(imax < Arr::Size(), "wrong! exceeding # array element");
    }

    __host__ __device__ static constexpr auto Size() { return Picks::Size(); }

    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I> i) const
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[i];
        return mArray[IP];
    }

    template <index_t I>
    __host__ __device__ constexpr auto& At(Number<I> i)
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[i];
        return mArray(IP);
    }

    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

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

    private:
    Arr& mArray;
};

// Arr: Array or StaticallyIndexedArray
// Picks: Sequence<...>
template <typename Arr, typename Picks>
struct ConstantContainerElementPicker
{
    using type = ConstantContainerElementPicker;
#if 0
    using data_type = typename Arr::data_type;
#endif

    __host__ __device__ constexpr ConstantContainerElementPicker() = delete;

    __host__ __device__ constexpr ConstantContainerElementPicker(const Arr& array) : mArray{array}
    {
        constexpr index_t imax =
            reduce_on_sequence(Picks{}, math::maximize<index_t>{}, Number<0>{});

        static_assert(imax < Arr::Size(), "wrong! exceeding # array element");
    }

    __host__ __device__ static constexpr auto Size() { return Picks::Size(); }

    template <index_t I>
    __host__ __device__ constexpr const auto& At(Number<I> i) const
    {
        static_assert(I < Size(), "wrong!");

        constexpr auto IP = Picks{}[i];
        return mArray[IP];
    }

    template <index_t I>
    __host__ __device__ constexpr const auto& operator[](Number<I> i) const
    {
        return At(i);
    }

    private:
    const Arr& mArray;
};

template <typename Arr, typename Picks, typename X>
__host__ __device__ constexpr auto operator+=(ContainerElementPicker<Arr, Picks>& y, const X& x)
{
    using Y                 = ContainerElementPicker<Arr, Picks>;
    constexpr index_t nsize = Y::Size();

    static_assert(nsize == X::Size(), "wrong! size not the same");

    static_for<0, nsize, 1>{}([&](auto i) { y(i) += x[i]; });

    return y;
}

template <typename Arr, typename Picks, typename X>
__host__ __device__ constexpr auto operator-=(ContainerElementPicker<Arr, Picks>& y, const X& x)
{
    using Y                 = ContainerElementPicker<Arr, Picks>;
    constexpr index_t nsize = Y::Size();

    static_assert(nsize == X::Size(), "wrong! size not the same");

    static_for<0, nsize, 1>{}([&](auto i) { y(i) -= x[i]; });

    return y;
}

template <typename Arr, typename Picks>
__host__ __device__ constexpr auto pick_container_element(Arr& a, Picks)
{
    return ContainerElementPicker<Arr, Picks>(a);
}

template <typename Arr, typename Picks>
__host__ __device__ constexpr auto pick_container_element(const Arr& a, Picks)
{
    return ConstantContainerElementPicker<Arr, Picks>(a);
}

} // namespace ck
#endif
