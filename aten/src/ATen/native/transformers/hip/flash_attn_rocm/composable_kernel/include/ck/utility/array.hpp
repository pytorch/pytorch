// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_ARRAY_HPP
#define CK_ARRAY_HPP

#include "functional2.hpp"
#include "sequence.hpp"

namespace ck {

template <typename TData, index_t NSize>
struct Array
{
    using type      = Array;
    using data_type = TData;

    TData mData[NSize];

    __host__ __device__ static constexpr index_t Size() { return NSize; }

    __host__ __device__ constexpr const TData& At(index_t i) const { return mData[i]; }

    __host__ __device__ constexpr TData& At(index_t i) { return mData[i]; }

    __host__ __device__ constexpr const TData& operator[](index_t i) const { return At(i); }

    __host__ __device__ constexpr TData& operator()(index_t i) { return At(i); }

    template <typename T>
    __host__ __device__ constexpr auto operator=(const T& a)
    {
        static_assert(T::Size() == Size(), "wrong! size not the same");

        static_for<0, Size(), 1>{}([&](auto i) { operator()(i) = a[i]; });

        return *this;
    }
};

// empty Array
template <typename TData>
struct Array<TData, 0>
{
    using type      = Array;
    using data_type = TData;

    __host__ __device__ static constexpr index_t Size() { return 0; }
};

template <typename X, typename... Xs>
__host__ __device__ constexpr auto make_array(X&& x, Xs&&... xs)
{
    using data_type = remove_cvref_t<X>;
    return Array<data_type, sizeof...(Xs) + 1>{std::forward<X>(x), std::forward<Xs>(xs)...};
}

// make empty array
template <typename X>
__host__ __device__ constexpr auto make_array()
{
    return Array<X, 0>{};
}

} // namespace ck
#endif
