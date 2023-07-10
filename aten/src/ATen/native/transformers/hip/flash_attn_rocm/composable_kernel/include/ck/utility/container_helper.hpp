// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_CONTAINER_HELPER_HPP
#define CK_CONTAINER_HELPER_HPP

#include "sequence.hpp"
#include "sequence_helper.hpp"
#include "array.hpp"
#include "tuple.hpp"
#include "tuple_helper.hpp"
#include "statically_indexed_array.hpp"
#include "container_element_picker.hpp"

namespace ck {

template <typename TData, index_t NSize>
__host__ __device__ constexpr auto container_push_back(const Array<TData, NSize>& a, const TData& x)
{
    Array<TData, NSize + 1> r;

    static_for<0, NSize, 1>{}([&r, &a ](auto i) constexpr { r(i) = a[i]; });

    r(Number<NSize>{}) = x;

    return r;
}

template <typename... Ts, typename T>
__host__ __device__ constexpr auto container_push_front(const Tuple<Ts...>& a, const T& x)
{
    return container_concat(make_tuple(x), a);
}

template <typename... Ts, typename T>
__host__ __device__ constexpr auto container_push_back(const Tuple<Ts...>& a, const T& x)
{
    return container_concat(a, make_tuple(x));
}

template <typename TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto
container_reorder_given_new2old(const Array<TData, NSize>& old_array, Sequence<IRs...> /*new2old*/)
{
    static_assert(NSize == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return make_array(old_array[Number<IRs>{}]...);
}

template <typename TData, index_t NSize, index_t... IRs>
__host__ __device__ constexpr auto
container_reorder_given_old2new(const Array<TData, NSize>& old_array, Sequence<IRs...> old2new)
{
    return container_reorder_given_new2old(
        old_array, typename sequence_map_inverse<decltype(old2new)>::type{});
}

template <typename... Ts, index_t... IRs>
__host__ __device__ constexpr auto container_reorder_given_new2old(const Tuple<Ts...>& old_tuple,
                                                                   Sequence<IRs...> /*new2old*/)
{
    static_assert(sizeof...(Ts) == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return make_tuple(old_tuple[Number<IRs>{}]...);
}

template <typename... Ts, index_t... IRs>
__host__ __device__ constexpr auto container_reorder_given_old2new(const Tuple<Ts...>& old_tuple,
                                                                   Sequence<IRs...> old2new)
{
    return container_reorder_given_new2old(
        old_tuple, typename sequence_map_inverse<decltype(old2new)>::type{});
}

template <index_t... Is, index_t... IRs>
__host__ __device__ constexpr auto container_reorder_given_new2old(Sequence<Is...> /* old_seq */,
                                                                   Sequence<IRs...> /*new2old*/)
{
    static_assert(sizeof...(Is) == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    return Sequence<Sequence<Is...>::At(Number<IRs>{})...>{};
}

template <index_t... Is, index_t... IRs>
__host__ __device__ constexpr auto container_reorder_given_old2new(Sequence<Is...> old_seq,
                                                                   Sequence<IRs...> /* old2new */)
{
    static_assert(sizeof...(Is) == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<Sequence<IRs...>>{}, "wrong! invalid reorder map");

    constexpr auto new2old = typename sequence_map_inverse<Sequence<IRs...>>::type{};

    return container_reorder_given_new2old(old_seq, new2old);
}

#if !CK_WORKAROUND_SWDEV_275126
// rocm-4.1 compiler would crash for recursive lambda
template <typename Container,
          typename Reduce,
          typename Init,
          index_t IBegin = 0,
          index_t IEnd   = Container::Size(),
          index_t IStep  = 1>
__host__ __device__ constexpr auto container_reduce(const Container& x,
                                                    Reduce reduce,
                                                    Init init,
                                                    Number<IBegin> = Number<0>{},
                                                    Number<IEnd>   = Number<Container::Size()>{},
                                                    Number<IStep>  = Number<1>{})
{
    static_assert((IEnd - IBegin) % IStep == 0, "wrong!");

    // f is recursive function, fs is a dummy of f
    // i is index, y_old is current scan, r_old is current reduction
    auto f = [&](auto fs, auto i, auto r_old) {
        auto r_new = reduce(x[i], r_old);

        if constexpr(i.value < IEnd - IStep)
        {
            // recursively call f/fs
            return fs(fs, i + Number<IStep>{}, r_new);
        }
        else
        {
            return r_new;
        }
    };

    // start recursion
    return f(f, Number<IBegin>{}, init);
}
#else
// i is index, y_old is current scan, r_old is current reduction
template <typename Container,
          typename Reduce,
          typename ROld,
          index_t I,
          index_t IEnd,
          index_t IStep>
__host__ __device__ constexpr auto container_reduce_impl(
    const Container& x, Reduce reduce, ROld r_old, Number<I> i, Number<IEnd>, Number<IStep>)
{
    auto r_new = reduce(x[i], r_old);

    if constexpr(i.value < IEnd - IStep)
    {
        return container_reduce_impl(
            x, reduce, r_new, i + Number<IStep>{}, Number<IEnd>{}, Number<IStep>{});
    }
    else
    {
        return r_new;
    }
}

// rocm-4.1 compiler would crash for recursive lambda
// container reduce with initial value
template <typename Container,
          typename Reduce,
          typename Init,
          index_t IBegin = 0,
          index_t IEnd   = Container::Size(),
          index_t IStep  = 1>
__host__ __device__ constexpr auto container_reduce(const Container& x,
                                                    Reduce reduce,
                                                    Init init,
                                                    Number<IBegin> = Number<0>{},
                                                    Number<IEnd>   = Number<Container::Size()>{},
                                                    Number<IStep>  = Number<1>{})
{
    static_assert((IEnd - IBegin) % IStep == 0, "wrong!");

    if constexpr(IEnd > IBegin)
    {
        return container_reduce_impl(
            x, reduce, init, Number<IBegin>{}, Number<IEnd>{}, Number<IStep>{});
    }
    else
    {
        return init;
    }
}
#endif

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
container_reverse_inclusive_scan(const Array<TData, NSize>& x, Reduce f, TData init)
{
    Array<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        r    = f(r, x[i]);
        y(i) = r;
    });

    r              = f(r, x[Number<0>{}]);
    y(Number<0>{}) = r;

    return y;
}

template <typename TData, index_t NSize, typename Reduce>
__host__ __device__ constexpr auto
container_reverse_exclusive_scan(const Array<TData, NSize>& x, Reduce f, TData init)
{
    Array<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        y(i) = r;
        r    = f(r, x[i]);
    });

    y(Number<0>{}) = r;

    return y;
}

template <index_t... Is, typename Reduce, index_t Init>
__host__ __device__ constexpr auto
container_reverse_exclusive_scan(const Sequence<Is...>& seq, Reduce f, Number<Init>)
{
    return reverse_exclusive_scan_sequence(seq, f, Number<Init>{});
}

#if !CK_WORKAROUND_SWDEV_275126
// rocm4.1 compiler would crash with recursive lambda
template <typename... Xs, typename Reduce, typename Init>
__host__ __device__ constexpr auto
container_reverse_exclusive_scan(const Tuple<Xs...>& x, Reduce reduce, Init init)
{
    constexpr index_t NSize = sizeof...(Xs);

    // f is recursive function, fs is a dummy of f
    // i is index, y_old is current scan, r_old is current reduction
    auto f = [&](auto fs, auto i, auto y_old, auto r_old) {
        auto r_new = reduce(x[i], r_old);

        auto y_new = container_push_front(y_old, r_new);

        if constexpr(i.value > 1)
        {
            // recursively call f/fs
            return fs(fs, i - Number<1>{}, y_new, r_new);
        }
        else
        {
            return y_new;
        }
    };

    // start recursion
    return f(f, Number<NSize - 1>{}, make_tuple(init), init);
}
#else
// i is index, y_old is current scan, r_old is current reduction
template <typename... Xs, typename Reduce, index_t I, typename YOld, typename ROld>
__host__ __device__ constexpr auto container_reverse_exclusive_scan_impl(
    const Tuple<Xs...>& x, Reduce reduce, Number<I> i, YOld y_old, ROld r_old)
{
    auto r_new = reduce(x[i], r_old);

    auto y_new = container_push_front(y_old, r_new);

    if constexpr(i.value > 1)
    {
        // recursively call f/fs
        return container_reverse_exclusive_scan_impl(x, reduce, i - Number<1>{}, y_new, r_new);
    }
    else
    {
        return y_new;
    }
}

template <typename... Xs, typename Reduce, typename Init>
__host__ __device__ constexpr auto
container_reverse_exclusive_scan(const Tuple<Xs...>& x, Reduce reduce, Init init)
{
    constexpr index_t NSize = sizeof...(Xs);

    return container_reverse_exclusive_scan_impl(
        x, reduce, Number<NSize - 1>{}, make_tuple(init), init);
}
#endif

// TODO: update to like container_reverse_exclusive_scan to deal with Tuple of Numebr<>
template <typename... Xs, typename Reduce, typename TData>
__host__ __device__ constexpr auto
container_reverse_inclusive_scan(const Tuple<Xs...>& x, Reduce f, TData init)
{
    constexpr index_t NSize = sizeof...(Xs);

    Tuple<Xs...> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        r    = f(r, x[i]);
        y(i) = r;
    });

    r              = f(r, x[Number<0>{}]);
    y(Number<0>{}) = r;

    return y;
}

template <typename X, typename... Ys>
__host__ __device__ constexpr auto container_concat(const X& x, const Ys&... ys)
{
    return container_concat(x, container_concat(ys...));
}

template <typename T, index_t NX, index_t NY>
__host__ __device__ constexpr auto container_concat(const Array<T, NX>& ax, const Array<T, NY>& ay)
{
    return unpack2(
        [&](auto&&... zs) { return make_array(std::forward<decltype(zs)>(zs)...); }, ax, ay);
}

template <typename... X, typename... Y>
__host__ __device__ constexpr auto container_concat(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return make_tuple(std::forward<decltype(zs)>(zs)...); }, tx, ty);
}

template <typename Container>
__host__ __device__ constexpr auto container_concat(const Container& x)
{
    return x;
}

template <typename T, index_t N, index_t... Is>
__host__ __device__ constexpr auto get_container_subset(const Array<T, N>& arr, Sequence<Is...>)
{
    static_assert(N >= sizeof...(Is), "wrong! size");

    return make_array(arr[Number<Is>{}]...);
}

template <typename... Ts, index_t... Is>
__host__ __device__ constexpr auto get_container_subset(const Tuple<Ts...>& tup, Sequence<Is...>)
{
    static_assert(sizeof...(Ts) >= sizeof...(Is), "wrong! size");

    return make_tuple(tup[Number<Is>{}]...);
}

template <typename T, index_t N, index_t... Is>
__host__ __device__ constexpr void
set_container_subset(Array<T, N>& y, Sequence<Is...> picks, const Array<T, sizeof...(Is)>& x)
{
    static_assert(N >= sizeof...(Is), "wrong! size");

    static_for<0, sizeof...(Is), 1>{}([&](auto i) { y(picks[i]) = x[i]; });
}

template <typename... Ys, index_t... Is, typename... Xs>
__host__ __device__ constexpr void
set_container_subset(Tuple<Ys...>& y, Sequence<Is...> picks, const Tuple<Xs...>& x)
{
    static_assert(sizeof...(Ys) >= sizeof...(Is) && sizeof...(Is) == sizeof...(Xs), "wrong! size");

    static_for<0, sizeof...(Is), 1>{}([&](auto i) { y(picks[i]) = x[i]; });
}

template <index_t... Is>
__host__ __device__ constexpr auto sequence_to_tuple_of_number(Sequence<Is...>)
{
    using Seq = Sequence<Is...>;

    return generate_tuple(
        [&](auto i) {
            constexpr index_t tmp = Seq::At(i);
            return Number<tmp>{};
        },
        Seq::Size());
}

} // namespace ck
#endif
