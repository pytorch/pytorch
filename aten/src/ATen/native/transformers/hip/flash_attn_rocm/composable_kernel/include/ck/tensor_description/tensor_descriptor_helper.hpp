// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"

namespace ck {

/*
 * These functions create tensor descriptor at runtime. If they are not constexpr, you will
 * likely see usage of scratch memory during construction of these tensor descriptors. So
 * it's better to call these functions on host and then pass the constructed tensor descritpors
 * to GPU. If the tensor descritpors being constructed are constexpr, then you can call these
 * functions on GPU without worrying about scratch memory usage.
 */

#if CK_WORKAROUND_SWDEV_275126
template <typename Lengths, typename Strides, index_t I, typename AccOld>
__host__ __device__ constexpr auto calculate_element_space_size_impl(const Lengths& lengths,
                                                                     const Strides& strides,
                                                                     Number<I> i,
                                                                     AccOld acc_old)
{
    auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

    if constexpr(i.value < Lengths::Size() - 1)
    {
        return calculate_element_space_size_impl(lengths, strides, i + Number<1>{}, acc_new);
    }
    else
    {
        return acc_new;
    }
}
#endif

// Lengths..., Strides... could be:
//   1) index_t, which is known at run-time, or
//   2) Number<>, which is known at compile-time
// element_space_size could be:
//   1) long_index_t, or
//   2) LongNumber<>
template <typename... Lengths,
          typename... Strides,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto make_naive_tensor_descriptor(const Tuple<Lengths...>& lengths,
                                                                const Tuple<Strides...>& strides)
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_embed_transform(lengths, strides));

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

#if !CK_WORKAROUND_SWDEV_275126
    // rocm-4.1 compiler would crash for recursive labmda
    // recursive function for reduction
    auto f = [&](auto fs, auto i, auto acc_old) {
        auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

        if constexpr(i.value < N - 1)
        {
            return fs(fs, i + Number<1>{}, acc_new);
        }
        else
        {
            return acc_new;
        }
    };

    const auto element_space_size = f(f, Number<0>{}, LongNumber<1>{});
#else
    const auto element_space_size =
        calculate_element_space_size_impl(lengths, strides, Number<0>{}, LongNumber<1>{});
#endif

    return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                            remove_cv_t<decltype(low_dim_hidden_idss)>,
                            remove_cv_t<decltype(up_dim_hidden_idss)>,
                            remove_cv_t<decltype(visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>>{transforms,
                                                                       element_space_size};
}

// Lengths... could be:
//   1) index_t, which is known at run-time, or
//   2) Number<>, which is known at compile-time
// element_space_size could be:
//   1) long_index_t, or
//   2) LongNumber<>
template <typename... Lengths>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor_packed(const Tuple<Lengths...>& lengths)
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_unmerge_transform(lengths));

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const auto element_space_size = container_reduce(lengths, math::multiplies{}, LongNumber<1>{});

    return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                            remove_cv_t<decltype(low_dim_hidden_idss)>,
                            remove_cv_t<decltype(up_dim_hidden_idss)>,
                            remove_cv_t<decltype(visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>>{transforms,
                                                                       element_space_size};
}

// Lengths... could be:
//   1) index_t, which is known at run-time, or
//   2) Number<>, which is known at compile-time
// align could be:
//   1) index_t, or
//   2) Number<>
template <typename... Lengths, typename Align>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor_aligned(const Tuple<Lengths...>& lengths, Align align)
{
    constexpr auto I1 = Number<1>{};

    constexpr index_t N = sizeof...(Lengths);

    const auto stride_n_minus_2 = math::integer_least_multiple(lengths[Number<N - 1>{}], align);

    auto strides = generate_tuple(
        [&](auto i) {
            if constexpr(i.value == N - 1)
            {
                return I1;
            }
            else if constexpr(i.value == N - 2)
            {
                return Number<stride_n_minus_2>{};
            }
            else
            {
                return container_reduce(lengths,
                                        math::multiplies{},
                                        Number<stride_n_minus_2>{},
                                        i + I1,
                                        Number<N - 1>{},
                                        I1);
            }
        },
        Number<N>{});

    return make_naive_tensor_descriptor(lengths, strides);
}

} // namespace ck
