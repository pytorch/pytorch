// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_permute_impl.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/utility/type.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;
using F64 = double;

struct Problem final
{
    static constexpr std::size_t NumDim = 3;

    using Shape = std::array<std::size_t, NumDim>;
    using Axes  = Shape;

    Problem() = delete;

    explicit Problem(const Shape& default_shape, const Axes& default_axes)
        : shape(default_shape), axes(default_axes)
    {
    }

    Shape shape;
    Axes axes;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

namespace detail {

template <typename Array, std::size_t Difference>
struct enlarge_array_size;

template <typename T, std::size_t Size, std::size_t Difference>
struct enlarge_array_size<std::array<T, Size>, Difference>
{
    using type = std::array<T, Size + Difference>;
};

template <typename Array, std::size_t Difference>
using enlarge_array_size_t = typename enlarge_array_size<Array, Difference>::type;

template <typename Array>
struct get_array_size;

template <typename T, std::size_t Size>
struct get_array_size<std::array<T, Size>> : std::integral_constant<std::size_t, Size>
{
};

template <typename Array>
inline constexpr std::size_t get_array_size_v = get_array_size<Array>::value;

template <typename T, typename = void>
struct is_iterator : std::false_type
{
};

template <typename T>
struct is_iterator<T,
                   std::void_t<decltype(*std::declval<T>()),
                               decltype(++std::declval<std::add_lvalue_reference_t<T>>()),
                               decltype(std::declval<std::add_lvalue_reference_t<T>>()++)>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

struct Placeholder final
{
    template <typename T>
    constexpr inline operator T() const noexcept;
};

template <typename Iterator, typename = void>
struct is_output_iterator : std::false_type
{
};

template <typename Iterator>
struct is_output_iterator<
    Iterator,
    std::void_t<decltype(*std::declval<Iterator>() = std::declval<Placeholder>())>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename T>
inline constexpr bool is_output_iterator_v = is_output_iterator<T>::value;

template <typename Iterator, typename = void>
struct is_bidirectional_iterator : std::false_type
{
};

template <typename Iterator>
struct is_bidirectional_iterator<
    Iterator,
    std::void_t<decltype(--std::declval<std::add_lvalue_reference_t<Iterator>>()),
                decltype(std::declval<std::add_lvalue_reference_t<Iterator>>()--)>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename Iterator>
inline constexpr bool is_bidirectional_iterator_v = is_bidirectional_iterator<Iterator>::value;

template <typename Iterator, typename = void>
struct is_random_access_iterator : std::false_type
{
};

template <typename Iterator>
struct is_random_access_iterator<Iterator,
                                 std::void_t<decltype(std::declval<Iterator>() + 1),
                                             decltype(std::declval<Iterator>() - 1),
                                             decltype(std::declval<Iterator>()[1])>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename Iterator>
inline constexpr bool is_random_access_iterator_v = is_random_access_iterator<Iterator>::value;

template <typename T, typename = void>
struct is_range : std::false_type
{
};

template <typename T>
struct is_range<T,
                std::void_t<decltype(begin(std::declval<T>())),
                            decltype(end(std::declval<T>())),
                            decltype(begin(std::declval<T>()) != end(std::declval<T>()))>>
    : std::bool_constant<is_iterator_v<ck::remove_cvref_t<decltype(begin(std::declval<T>()))>>>
{
};

template <typename T>
inline constexpr bool is_range_v = is_range<T>::value;

template <typename Range, typename = void>
struct is_sized_range : std::false_type
{
};

template <typename Range>
struct is_sized_range<Range, std::void_t<decltype(size(std::declval<Range>()))>>
    : std::bool_constant<is_range_v<Range>>
{
};

template <typename Range>
inline constexpr bool is_sized_range_v = is_sized_range<Range>::value;

template <typename Range, typename = void>
struct is_bidirectional_range : std::false_type
{
};

template <typename Range>
struct is_bidirectional_range<Range, std::void_t<>>
    : std::bool_constant<
          is_range_v<Range> &&
          is_bidirectional_iterator_v<ck::remove_cvref_t<decltype(begin(std::declval<Range>()))>>>
{
};

template <typename Range>
inline constexpr bool is_bidirectional_range_v = is_bidirectional_range<Range>::value;

template <typename Range, typename = void>
struct is_random_access_range : std::false_type
{
};

template <typename Range>
struct is_random_access_range<Range, std::void_t<>>
    : std::bool_constant<
          is_range_v<Range> &&
          is_random_access_iterator_v<ck::remove_cvref_t<decltype(begin(std::declval<Range>()))>>>
{
};

template <typename Range>
inline constexpr bool is_random_access_range_v = is_random_access_range<Range>::value;

template <typename Range>
class to_array_proxy
{
    static_assert(is_range_v<Range>);

    public:
    explicit to_array_proxy(const Range& source) noexcept : source_(source) {}

    template <typename T, std::size_t Size>
    operator std::array<T, Size>() const
    {
        std::array<T, Size> destination;

        std::copy_n(std::begin(source_),
                    std::min<std::size_t>(Size, std::size(source_)),
                    std::begin(destination));

        return destination;
    }

    private:
    const Range& source_;
};

} // namespace detail

template <typename Range>
inline auto to_array(Range& range) noexcept
    -> std::enable_if_t<detail::is_range_v<Range>,
                        detail::to_array_proxy<ck::remove_cvref_t<Range>>>
{
    return detail::to_array_proxy<ck::remove_cvref_t<Range>>{range};
}

template <typename Axes>
inline auto is_valid_axes(const Axes& axes)
    -> std::enable_if_t<detail::is_random_access_range_v<Axes>, bool>
{
    using std::empty;
    if(empty(axes))
    {
        return false;
    }

    using std::begin, std::end;
    std::vector<std::size_t> sorted_axes(begin(axes), end(axes));

    std::sort(begin(sorted_axes), end(sorted_axes));
    const auto last = std::unique(begin(sorted_axes), end(sorted_axes));

    return (last == end(sorted_axes)) && (*begin(sorted_axes) == 0) &&
           (*std::prev(last) == size(axes) - 1);
}

template <typename Shape>
inline auto is_valid_shape(const Shape& shape) -> std::enable_if_t<detail::is_range_v<Shape>, bool>
{
    static_assert(std::is_unsigned_v<ck::remove_cvref_t<decltype(*std::begin(shape))>>);

    using std::begin, std::end;
    using std::empty;
    return !empty(shape) && std::all_of(begin(shape), end(shape), [](auto dim) { return 0 < dim; });
}

template <typename Shape, typename Indices>
inline auto is_valid_indices(const Shape& shape, const Indices& indices)
    -> std::enable_if_t<detail::is_sized_range_v<Shape> && detail::is_sized_range_v<Indices>, bool>
{
    static_assert(std::is_unsigned_v<ck::remove_cvref_t<decltype(*std::begin(indices))>>);

    if(!is_valid_shape(shape))
    {
        return false;
    }

    using std::empty;
    if(empty(indices))
    {
        return false;
    }

    using std::size;
    if(size(shape) != size(indices))
    {
        return false;
    }

    using std::begin, std::end;

    auto dim = begin(shape);
    auto idx = begin(indices);
    for(; dim != end(shape) && idx != end(indices); ++dim, ++idx)
    {
        if(*dim <= *idx)
        {
            return false;
        }
    }

    return true;
}

template <std::size_t Size>
std::array<std::size_t, Size> transpose(const std::array<std::size_t, Size>& shape,
                                        const std::array<std::size_t, Size>& axes)
{
    assert(is_valid_shape(shape) && is_valid_axes(axes));

    std::array<std::size_t, Size> transposed;
    auto iter = std::begin(transposed);
    for(const auto axis : axes)
    {
        *iter++ = shape[axis];
    }

    return transposed;
}

auto extend_shape(const Problem::Shape& shape, std::size_t new_dim)
{
    detail::enlarge_array_size_t<Problem::Shape, 1> extended_shape;

    using std::begin, std::end;

    ck::ranges::copy(shape, begin(extended_shape));
    extended_shape.back() = new_dim;

    return extended_shape;
}

auto extend_axes(const Problem::Axes& axes)
{
    detail::enlarge_array_size_t<Problem::Axes, 1> extended_axes;

    using std::begin, std::end;

    ck::ranges::copy(axes, begin(extended_axes));
    extended_axes.back() = detail::get_array_size_v<Problem::Axes>;

    return extended_axes;
}

template <typename Shape, typename Indices>
auto advance_indices(const Shape& shape, Indices& indices) -> std::enable_if_t<
    detail::is_bidirectional_range_v<Shape> && detail::is_sized_range_v<Shape> &&
        detail::is_bidirectional_range_v<Indices> && detail::is_sized_range_v<Indices>,
    bool>
{
    using std::size;
    if(!(is_valid_shape(shape) && is_valid_indices(shape, indices) && size(shape) == size(indices)))
    {
        return false;
    }

    bool carry = true;

    using std::rbegin, std::rend;
    auto dim = rbegin(shape);
    auto idx = rbegin(indices);
    for(; carry && dim != rend(shape) && idx != rend(indices); ++dim, ++idx)
    {
        *idx  = (*idx + carry);
        carry = ((*idx == *dim) ? (*idx = 0, true) : false);
    }

    return !carry;
}

template <typename Src, typename Axes, typename Functor, typename Dest>
auto host_permute(const Tensor<Src>& src, const Axes& axes, Functor functor, Tensor<Dest>& dest)
    -> std::enable_if_t<detail::is_random_access_range_v<Axes> && detail::is_sized_range_v<Axes> &&
                            std::is_invocable_v<Functor,
                                                std::add_lvalue_reference_t<Dest>,
                                                std::add_lvalue_reference_t<Src>>,
                        bool>
{
    const auto& shape            = src.mDesc.GetLengths();
    const auto& transposed_shape = dest.mDesc.GetLengths();
    if(!(is_valid_shape(shape) && is_valid_shape(transposed_shape)))
    {
        return false;
    }

    using std::size;
    if(!is_valid_axes(axes))
    {
        return false;
    }

    static_assert(detail::is_sized_range_v<ck::remove_cvref_t<decltype(shape)>> &&
                  detail::is_sized_range_v<ck::remove_cvref_t<decltype(transposed_shape)>>);

    if(size(shape) != size(transposed_shape))
    {
        return false;
    }

    static_assert(detail::is_random_access_range_v<ck::remove_cvref_t<decltype(shape)>> &&
                  detail::is_random_access_range_v<ck::remove_cvref_t<decltype(transposed_shape)>>);
    {
        for(std::size_t idx = 0; idx < size(shape); ++idx)
        {
            if(transposed_shape[idx] != shape[axes[idx]])
            {
                return false;
            }
        }
    }

    std::vector<std::size_t> indices(size(shape), 0);
    if(!is_valid_indices(shape, indices))
    {
        return false;
    }

    switch(size(shape))
    {
    case 3: {
        do
        {
            Dest output = 0;
            functor(output, src(indices[0], indices[1], indices[2]));
            dest(indices[axes[0]], indices[axes[1]], indices[axes[2]]) = output;
        } while(advance_indices(shape, indices));
    }
    break;
    case 4: {
        do
        {
            Dest output = 0;
            functor(output, src(indices[0], indices[1], indices[2], indices[3]));
            dest(indices[axes[0]], indices[axes[1]], indices[axes[2]], indices[axes[3]]) = output;
        } while(advance_indices(shape, indices));
    }
    break;
    default: return false;
    }

    return true;
}
