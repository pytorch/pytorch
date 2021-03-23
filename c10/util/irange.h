// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <c10/util/Exception.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <type_traits>

namespace c10 {

namespace detail {

template <typename I, typename std::enable_if<std::is_integral<I>::value, int>::type = 0>
struct integer_iterator : std::iterator<std::input_iterator_tag, I> {
    explicit integer_iterator(I value) : value(value) {}

    I operator*() const { return value; }

    I const* operator->() const { return &value; }

    integer_iterator& operator++() {
        ++value;
        return *this;
    }

    integer_iterator operator++(int) {
        const auto copy = *this;
        ++*this;
        return copy;
    }

    bool operator==(const integer_iterator& other) const {
        return value == other.value;
    }

    bool operator!=(const integer_iterator& other) const {
        return value != other.value;
    }

 protected:
    I value;
};

} // namespace detail

template <typename I, typename std::enable_if<std::is_integral<I>::value, bool>::type = true>
struct integer_range {
 public:
    integer_range(I begin, I end) : begin_(begin), end_(end) {}
    detail::integer_iterator<I> begin() const { return begin_; }
    detail::integer_iterator<I> end() const { return end_; }

 private:
    detail::integer_iterator<I> begin_;
    detail::integer_iterator<I> end_;
};

/// Creates an integer range for the half-open interval [begin, end)
/// If end<=begin, then the range is empty.
/// The range has the type of the `end` integer; `begin` integer is
/// cast to this type.
template <
    typename Integer1,
    typename Integer2,
    typename std::enable_if<std::is_integral<Integer1>::value, bool>::type = true,
    typename std::enable_if<std::is_integral<Integer2>::value, bool>::type = true
>
integer_range<Integer2> irange(Integer1 begin, Integer2 end) {
    //If end<=begin then the range is empty; we can achieve this effect by
    //choosing the larger of {begin, end} as the loop terminator
    return {static_cast<Integer2>(begin), std::max(static_cast<Integer2>(begin), end)};
}

/// Creates an integer range for the half-open interval [0, end)
/// If end<=begin, then the range is empty
template <typename Integer, typename std::enable_if<std::is_integral<Integer>::value, bool>::type = true>
integer_range<Integer> irange(Integer end) {
    //If end<=begin then the range is empty; we can achieve this effect by
    //choosing the larger of {0, end} as the loop terminator
    //Handles the case where end<0. irange only works for ranges >=0
    return {Integer(), std::max(Integer(), end)};
}

} // namespace torch
