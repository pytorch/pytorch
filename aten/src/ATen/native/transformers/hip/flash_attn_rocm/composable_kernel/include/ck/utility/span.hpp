// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <array>
#include <type_traits>

namespace ck {

template <typename T>
class span
{
    public:
    using element_type    = T;
    using value_type      = std::remove_cv_t<element_type>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer         = element_type*;
    using const_pointer   = const element_type*;
    using reference       = element_type&;
    using const_reference = const element_type&;
    using iterator        = pointer;
    using const_iterator  = pointer;

    constexpr span() : span(nullptr, size_type{0}) {}

    constexpr span(pointer first, size_type count) : ptr_(first), size_(count) {}

    constexpr span(pointer first, pointer last) : span(first, last - first) {}

    template <std::size_t N>
    constexpr span(element_type (&arr)[N]) noexcept : span(arr, N)
    {
    }

    template <std::size_t N>
    constexpr span(std::array<value_type, N>& arr) noexcept : span(arr.data(), N)
    {
    }

    template <typename Container>
    constexpr span(const Container& container) : span(container.data(), container.size())
    {
    }

    constexpr iterator begin() const noexcept { return ptr_; }
    constexpr const_iterator cbegin() const noexcept { return begin(); }

    constexpr iterator end() const noexcept { return begin() + size(); }
    constexpr const_iterator cend() const noexcept { return end(); }

    constexpr reference front() const { return *begin(); }
    constexpr reference back() const { return *(--end()); }

    constexpr reference operator[](size_type idx) const { return *(begin() + idx); }
    constexpr pointer data() const noexcept { return ptr_; }

    constexpr size_type size() const noexcept { return size_; }

    private:
    pointer ptr_;
    size_type size_;
};

} // namespace ck
