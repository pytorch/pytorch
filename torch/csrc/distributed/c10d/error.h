// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cerrno>
#include <system_error>

#include <fmt/format.h>

namespace fmt {

template <>
struct formatter<std::error_code> {
  constexpr decltype(auto) parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  decltype(auto) format(const std::error_code& err, FormatContext& ctx) {
#ifdef _WIN32
    return format_to(ctx.out(), "(Error: {})", err.value());
#else
    return format_to(ctx.out(), "(Error: {} - {})", err.value(), err.message());
#endif
  }
};

} // namespace fmt

namespace c10d {
namespace detail {

inline std::error_code lastError() noexcept {
  return std::error_code{errno, std::generic_category()};
}

} // namespace detail
} // namespace c10d
