// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

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
    return format_to(ctx.out(),
                     "({} error: {} - {})",
                     err.category().name(),
                     err.value(),
                     err.message());
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
