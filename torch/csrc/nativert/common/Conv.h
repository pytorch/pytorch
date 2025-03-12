#pragma once

#include <optional>
#include <string_view>

namespace torch::nativert {

template <typename T>
std::optional<T> tryTo(std::string_view symbol) = delete;

/*
 * Convert a string to an integer. prefixes like "0x" or trailing whitespaces
 * are not supported. Similayly, integer string with trailing characters like
 * "123abc" will be rejected either.
 */
template <>
std::optional<int64_t> tryTo<int64_t>(std::string_view symbol);

/*
 * Convert a string to a double. prefixes like "0x" or trailing whitespaces
 * are not supported. Similayly, integer string with trailing characters like
 * "123abc" will be rejected either.
 */
template <>
std::optional<double> tryTo<double>(std::string_view symbol);

} // namespace torch::nativert
