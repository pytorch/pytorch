/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "GenericTraceActivity.h"
#include "TypedMetadataJson.h"
#include "output_base.h"

namespace libkineto {

namespace {
template <typename>
inline constexpr bool kAlwaysFalse = false;

// Render a stored metadata value back to its legacy string form.
std::string metadataValueToString(const TypedValue& value) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, RawJson>) {
          return v.value;
        } else if constexpr (
            std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
            std::is_same_v<T, double> || std::is_same_v<T, bool>) {
          return fmt::format("{}", v);
        } else if constexpr (
            std::is_same_v<T, std::vector<int64_t>> ||
            std::is_same_v<T, std::vector<std::string>> ||
            std::is_same_v<T, InputShapes>) {
          // Serialize collections through the JSON visitor's array helper.
          std::string out;
          internal::JsonTypedMetadataVisitor::appendArray(out, v);
          return out;
        } else {
          static_assert(
              kAlwaysFalse<T>,
              "unhandled TypedValue alternative in metadataValueToString");
        }
      },
      value);
}
} // namespace

void GenericTraceActivity::log(ActivityLogger& logger) const {
  logger.handleGenericActivity(*this);
}

const std::string GenericTraceActivity::getMetadataValue(
    const std::string& key) const {
  const auto it = metadataMap_.find(key);
  return it == metadataMap_.end() ? "" : metadataValueToString(it->second);
}

const std::string GenericTraceActivity::metadataJson() const {
  internal::JsonTypedMetadataVisitor visitor;
  visitTypedMetadata(visitor);
  return std::move(visitor).json();
}

} // namespace libkineto
