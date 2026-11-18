/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "TypedMetadata.h"

#include <fmt/format.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// Migration-only typed metadata -> metadataJson compatibility helpers.
//
// These helpers preserve legacy metadataJson formatting while Kineto producers
// are migrated to typed metadata. Eventually, we want to decouple metadataJson
// from ITraceActivity and rely only on typed metadata as the canonical Kineto
// metadata output. We use these helpers to reduce the risk of drift during
// migration.
//
// IMPORTANT: This file may change or go away once metadataJson is no longer
// needed as an ITraceActivity output.

namespace libkineto::internal {

class JsonTypedMetadataVisitor final : public ITypedMetadataVisitor {
 public:
  JsonTypedMetadataVisitor() {
    json_.reserve(kInitialJsonCapacity);
  }

  [[nodiscard]] std::string json() && {
    return std::move(json_);
  }

  // Public so GenericTraceActivity's metadata serialization can render
  // collections without duplicating this logic.
  template <typename T>
  static void appendArray(std::string& json, const std::vector<T>& values) {
    json += '[';
    bool first = true;
    for (const auto& value : values) {
      if (!first) {
        json += ", ";
      }
      appendArrayValue(json, value);
      first = false;
    }
    json += ']';
  }

 private:
  // Sized to hold the largest common CUDA activity (kernels) in one allocation,
  // with some buffer
  static constexpr size_t kInitialJsonCapacity = 1024;

  // Widest int64_t is "-9223372036854775808" (20 chars)
  static constexpr size_t kMaxInt64Chars = 20;

  void visitValue(const MetadataField<int64_t>& field, int64_t value) override {
    appendField(field, [&](std::string& json) { appendIntValue(json, value); });
  }

  void visitValue(const MetadataField<double>& field, double value) override {
    appendField(
        field, [&](std::string& json) { appendDoubleValue(json, value); });
  }

  void visitValue(const MetadataField<bool>& field, bool value) override {
    appendField(
        field, [&](std::string& json) { json += value ? "true" : "false"; });
  }

  void visitValue(
      const MetadataField<std::string>& field,
      std::string_view value) override {
    appendField(field, [&](std::string& json) { appendQuoted(json, value); });
  }

  void visitValue(
      const MetadataField<std::vector<int64_t>>& field,
      const std::vector<int64_t>& value) override {
    appendField(field, [&](std::string& json) { appendArray(json, value); });
  }

  void visitValue(
      const MetadataField<std::vector<std::string>>& field,
      const std::vector<std::string>& value) override {
    appendField(field, [&](std::string& json) { appendArray(json, value); });
  }

  void visitValue(const MetadataField<RawJson>& field, const RawJson& value)
      override {
    appendField(field, [&](std::string& json) { json += value.value; });
  }

  void visitValue(const MetadataField<uint64_t>& field, uint64_t value)
      override {
    appendField(
        field, [&](std::string& json) { appendUIntValue(json, value); });
  }

  void visitValue(
      const MetadataField<InputShapes>& field,
      const InputShapes& value) override {
    appendField(field, [&](std::string& json) { appendArray(json, value); });
  }

  // Emit a visible placeholder rather than silently dropping a metadata type
  // the JSON serializer doesn't handle, so the gap shows up in the trace.
  void visitUnsupported(std::string_view name) override {
    appendKey(name);
    appendQuoted(json_, "<unsupported metadata type>");
  }

  void beginDict(std::string_view name) override {
    appendKey(name);
    json_ += '{';
    firstEntry_ = true;
  }

  void endDict() override {
    json_ += '}';
    firstEntry_ = false;
  }

  template <typename T, typename WriteValue>
  void appendField(const MetadataField<T>& field, WriteValue writeValue) {
    appendKey(field.name);
    writeValue(json_);
  }

  void appendKey(std::string_view key) {
    if (!firstEntry_) {
      json_ += ", ";
    }
    firstEntry_ = false;
    appendQuoted(json_, key);
    json_ += ": ";
  }

  static void appendQuoted(std::string& json, std::string_view value) {
    json += '"';
    json += value;
    json += '"';
  }

  static void appendIntValue(std::string& json, int64_t value) {
    char buf[kMaxInt64Chars];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value);
    json.append(buf, static_cast<size_t>(result.ptr - buf));
  }

  static void appendUIntValue(std::string& json, uint64_t value) {
    char buf[kMaxInt64Chars];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value);
    json.append(buf, static_cast<size_t>(result.ptr - buf));
  }

  static void appendDoubleValue(std::string& json, double value) {
    if (std::isfinite(value)) {
      fmt::format_to(std::back_inserter(json), "{}", value);
      return;
    }
    appendQuoted(json, fmt::format("{}", value));
  }

  static void appendArrayValue(std::string& json, int64_t value) {
    appendIntValue(json, value);
  }

  static void appendArrayValue(std::string& json, const std::string& value) {
    appendQuoted(json, value);
  }

  // Nested-array entries so InputShapes serializes through the appendArray
  // recursion
  static void appendArrayValue(
      std::string& json,
      const std::vector<int64_t>& value) {
    appendArray(json, value);
  }

  static void appendArrayValue(
      std::string& json,
      const std::variant<std::vector<int64_t>, TensorListShapes>& value) {
    std::visit(
        [&json](const auto& shapes) { appendArray(json, shapes); }, value);
  }

  std::string json_;
  bool firstEntry_ = true;
};

} // namespace libkineto::internal
