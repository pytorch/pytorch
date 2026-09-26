/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "ITraceActivity.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"
#include "TypedMetadata.h"

namespace libkineto {

// Link type, used in GenericTraceActivity.flow.type
constexpr unsigned int kLinkFwdBwd = 1;
constexpr unsigned int kLinkAsyncCpuGpu = 2;

// @lint-ignore-every CLANGTIDY
// cppcoreguidelines-non-private-member-variables-in-classes
// @lint-ignore-every CLANGTIDY cppcoreguidelines-pro-type-member-init
class GenericTraceActivity : public ITraceActivity {
 public:
  GenericTraceActivity()
      : activityType(ActivityType::ENUM_COUNT), traceSpan_(nullptr) {}

  GenericTraceActivity(
      const TraceSpan& trace,
      ActivityType type,
      const std::string& name)
      : activityType(type), activityName(name), traceSpan_(&trace) {}

  int64_t deviceId() const override {
    return device;
  }

  int64_t resourceId() const override {
    return resource;
  }

  void setDevice(int32_t newDevice) {
    device = newDevice;
  }

  int32_t getThreadId() const override {
    return threadId;
  }

  int64_t timestamp() const override {
    return startTime;
  }

  int64_t duration() const override {
    return endTime - startTime;
  }

  int64_t correlationId() const override {
    return id;
  }

  ActivityType type() const override {
    return activityType;
  }

  const ITraceActivity* linkedActivity() const override {
    return linked;
  }

  int flowType() const override {
    return flow.type;
  }

  int64_t flowId() const override {
    return flow.id;
  }

  bool flowStart() const override {
    return flow.start;
  }

  const std::string name() const override {
    return activityName;
  }

  const TraceSpan* traceSpan() const override {
    return traceSpan_;
  }

  void log(ActivityLogger& logger) const override;

  // Encode client side metadata as a key/value (as a JSON fragment)
  template <typename ValType>
  void addMetadata(const std::string& key, const ValType& value) {
    metadataMap_.emplace(key, RawJson{fmt::format("{}", value)});
  }

  // Typed metadata: the value is stored as the field's declared type
  template <typename T, typename V>
  void addMetadata(const MetadataField<T>& field, const V& value) {
    static_assert(
        std::is_same_v<T, std::decay_t<V>>,
        "value type must match field's declared type");
    metadataMap_.emplace(std::string{field.name}, TypedValue{value});
  }

  // Adds typed metadata dynamically by key. Catalog registration is not
  // required.
  void addTypedMetadata(std::string_view key, TypedValue value) {
    metadataMap_.emplace(std::string{key}, std::move(value));
  }

  // The value is a plain string to be emitted quoted in JSON.
  void addMetadataQuoted(const std::string& key, const std::string& value) {
    metadataMap_.emplace(key, value);
  }

  // Store a typed counter value. Preferred over addMetadata for counter
  // activities — preserves full double precision and avoids the JSON
  // serialize/deserialize round-trip in output backends.
  void addCounterValue(const std::string& name, double value) {
    counterValues_.emplace_back(name, value);
  }

  const std::vector<std::pair<std::string, double>>& counterValues()
      const override {
    return counterValues_;
  }

  // Return the metadata value in string format
  const std::string getMetadataValue(const std::string& key) const override;

  // Typed read-back
  template <typename T>
  std::optional<T> getMetadataValue(const MetadataField<T>& field) const {
    const auto it = metadataMap_.find(std::string{field.name});
    if (it != metadataMap_.end()) {
      if (const T* value = std::get_if<T>(&it->second)) {
        return *value;
      }
    }
    return std::nullopt;
  }

  const std::string metadataJson() const override;

  void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override {
    // Dynamically build a MetadataField during visit
    for (const auto& kv : metadataMap_) {
      std::visit(
          [&](const auto& v) {
            visitor.visit(
                MetadataField<std::decay_t<decltype(v)>>{kv.first}, v);
          },
          kv.second);
    }
  }

  virtual ~GenericTraceActivity() override {}

  int64_t startTime{0};
  int64_t endTime{0};
  int32_t id{0};
  int32_t device{0};
  int64_t resource{0};
  int32_t threadId{0};
  ActivityType activityType;
  std::string activityName;
  struct Flow {
    Flow() : id(0), type(0), start(0) {}
    // Ids must be unique within each type
    uint32_t id;
    // Type will be used to connect flows between profilers, as
    // well as look up flow information (name etc)
    uint32_t type : 4;
    uint32_t start : 1;
  } flow;
  const ITraceActivity* linked{nullptr};

 private:
  const TraceSpan* traceSpan_;
  std::unordered_map<std::string, TypedValue> metadataMap_;
  // Typed counter values: (name, double) to avoid round-tripping though string
  std::vector<std::pair<std::string, double>> counterValues_;
};

} // namespace libkineto
