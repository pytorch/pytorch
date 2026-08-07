/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/TypedMetadata.h"
#include "include/TypedMetadataJson.h"

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

using namespace libkineto;

namespace {

constexpr MetadataField<int64_t> kCount{"count"};
constexpr MetadataField<std::vector<int64_t>> kIds{"ids"};
constexpr MetadataField<std::string> kLabel{"label"};
constexpr MetadataField<double> kRatio{"ratio"};
constexpr MetadataField<bool> kEnabled{"enabled"};
constexpr MetadataField<std::vector<std::string>> kNames{"names"};
constexpr MetadataField<std::pair<int64_t, int64_t>> kPair{"pair"};
constexpr MetadataField<std::string> kSpecialChars{"special"};
constexpr MetadataField<RawJson> kRaw{"raw"};
constexpr MetadataField<uint64_t> kAddress{"address"};
constexpr MetadataField<InputShapes> kInputDims{"input_dims"};
constexpr MetadataDict kNested{"nested"};
constexpr MetadataField<int64_t> kNestedCount{"nested_count"};
constexpr MetadataField<int64_t> kNestedOther{"nested_other"};
constexpr MetadataField<int64_t> kAfterNested{"after_nested"};
constexpr MetadataDict kOuter{"outer"};
constexpr MetadataDict kInner{"inner"};

using RecordedValue = std::variant<
    int64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<std::string>>;

class RecordingTypedMetadataVisitor : public ITypedMetadataVisitor {
 public:
  void visitValue(const MetadataField<int64_t>& field, int64_t value) override {
    record(field, value);
  }

  void visitValue(const MetadataField<double>& field, double value) override {
    record(field, value);
  }

  void visitValue(const MetadataField<bool>& field, bool value) override {
    record(field, value);
  }

  void visitValue(
      const MetadataField<std::string>& field,
      std::string_view value) override {
    record(field, std::string{value});
  }

  void visitValue(
      const MetadataField<std::vector<int64_t>>& field,
      const std::vector<int64_t>& value) override {
    record(field, value);
  }

  void visitValue(
      const MetadataField<std::vector<std::string>>& field,
      const std::vector<std::string>& value) override {
    record(field, value);
  }

  void visitValue(const MetadataField<RawJson>& field, const RawJson& value)
      override {
    record(field, std::string{value.value});
  }

  void visitValue(
      [[maybe_unused]] const MetadataField<uint64_t>& field,
      [[maybe_unused]] uint64_t value) override {}

  void visitValue(
      [[maybe_unused]] const MetadataField<InputShapes>& field,
      [[maybe_unused]] const InputShapes& value) override {}

  void visitUnsupported(std::string_view /*name*/) override {}

  void beginDict(std::string_view /*name*/) override {}
  void endDict() override {}

  std::map<std::string, RecordedValue> values;

 private:
  template <typename T>
  void record(const MetadataField<T>& field, RecordedValue value) {
    values[std::string{field.name}] = std::move(value);
  }
};

class UnsupportedRecordingTypedMetadataVisitor final
    : public RecordingTypedMetadataVisitor {
 public:
  void visitUnsupported(std::string_view name) override {
    unsupportedFields.emplace_back(name);
  }

  std::vector<std::string> unsupportedFields;
};

} // namespace

TEST(TypedMetadataVisitorTest, VisitsTypedFields) {
  RecordingTypedMetadataVisitor recorder;
  ITypedMetadataVisitor& visitor = recorder;

  visitor.visit(kCount, int64_t{5});
  visitor.visit(kIds, std::vector<int64_t>{1, 2});
  visitor.visit(kLabel, std::string{"label"});
  visitor.visit(kRatio, 1.5);
  visitor.visit(kEnabled, true);
  visitor.visit(kNames, std::vector<std::string>{"a", "b"});

  EXPECT_EQ(std::get<int64_t>(recorder.values.at("count")), int64_t{5});
  EXPECT_EQ(
      std::get<std::vector<int64_t>>(recorder.values.at("ids")),
      std::vector<int64_t>({1, 2}));
  EXPECT_EQ(
      std::get<std::string>(recorder.values.at("label")), std::string{"label"});
  EXPECT_EQ(std::get<double>(recorder.values.at("ratio")), 1.5);
  EXPECT_EQ(std::get<bool>(recorder.values.at("enabled")), true);
  EXPECT_EQ(
      std::get<std::vector<std::string>>(recorder.values.at("names")),
      std::vector<std::string>({"a", "b"}));
}

TEST(TypedMetadataVisitorTest, SerializesVisitedFieldsToJson) {
  internal::JsonTypedMetadataVisitor jsonVisitor;
  ITypedMetadataVisitor& visitor = jsonVisitor;

  visitor.visit(kCount, int64_t{5});
  visitor.visit(kIds, std::vector<int64_t>{1, 2});
  visitor.visit(kLabel, std::string{"label"});
  visitor.visit(kRatio, 1.5);
  visitor.visit(kEnabled, true);
  visitor.visit(kNames, std::vector<std::string>{"a", "b"});
  visitor.visit(kSpecialChars, std::string{"quote \" slash \\ newline\n"});
  visitor.visit(kNested, [&](auto& d) {
    d.visit(kNestedCount, int64_t{7});
    d.visit(kNestedOther, int64_t{8});
  });
  visitor.visit(kAfterNested, int64_t{9});

  const auto json = std::move(jsonVisitor).json();

  EXPECT_NE(json.find("\"count\": 5"), std::string::npos);
  EXPECT_NE(json.find("\"ids\": [1, 2]"), std::string::npos);
  EXPECT_NE(json.find("\"label\": \"label\""), std::string::npos);
  EXPECT_NE(json.find("\"ratio\": 1.5"), std::string::npos);
  EXPECT_NE(json.find("\"enabled\": true"), std::string::npos);
  EXPECT_NE(json.find("\"names\": [\"a\", \"b\"]"), std::string::npos);
  // String values pass through verbatim; output_json.cpp sanitizes the full
  // metadata fragment before appending it to the Chrome trace.
  EXPECT_NE(
      json.find("\"special\": \"quote \" slash \\ newline\n\""),
      std::string::npos);
  EXPECT_NE(
      json.find("\"nested\": {\"nested_count\": 7, \"nested_other\": 8}"),
      std::string::npos);
  EXPECT_NE(json.find("\"after_nested\": 9"), std::string::npos);
}

TEST(TypedMetadataVisitorTest, SerializesNestedDictsToJson) {
  internal::JsonTypedMetadataVisitor jsonVisitor;
  ITypedMetadataVisitor& visitor = jsonVisitor;

  visitor.visit(kOuter, [&](auto& outer) {
    outer.visit(kCount, int64_t{1});
    outer.visit(kInner, [&](auto& inner) { inner.visit(kEnabled, true); });
    outer.visit(kRatio, 2.5);
  });
  visitor.visit(kAfterNested, int64_t{9});

  const auto json = std::move(jsonVisitor).json();

  // A dict nested inside another dict, with sibling fields before and after the
  // inner dict, then a field back at the top level.
  EXPECT_NE(
      json.find(
          "\"outer\": {\"count\": 1, \"inner\": {\"enabled\": true}, \"ratio\": 2.5}"),
      std::string::npos);
  EXPECT_NE(json.find("\"after_nested\": 9"), std::string::npos);
}

TEST(TypedMetadataVisitorTest, FallsBackToVisitUnsupported) {
  UnsupportedRecordingTypedMetadataVisitor recorder;
  ITypedMetadataVisitor& visitor = recorder;

  visitor.visit(kCount, int64_t{5});
  visitor.visit(kPair, std::pair<int64_t, int64_t>{1, 2});

  EXPECT_EQ(std::get<int64_t>(recorder.values.at("count")), int64_t{5});
  EXPECT_EQ(recorder.unsupportedFields, std::vector<std::string>({"pair"}));
}

TEST(TypedMetadataVisitorTest, SerializesRawJsonVerbatim) {
  internal::JsonTypedMetadataVisitor jsonVisitor;
  ITypedMetadataVisitor& visitor = jsonVisitor;

  visitor.visit(kCount, int64_t{5});
  visitor.visit(kRaw, RawJson{"[1, 2, 3]"});

  const auto json = std::move(jsonVisitor).json();

  // RawJson is emitted as-is, unquoted, so an array stays an array.
  EXPECT_NE(json.find("\"raw\": [1, 2, 3]"), std::string::npos);
}

TEST(TypedMetadataVisitorTest, SerializesUInt64AndInputShapesToJson) {
  internal::JsonTypedMetadataVisitor jsonVisitor;
  ITypedMetadataVisitor& visitor = jsonVisitor;

  visitor.visit(kAddress, uint64_t{18446744073709551615ULL});
  InputShapes dims;
  dims.emplace_back(std::vector<int64_t>{2, 2});
  dims.emplace_back(TensorListShapes{{4, 1}, {4, 1}});
  visitor.visit(kInputDims, dims);

  const auto json = std::move(jsonVisitor).json();

  // uint64 is emitted as an unsigned number, not truncated.
  EXPECT_NE(json.find("\"address\": 18446744073709551615"), std::string::npos);
  // Each arg is a single tensor's shape or a nested list of tensor shapes.
  EXPECT_NE(
      json.find("\"input_dims\": [[2, 2], [[4, 1], [4, 1]]]"),
      std::string::npos);
}
