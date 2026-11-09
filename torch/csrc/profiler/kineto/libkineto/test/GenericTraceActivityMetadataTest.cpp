/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/GenericTraceActivity.h"
#include "include/TypedMetadata.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

using namespace libkineto;

namespace {

// Records each visited field tagged by the visitValue overload that fired, so
// tests can tell the verbatim RawJson path (string-keyed addMetadata) apart
// from the typed path (the MetadataField overload).
class RecordingVisitor : public ITypedMetadataVisitor {
 public:
  std::map<std::string, std::string> recorded;

  void visitValue(const MetadataField<int64_t>& field, int64_t value) override {
    recorded[std::string{field.name}] = fmt::format("int64={}", value);
  }
  void visitValue(const MetadataField<double>& field, double value) override {
    recorded[std::string{field.name}] = fmt::format("double={}", value);
  }
  void visitValue(const MetadataField<bool>& field, bool value) override {
    recorded[std::string{field.name}] = fmt::format("bool={}", value);
  }
  void visitValue(
      const MetadataField<std::string>& field,
      std::string_view value) override {
    recorded[std::string{field.name}] = fmt::format("string={}", value);
  }
  void visitValue(const MetadataField<RawJson>& field, const RawJson& value)
      override {
    recorded[std::string{field.name}] = fmt::format("rawjson={}", value.value);
  }
  void visitValue(
      const MetadataField<std::vector<int64_t>>& field,
      const std::vector<int64_t>& /*value*/) override {
    recorded[std::string{field.name}] = "vector<int64>";
  }
  void visitValue(
      const MetadataField<std::vector<std::string>>& field,
      const std::vector<std::string>& /*value*/) override {
    recorded[std::string{field.name}] = "vector<string>";
  }
  void visitValue(const MetadataField<uint64_t>& field, uint64_t value)
      override {
    recorded[std::string{field.name}] = fmt::format("uint64={}", value);
  }
  void visitValue(
      const MetadataField<InputShapes>& field,
      const InputShapes& /*value*/) override {
    recorded[std::string{field.name}] = "inputshapes";
  }
  void visitUnsupported(std::string_view name) override {
    recorded[std::string{name}] = "unsupported";
  }
  void beginDict(std::string_view /*name*/) override {}
  void endDict() override {}
};

constexpr uint64_t kBigUnsigned = 18446744073709551615ULL;

constexpr MetadataField<int64_t> kCount{"count"};
constexpr MetadataField<double> kRatio{"ratio"};
constexpr MetadataField<bool> kFlag{"flag"};
constexpr MetadataField<std::string> kLabel{"label"};
constexpr MetadataField<uint64_t> kAddress{"address"};
constexpr MetadataField<InputShapes> kInputDims{"input_dims"};

} // namespace

TEST(GenericTraceActivityMetadataTest, StringKeyAddMetadataStoresRawJson) {
  GenericTraceActivity activity;
  activity.addMetadata("count", 5);
  activity.addMetadata("ratio", 1.5);
  activity.addMetadata("flag", true);
  activity.addMetadataQuoted("label", "hello");
  activity.addMetadata("dims", "[1, 2, 3]");
  activity.addMetadata("addr", kBigUnsigned);

  RecordingVisitor visitor;
  activity.visitTypedMetadata(visitor);

  // String-keyed addMetadata stores every non-string value as a verbatim JSON
  // fragment
  const std::map<std::string, std::string> expected{
      {"count", "rawjson=5"},
      {"ratio", "rawjson=1.5"},
      {"flag", "rawjson=true"},
      {"label", "string=hello"},
      {"dims", "rawjson=[1, 2, 3]"},
      {"addr", "rawjson=18446744073709551615"},
  };
  EXPECT_EQ(visitor.recorded, expected);
}

TEST(GenericTraceActivityMetadataTest, TypedFieldOverloadStoresDeclaredType) {
  GenericTraceActivity activity;
  activity.addMetadata(kCount, int64_t{5});
  activity.addMetadata(kRatio, 1.5);
  activity.addMetadata(kFlag, true);
  activity.addMetadata(kLabel, std::string{"hello"});

  RecordingVisitor visitor;
  activity.visitTypedMetadata(visitor);

  // The MetadataField overload stores each value as its declared type.
  const std::map<std::string, std::string> expected{
      {"count", "int64=5"},
      {"ratio", "double=1.5"},
      {"flag", "bool=true"},
      {"label", "string=hello"},
  };
  EXPECT_EQ(visitor.recorded, expected);
}

TEST(GenericTraceActivityMetadataTest, AddTypedMetadataStoresTypedValue) {
  GenericTraceActivity activity;
  activity.addTypedMetadata("count", TypedValue{int64_t{5}});

  RecordingVisitor visitor;
  activity.visitTypedMetadata(visitor);

  const std::map<std::string, std::string> expected{{"count", "int64=5"}};
  EXPECT_EQ(visitor.recorded, expected);
}

TEST(GenericTraceActivityMetadataTest, MetadataJsonPreservesValueShapes) {
  GenericTraceActivity activity;
  activity.addMetadata("count", 5);
  activity.addMetadata("ratio", 1.5);
  activity.addMetadata("flag", true);
  activity.addMetadataQuoted("label", "hello");
  activity.addMetadata("dims", "[1, 2, 3]");
  activity.addMetadata("addr", kBigUnsigned);

  // Order is unspecified (unordered_map), so assert each field independently.
  const std::string json = activity.metadataJson();
  EXPECT_NE(json.find("\"count\": 5"), std::string::npos);
  EXPECT_NE(json.find("\"ratio\": 1.5"), std::string::npos);
  EXPECT_NE(json.find("\"flag\": true"), std::string::npos);
  EXPECT_NE(json.find("\"label\": \"hello\""), std::string::npos);
  // A raw JSON array stays an array, not a quoted string.
  EXPECT_NE(json.find("\"dims\": [1, 2, 3]"), std::string::npos);
  // 64-bit unsigned is emitted verbatim, not truncated.
  EXPECT_NE(json.find("\"addr\": 18446744073709551615"), std::string::npos);
}

TEST(GenericTraceActivityMetadataTest, GetMetadataValueReturnsStringForm) {
  GenericTraceActivity activity;
  activity.addMetadata("count", 5);
  activity.addMetadataQuoted("label", "hello");
  activity.addMetadata("dims", "[1, 2, 3]");
  activity.addMetadata("addr", kBigUnsigned);

  // Each value comes back in the same string form addMetadata used to store
  EXPECT_EQ(activity.getMetadataValue("count"), "5");
  EXPECT_EQ(activity.getMetadataValue("label"), "hello");
  EXPECT_EQ(activity.getMetadataValue("dims"), "[1, 2, 3]");
  EXPECT_EQ(activity.getMetadataValue("addr"), "18446744073709551615");
  EXPECT_EQ(activity.getMetadataValue("missing"), "");
}

TEST(GenericTraceActivityMetadataTest, TypedGetMetadataValueReturnsTypedValue) {
  GenericTraceActivity activity;
  activity.addMetadata(kCount, int64_t{5});
  activity.addMetadata(kLabel, std::string{"hello"});
  activity.addMetadata("rawInt", 7); // string path -> stored as RawJson

  EXPECT_EQ(activity.getMetadataValue(kCount), std::optional<int64_t>{5});
  EXPECT_EQ(
      activity.getMetadataValue(kLabel), std::optional<std::string>{"hello"});
  // Missing key.
  EXPECT_EQ(activity.getMetadataValue(kRatio), std::nullopt);
  // Stored as RawJson via the string path, so a typed int64 read doesn't match.
  EXPECT_EQ(
      activity.getMetadataValue(MetadataField<int64_t>{"rawInt"}),
      std::nullopt);
}

TEST(GenericTraceActivityMetadataTest, TypedFieldOverloadStoresRichTypes) {
  GenericTraceActivity activity;
  activity.addMetadata(kAddress, kBigUnsigned);
  InputShapes dims;
  dims.emplace_back(std::vector<int64_t>{2, 2});
  dims.emplace_back(TensorListShapes{{4, 1}, {4, 1}});
  activity.addMetadata(kInputDims, dims);

  // uint64 is stored typed and read back without truncation.
  EXPECT_EQ(
      activity.getMetadataValue(kAddress),
      std::optional<uint64_t>{kBigUnsigned});

  // Both rich types reach the visitor as their declared types.
  RecordingVisitor visitor;
  activity.visitTypedMetadata(visitor);
  EXPECT_EQ(visitor.recorded.at("address"), "uint64=18446744073709551615");
  EXPECT_EQ(visitor.recorded.at("input_dims"), "inputshapes");
}

TEST(
    GenericTraceActivityMetadataTest,
    GetMetadataValueSerializesTypedCollections) {
  GenericTraceActivity activity;
  activity.addMetadata(
      MetadataField<std::vector<int64_t>>{"vec"},
      std::vector<int64_t>{1, 2, 3});
  InputShapes dims;
  dims.emplace_back(std::vector<int64_t>{2, 2});
  dims.emplace_back(TensorListShapes{{4, 1}, {4, 1}});
  activity.addMetadata(kInputDims, dims);

  // The string accessor renders a typed collection as the same JSON array
  // string metadataJson() emits, not "".
  EXPECT_EQ(activity.getMetadataValue("vec"), "[1, 2, 3]");
  EXPECT_EQ(
      activity.getMetadataValue("input_dims"), "[[2, 2], [[4, 1], [4, 1]]]");
}
