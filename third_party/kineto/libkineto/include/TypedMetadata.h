/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace libkineto {

template <typename T>
struct MetadataField {
  using FieldType = T;

  std::string_view name;
};

struct MetadataDict {
  std::string_view name;
};

// Used to identify JSON fields from normal strings.
struct RawJson {
  std::string value;
};

// For a TensorList op argument: the per-dimension shape of each tensor in the
// list.
using TensorListShapes = std::vector<std::vector<int64_t>>;

// "Input Dims" / "Input Strides" payload: one entry per op argument, each
// either a single tensor's shape or, for a TensorList argument, a list of
// tensor shapes.
using InputShapes =
    std::vector<std::variant<std::vector<int64_t>, TensorListShapes>>;

// The set of value types the typed-metadata system supports.
using TypedValue = std::variant<
    int64_t,
    uint64_t,
    double,
    bool,
    std::string,
    std::vector<int64_t>,
    std::vector<std::string>,
    RawJson,
    InputShapes>;

/*
 * ITypedMetadataVisitor is a per-activity visitor for structured metadata with
 * field-level type checking.
 *
 * Usage:
 *   // Declare each field once. FieldType is int64_t here, and visit() enforces
 *   // that declared type.
 *   inline constexpr MetadataField<int64_t> kCorrelation{"correlation"};
 *
 *   // Producer: emit metadata from ITraceActivity::visitTypedMetadata().
 *   void visitTypedMetadata(ITypedMetadataVisitor& visitor) const override {
 *     visitor.visit(kCorrelation, int64_t{activity.correlationId});
 *   }
 *
 *   // Consumer: implement visitValue() overloads for handled field types.
 *   class MyVisitor : public ITypedMetadataVisitor {
 *     ...
 *   };
 *
 *   // Consumer: pass the visitor to an activity for generic output conversion.
 *   MyVisitor visitor;
 *   activity.visitTypedMetadata(visitor);
 */
class ITypedMetadataVisitor {
 public:
  virtual ~ITypedMetadataVisitor() = default;

  template <typename T, typename V>
  void visit(const MetadataField<T>& field, const V& value) {
    using FieldType = typename MetadataField<T>::FieldType;
    static_assert(
        std::is_same_v<FieldType, std::decay_t<V>>,
        "value type must match field's declared type");
    visitValue(field, value);
  }

  // Visits a nested dict: everything `visitChildren` visits is grouped under
  // dict.name. Example:
  // visitor.visit(kStats, [](auto& d) { d.visit(kCount, int64_t{1}); });
  void visit(
      const MetadataDict& dict,
      std::invocable<ITypedMetadataVisitor&> auto&& visitChildren) {
    beginDict(dict.name);
    visitChildren(*this);
    endDict();
  }

 protected:
  // beginDict/endDict bracket a nested dict and are always invoked as a
  // pair by visit(MetadataDict, ...), with the dict's children visited in
  // between.
  virtual void beginDict(std::string_view name) = 0;
  virtual void endDict() = 0;

  virtual void visitUnsupported(std::string_view name) = 0;

  template <typename T>
  void visitValue(const MetadataField<T>& field, const T& /*value*/) {
    visitUnsupported(field.name);
  }

  void visitValue(
      const MetadataField<std::string>& field,
      const std::string& value) {
    visitValue(field, std::string_view{value});
  }

  virtual void visitValue(
      const MetadataField<int64_t>& field,
      int64_t value) = 0;

  virtual void visitValue(const MetadataField<double>& field, double value) = 0;

  virtual void visitValue(const MetadataField<bool>& field, bool value) = 0;

  virtual void visitValue(
      const MetadataField<std::string>& field,
      std::string_view value) = 0;

  virtual void visitValue(
      const MetadataField<std::vector<int64_t>>& field,
      const std::vector<int64_t>& value) = 0;

  virtual void visitValue(
      const MetadataField<std::vector<std::string>>& field,
      const std::vector<std::string>& value) = 0;

  virtual void visitValue(
      const MetadataField<RawJson>& field,
      const RawJson& value) = 0;

  virtual void visitValue(
      const MetadataField<uint64_t>& field,
      uint64_t value) = 0;

  virtual void visitValue(
      const MetadataField<InputShapes>& field,
      const InputShapes& value) = 0;
};

} // namespace libkineto
