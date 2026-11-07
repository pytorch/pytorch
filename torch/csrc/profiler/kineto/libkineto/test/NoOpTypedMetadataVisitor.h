/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "include/TypedMetadata.h"

namespace libkineto {

class NoOpTypedMetadataVisitor : public ITypedMetadataVisitor {
 protected:
  void beginDict(std::string_view) override {}
  void endDict() override {}
  void visitUnsupported(std::string_view) override {}

  void visitValue(const MetadataField<int64_t>&, int64_t) override {}
  void visitValue(const MetadataField<uint64_t>&, uint64_t) override {}
  void visitValue(const MetadataField<double>&, double) override {}
  void visitValue(const MetadataField<bool>&, bool) override {}
  void visitValue(
      const MetadataField<std::string>&,
      std::string_view) override {}
  void visitValue(
      const MetadataField<std::vector<int64_t>>&,
      const std::vector<int64_t>&) override {}
  void visitValue(
      const MetadataField<std::vector<std::string>>&,
      const std::vector<std::string>&) override {}
  void visitValue(const MetadataField<RawJson>&, const RawJson&) override {}
  void visitValue(
      const MetadataField<InputShapes>&,
      const InputShapes&) override {}
};

} // namespace libkineto
