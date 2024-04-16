// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>

namespace torch {

ParamCommsDebugInfo::ParamCommsDebugInfo(
    int pgId,
    int rank,
    std::string&& colName,
    int inNelems,
    int outNelems,
    at::ScalarType dType,
    std::vector<int64_t> inSplitSizes,
    std::vector<int64_t> outSplitSizes,
    int globalRankStart,
    int globalRankStride,
    int worldSize)
    : pgId_(pgId),
      rank_(rank),
      worldSize_(worldSize),
      columnName_(colName),
      inMessageNelems_(inNelems),
      outMessageNelems_(outNelems),
      dType_(dType),
      inputSplitSizes_(std::move(inSplitSizes)),
      outputSplitSizes_(std::move(outSplitSizes)),
      globalRankStart_(globalRankStart),
      globalRankStride_(globalRankStride) {
  if (globalRankStride > 0) {
    for (int i = 0; i < worldSize; i++) {
      groupRanks_.push_back(globalRankStart + i * globalRankStride);
    }
  }
}

} // namespace torch
