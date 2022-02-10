// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <c10d/ParamCommsUtils.hpp>

namespace torch {

extern const std::string kParamCommsCallName = "record_param_comms";

ParamCommsDebugInfo::ParamCommsDebugInfo(
    int rank,
    std::string&& colName,
    int inSize,
    int outSize,
    at::ScalarType dType,
    std::vector<int64_t> inSplitSizes,
    std::vector<int64_t> outSplitSizes) :
      rank_(rank),
      columnName_(colName),
      inMessageSize_(inSize),
      outMessageSize_(outSize),
      dType_(dType),
      inputSplitSizes_(std::move(inSplitSizes)),
      outputSplitSizes_(std::move(outSplitSizes)) {}

} // namespace torch
