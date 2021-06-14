#pragma once

#include <string>
#include <vector>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <ATen/core/ivalue.h>

namespace torch {

extern const std::string kParamCommsCallName;

class ParamCommsDebugInfo
    : public c10::DebugInfoBase {

 public:
  ParamCommsDebugInfo() = default;
  ParamCommsDebugInfo(
    int rank,
    std::string&& colName,
    int inSize,
    int outSize,
    at::ScalarType dType,
    std::vector<int64_t> inSplitSizes,
    std::vector<int64_t> outSplitSizes);

  ~ParamCommsDebugInfo() override = default;

  int getRank() const {
    return rank_;
  }

  const std::string getColumnName() const {
    return columnName_;
  }

  int getInMessageSize() const {
    return inMessageSize_;
  }

  int getOutMessageSize() const {
    return outMessageSize_;
  }

  at::ScalarType getDType() const {
    return dType_;
  }

  const std::vector<int64_t>& getInputSplitSizes() const {
    return inputSplitSizes_;
  }

  const std::vector<int64_t>& getOutputSplitSizes() const {
    return outputSplitSizes_;
  }

 private:
  int rank_{};
  std::string columnName_;
  int inMessageSize_{};
  int outMessageSize_{};
  at::ScalarType dType_ = at::kByte;
  std::vector<int64_t> inputSplitSizes_;
  std::vector<int64_t> outputSplitSizes_;
};


#define RECORD_PARAM_COMMS(rank, colName, inSize, outSize, dType, inSplitSizes, outSplitSizes) \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>( \
    rank, \
    colName, \
    inSize, \
    outSize, \
    dType, \
    inSplitSizes, \
    outSplitSizes); \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  RECORD_FUNCTION(torch::kParamCommsCallName, std::vector<c10::IValue>());

} // namespace torch
