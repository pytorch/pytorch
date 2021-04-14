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
    std::vector<int64_t>&& inSplitSizes,
    std::vector<int64_t>&& outSplitSizes);

  ~ParamCommsDebugInfo() override = default;

  [[nodiscard]] int getRank() const {
    return rank_;
  }

  [[nodiscard]] const std::string getColumnName() const {
    return columnName_;
  }

  [[nodiscard]] int getInMessageSize() const {
    return inMessageSize_;
  }

  [[nodiscard]] int getOutMessageSize() const {
    return outMessageSize_;
  }

  [[nodiscard]] at::ScalarType getDType() const {
    return dType_;
  }

  [[nodiscard]] const std::vector<int64_t>& getInputSplitSizes() const {
    return inputSplitSizes_;
  }

  [[nodiscard]] const std::vector<int64_t>& getOutputSplitSizes() const {
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

// TODO(jchae): handle non empty in/out split sizes
#define RECORD_PARAM_COMMS(rank, colName, inSize, outSize, dType, inSplitSizes, outSplitSizes) \
  std::vector<int64_t> iss; \
  std::vector<int64_t> oss; \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>( \
    rank, \
    colName, \
    inSize, \
    outSize, \
    dType, \
    std::move(iss), \
    std::move(oss)); \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  RECORD_FUNCTION(torch::kParamCommsCallName, std::vector<c10::IValue>());

} // namespace torch
