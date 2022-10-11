#pragma once

#include <string>
#include <vector>
#include <c10/macros/Macros.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <ATen/record_function.h>
#include <ATen/core/ivalue.h>

namespace torch {

extern TORCH_API const std::string kParamCommsCallName;

class TORCH_API ParamCommsDebugInfo
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

#define RECORD_PARAM_COMMS(                                                    \
    seq,                                                                       \
    pg_ptr,                                                                    \
    rank,                                                                      \
    colName,                                                                   \
    inSize,                                                                    \
    outSize,                                                                   \
    dType,                                                                     \
    inSplitSizes,                                                              \
    outSplitSizes)                                                             \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(          \
      rank, colName, inSize, outSize, dType, inSplitSizes, outSplitSizes);     \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  std::initializer_list<const c10::IValue> paramList = {                       \
      c10::IValue(seq),                                                        \
      c10::IValue(pg_ptr),                                                   \
      rank,                                                                    \
      colName,                                                                 \
      inSplitSizes,                                                            \
      outSplitSizes};                                                          \
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);                     \
  RECORD_FUNCTION(torch::kParamCommsCallName, paramInputs);

#define RECORD_PARAM_COMMS_DATA(                                               \
    seq,                                                                       \
    pg_ptr,                                                                    \
    InputTensors,                                                               \
    OutputTensors,                                                              \
    rank,                                                                      \
    colName,                                                                   \
    inSize,                                                                    \
    outSize,                                                                   \
    dType,                                                                     \
    inSplitSizes,                                                              \
    outSplitSizes)                                                             \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(          \
      rank, colName, inSize, outSize, dType, inSplitSizes, outSplitSizes);     \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  std::initializer_list<const c10::IValue> paramList = {                       \
      c10::IValue(InputTensors),                                                \
      c10::IValue(seq),                                                        \
      c10::IValue(pg_ptr),                                                     \
      rank,                                                                    \
      colName,                                                                 \
      inSplitSizes,                                                            \
      outSplitSizes};                                                          \
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);                     \
  RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(                                         \
      torch::kParamCommsCallName,                                              \
      paramInputs,                                                             \
      std::vector<c10::IValue>(1, c10::IValue(OutputTensors)));
} // namespace torch
