#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>
#include <c10/macros/Macros.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <string>
#include <vector>

namespace torch {

class TORCH_API ParamCommsDebugInfo : public c10::DebugInfoBase {
 public:
  ParamCommsDebugInfo() = default;
  ParamCommsDebugInfo(
      std::tuple<std::string, std::string> pgName,
      int rank,
      std::string&& collName,
      int inNelems,
      int outNelems,
      at::ScalarType dType,
      std::vector<int64_t> inSplitSizes,
      std::vector<int64_t> outSplitSizes,
      int globalRankStart,
      int globalRankStride,
      int worldSize);

  ~ParamCommsDebugInfo() override = default;

  const std::string getProcessGroupName() const {
    return std::get<0>(pgName_);
  }

  const std::string getProcessGroupDesc() const {
    return std::get<1>(pgName_);
  }

  int getRank() const {
    return rank_;
  }

  int getWorldSize() const {
    return worldSize_;
  }

  int getGlobalRankStart() const {
    return globalRankStart_;
  }

  int getGlobalRankStride() const {
    return globalRankStride_;
  }

  const std::string getCollectiveName() const {
    return collectiveName_;
  }

  int getInMessageNelems() const {
    return inMessageNelems_;
  }

  int getOutMessageNelems() const {
    return outMessageNelems_;
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

  const std::vector<int64_t>& getGroupRanks() const {
    return groupRanks_;
  }

 private:
  std::tuple<std::string, std::string> pgName_; // <group_name, group_desc>
  int rank_{};
  int worldSize_{};
  std::string collectiveName_;
  int inMessageNelems_{};
  int outMessageNelems_{};
  at::ScalarType dType_ = at::kByte;
  std::vector<int64_t> inputSplitSizes_;
  std::vector<int64_t> outputSplitSizes_;
  int globalRankStart_{};
  int globalRankStride_{};
  std::vector<int64_t> groupRanks_{};
};

#define RECORD_PARAM_COMMS(                                                    \
    seq,                                                                       \
    pgName,                                                                    \
    rank,                                                                      \
    collName,                                                                  \
    inNelems,                                                                  \
    outNelems,                                                                 \
    dType,                                                                     \
    inSplitSizes,                                                              \
    outSplitSizes,                                                             \
    globalRankStart,                                                           \
    globalRankStride,                                                          \
    worldSize)                                                                 \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(          \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inNelems,                                                                \
      outNelems,                                                               \
      dType,                                                                   \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize);                                                              \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  std::initializer_list<const c10::IValue> paramList = {                       \
      c10::IValue(seq),                                                        \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize};                                                              \
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);                     \
  RECORD_FUNCTION(at::kParamCommsCallName, paramInputs);

#define RECORD_PARAM_COMMS_DATA(                                               \
    seq,                                                                       \
    pgName,                                                                    \
    InputTensors,                                                              \
    OutputTensors,                                                             \
    rank,                                                                      \
    collName,                                                                  \
    inNelems,                                                                  \
    outNelems,                                                                 \
    dType,                                                                     \
    inSplitSizes,                                                              \
    outSplitSizes,                                                             \
    globalRankStart,                                                           \
    globalRankStride,                                                          \
    worldSize)                                                                 \
  auto paramCommsInfo = std::make_shared<torch::ParamCommsDebugInfo>(          \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inNelems,                                                                \
      outNelems,                                                               \
      dType,                                                                   \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize);                                                              \
  c10::DebugInfoGuard g(c10::DebugInfoKind::PARAM_COMMS_INFO, paramCommsInfo); \
  std::initializer_list<const c10::IValue> paramList = {                       \
      c10::IValue(InputTensors),                                               \
      c10::IValue(seq),                                                        \
      pgName,                                                                  \
      rank,                                                                    \
      collName,                                                                \
      inSplitSizes,                                                            \
      outSplitSizes,                                                           \
      globalRankStart,                                                         \
      globalRankStride,                                                        \
      worldSize};                                                              \
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);                     \
  RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(                                         \
      at::kParamCommsCallName,                                                 \
      paramInputs,                                                             \
      std::vector<c10::IValue>(1, c10::IValue(OutputTensors)));
} // namespace torch
