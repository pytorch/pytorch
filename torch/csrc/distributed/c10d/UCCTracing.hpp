#pragma once

#ifdef USE_C10D_UCC

#include <torch/csrc/distributed/c10d/UCCUtils.hpp>

namespace c10d {

#define RECORD_COMMS_TRACE(                                                    \
    _comms_tracer, _work, _opType, _rank, _comm_size, _inTensors, _outTensors) \
  do {                                                                         \
    if (torch_ucc_config.enable_comms_logger) {                                \
      _comms_tracer->recordComms(                                              \
          opTypeToString(_opType),                                             \
          (uintptr_t)_work.get(),                                              \
          _rank,                                                               \
          _comm_size,                                                          \
          _inTensors,                                                          \
          _outTensors);                                                        \
    }                                                                          \
  } while (0)

// interfaces to collect communication traces
class TORCH_API CommTraceLogger : public torch::CustomClassHolder {
 private:
  std::vector<std::string> comms_trace_;
  std::vector<std::string> curBlocks_; /* unused */
  std::vector<int64_t> curOutSplitSizes_;
  std::vector<int64_t> curInSplitSizes_;
  int curRoot_ = -1;
  unsigned long seqnum = 0;

 public:
  void setCurBlock(const std::string& name); /* unused */
  void popBlock(); /* unused */
  // record root info if applicable, e.g., broadcast, gather, scatter
  void recordOptionalInfo(int root = -1);
  // record input/output splits of Alltoallv
  void recordOptionalInfo(
      const std::vector<int64_t>& outputSplitSizes = {},
      const std::vector<int64_t>& inputSplitSizes = {});
  // record essential comms information
  void recordComms(
      const std::string& collName,
      const uintptr_t workReq = 0,
      const int rank = -1,
      const int world_size = -1,
      const std::vector<at::Tensor>& inputTensors = {},
      const std::vector<at::Tensor>& outputTensor = {});
  // return collected comms traces
  std::vector<std::string>& getCommsTrace() {
    return comms_trace_;
  }
};

} // namespace c10d

#endif // USE_C10D_UCC
