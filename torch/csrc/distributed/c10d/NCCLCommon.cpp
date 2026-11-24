#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/NCCLCommon.hpp>

#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>

namespace c10d {

using FlightRecorderCUDA = FlightRecorder<at::cuda::CUDAEvent>;

std::function<nccl_comm_dump_map_t()>& get_nccl_comm_dump_map_fn() {
  static std::function<nccl_comm_dump_map_t()> fn;
  return fn;
}

static nccl_comm_dump_map_t getNCCLCommDumpMap() {
  auto& fn = get_nccl_comm_dump_map_fn();
  if (!fn) {
    return {};
  }
  return fn();
}

void reset_nccl_trace() {
  FlightRecorderCUDA::get()->reset_all();
}

std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto ncclDumpMap = getNCCLCommDumpMap();
#if defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  for (const auto& [ncclUniqueIDStr, dump] : ncclDumpMap) {
    printNcclCommProxyTrace("Received dump signal " + ncclUniqueIDStr, dump);
  }
#endif // defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  return FlightRecorderCUDA::get()->dump(
      ncclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

std::string dump_nccl_trace_json(bool includeCollectives, bool onlyActive) {
  auto ncclDumpMap = getNCCLCommDumpMap();
  return FlightRecorderCUDA::get()->dump_json(
      ncclDumpMap, includeCollectives, onlyActive);
}

std::optional<std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper() {
  static std::optional<
      std::function<void(std::function<void(const std::string&)>)>>
      dumper(std::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

/* Implementation of TensorShelf class */

void TensorShelf::stash(std::vector<at::Tensor>& tensors) {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.insert(tVector_.end(), tensors.begin(), tensors.end());
}

void TensorShelf::stash(TensorShelf& other) {
  std::vector<at::Tensor>& otherVec = other.get();
  this->stash(otherVec);
}

void TensorShelf::unstash() {
  this->clear();
}

bool TensorShelf::empty() {
  std::lock_guard<std::mutex> lock(mutex_);
  return tVector_.empty();
}

void TensorShelf::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.clear();
}

std::vector<at::Tensor>& TensorShelf::get() {
  return tVector_;
}

NCCLOptions::NCCLOptions(bool is_high_priority_stream)
    : Backend::Options(NCCL_BACKEND_NAME, kProcessGroupNCCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

} // namespace c10d

#endif // USE_C10D_NCCL
