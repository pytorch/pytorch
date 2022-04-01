#pragma once

#include <memory>
#include <mutex>
#include <utility>

#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

// `reportBackendEventToActiveKinetoProfiler` reports times rather than counts,
// so `OpEventData` has to be able to store both cases.
union TimeStamp {
  int64_t us_; // Backend event.
  approx_time_t count_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TORCH_API OpEventData {
  OpEventData() = default;

  OpEventData(
      const uint64_t correlation_id,
      const uint64_t start_thread_id,
      const int64_t sequence_number,
      const uint64_t forward_thread_id,
      const at::RecordScope scope,
      const bool is_async,
      const int64_t debug_handle,
      const std::string name)
      : correlation_id_{correlation_id},
        start_thread_id_{start_thread_id},
        sequence_number_{sequence_number},
        forward_thread_id_{forward_thread_id},
        record_function_scope_{(uint8_t)scope},
        is_async_{is_async},
        debug_handle_{debug_handle},
        kineto_info_{kineto::kineto_ids()},
        name_{std::move(name)} {
    end_time_.count_ = std::numeric_limits<approx_time_t>::min();
  }

  OpEventData(
      const int64_t start_time,
      const int64_t end_time,
      const at::RecordScope scope,
      const int64_t debug_handle,
      const std::string name,
      const std::string backend)
      : correlation_id_{std::numeric_limits<uint64_t>::max()},
        start_thread_id_{at::RecordFunction::currentThreadId()},
        end_thread_id_{start_thread_id_},
        sequence_number_{-1},
        forward_thread_id_{start_thread_id_},
        record_function_scope_{(uint8_t)scope},
        is_async_{false},
        debug_handle_{debug_handle},
        kineto_info_{kineto::kineto_ids()},
        name_{std::move(name)},
        backend_{std::move(backend)} {
    start_time_.us_ = start_time;
    end_time_.us_ = end_time;
  }

  // POD members
  TimeStamp start_time_;
  TimeStamp end_time_;
  uint64_t correlation_id_;
  uint64_t start_thread_id_;
  uint64_t end_thread_id_;
  int64_t sequence_number_;
  uint64_t forward_thread_id_;
  uint8_t record_function_scope_;
  bool is_async_;
  int64_t debug_handle_;
  kineto::DeviceAndResource kineto_info_;

  std::string name_;

  // report_input_shapes
  std::vector<std::vector<int64_t>> shapes_;
  std::vector<std::string> dtypes_;

  // with_stack
  std::vector<std::string> stack_;

  // with_modules
  c10::optional<std::vector<std::string>> module_hierarchy_;

  // with_flops
  std::unordered_map<std::string, c10::IValue> extra_args_;

  // reportBackendEventToActiveKinetoProfiler
  c10::optional<std::string> backend_;

  // ProfilerState::KINETO_GPU_FALLBACK
  torch::profiler::impl::CUDAEventStub cuda_event_start_ = nullptr;
  torch::profiler::impl::CUDAEventStub cuda_event_end_ = nullptr;
};

class TORCH_API ThreadLocalSubqueue {
 public:
  template <class... Args>
  OpEventData* emplace_back(Args&&... args) {
    return data_.emplace_back(std::forward<Args>(args)...);
  }

 private:
  friend class RecordQueue;

  // See `containers.h` for block size benchmarks.
  static constexpr size_t BlockSize = 1024;
  AppendOnlyList<OpEventData, BlockSize> data_;
};

class TORCH_API RecordQueue {
 public:
  RecordQueue();
  ThreadLocalSubqueue* getSubqueue();
  std::deque<OpEventData> getRecords(std::function<time_t(approx_time_t)> time_converter);

 private:
  uint32_t id_;
  ska::flat_hash_map<uint64_t, std::unique_ptr<ThreadLocalSubqueue>> sub_queues_;
  std::mutex sub_queue_mutex_;
};

} // namespace impl
} // namespace profiler
} // namespace torch
