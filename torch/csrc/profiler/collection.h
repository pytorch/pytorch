#pragma once

#include <memory>
#include <mutex>
#include <utility>

#include <ATen/Context.h>
#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/variant.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

struct OpEvent {
  OpEvent() = default;
  OpEvent(
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
        name_{name} {}

  approx_time_t start_time_;
  approx_time_t end_time_{std::numeric_limits<approx_time_t>::min()};
  uint64_t correlation_id_;
  uint64_t start_thread_id_;
  uint64_t end_thread_id_;
  int64_t sequence_number_;
  uint64_t forward_thread_id_;
  uint8_t record_function_scope_;
  bool is_async_;
  int64_t debug_handle_;
  std::string name_;
};

struct Inputs {
  std::vector<std::vector<int64_t>> shapes_;
  std::vector<std::string> dtypes_;
};

struct FallbackPair {
  CUDAEventStub cuda_event_start_ = nullptr;
  CUDAEventStub cuda_event_end_ = nullptr;
};

struct BackendEvent {
  int64_t start_time_us_;
  int64_t end_time_us_;
  uint8_t record_function_scope_;
  int64_t debug_handle_;
  std::string name_;
  std::string backend_;
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct Result {
  std::string name() const;
  uint64_t correlation_id() const;

  int64_t start_time_us_;
  int64_t end_time_us_;
  uint64_t start_tid_;
  kineto::DeviceAndResource kineto_info_;

  c10::variant<OpEvent, BackendEvent> event_;

  // OpEvent only.
  Inputs inputs_;
  std::vector<std::string> jit_stack_;
  std::vector<std::string> jit_modules_;
  std::unordered_map<std::string, c10::IValue> extra_args_;
  FallbackPair gpu_fallback_;
};

struct KinetoObserverContext : public at::ObserverContext {
  explicit KinetoObserverContext(OpEvent* event)
    : event_{event} {}

  OpEvent* event_;
  FallbackPair* fallback_ {nullptr};
};

constexpr int IO_ENCODER_DEFAULT_BLOCK_SIZE = 1024;

// InputOutputEncoder
// Stores each op_events' shapes and dtypes into a contiguous AppendOnlyList
// so that we no longer create vectors for shapes and dtypes on every op.
// Those vectors can be created during post-processing.
class InputOutputEncoder final {
 public:
  void push(const std::vector<c10::IValue>& values);

  // Used during post-processing to create vectors for shapes and dtype.
  auto getNextShapesAndDtypes();

  void clear();

 private:
  enum class Tag {
    Tensor = 0,
    UndefinedTensor,
    TensorListBegin, // TODO: generalize to other lists.
    Scalar,
    Other,
    TERMINATOR
  };

  struct TensorMetadata {
    void* ptr_;
    c10::ScalarType dtype_;
    uint32_t dim_;
  };

  void push(const at::Tensor& t);

  AppendOnlyList<Tag, IO_ENCODER_DEFAULT_BLOCK_SIZE> tags_;
  AppendOnlyList<TensorMetadata, IO_ENCODER_DEFAULT_BLOCK_SIZE> tensor_metadata_;
  AppendOnlyList<int64_t, IO_ENCODER_DEFAULT_BLOCK_SIZE> tensor_sizes_;
};


class TORCH_API ThreadLocalSubqueue {
 public:
  ThreadLocalSubqueue(const uint64_t tid, const ProfilerConfig& config);

  std::unique_ptr<KinetoObserverContext> begin_op(const at::RecordFunction& fn, uint64_t correlation_id);

  template <class... Args>
  void emplace_backend_event(Args&&... args) {
    backend_events_.emplace_back(std::forward<Args>(args)...);
  }

  uint64_t tid() const {
    return tid_;
  }

  const kineto::DeviceAndResource& kineto_info() const {
    return kineto_info_;
  }

 private:
  uint64_t tid_;
  ProfilerConfig config_;
  kineto::DeviceAndResource kineto_info_;

  friend class RecordQueue;
  // See `containers.h` for block size benchmarks.
  static constexpr size_t BlockSize = 512;
  AppendOnlyList<OpEvent, BlockSize> op_events_;

  // report_input_shapes
  InputOutputEncoder inputs_outputs_;

  // with_stack
  AppendOnlyList<std::vector<std::string>, BlockSize> jit_stack_;

  // with_modules
  AppendOnlyList<std::vector<std::string>, BlockSize> jit_modules_;

  // with_flops
  AppendOnlyList<std::unordered_map<std::string, c10::IValue>, BlockSize> extra_args_;

  // ProfilerState::KINETO_GPU_FALLBACK
  AppendOnlyList<FallbackPair, BlockSize> gpu_fallback_;

  // reportBackendEventToActiveKinetoProfiler
  AppendOnlyList<BackendEvent, BlockSize> backend_events_;
};

class TORCH_API RecordQueue {
 public:
  explicit RecordQueue(const ProfilerConfig& config);

  ThreadLocalSubqueue* getSubqueue();

  // NB: This is a destructive operation.
  std::deque<Result> getRecords(std::function<time_t(approx_time_t)> time_converter);

 private:
  uint32_t id_;
  ProfilerConfig config_;
  ska::flat_hash_map<uint64_t, std::unique_ptr<ThreadLocalSubqueue>> sub_queues_;
  std::mutex sub_queue_mutex_;
};

} // namespace impl
} // namespace profiler
} // namespace torch
