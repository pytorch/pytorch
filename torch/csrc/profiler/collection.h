#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/strong_type.h>
#include <c10/util/variant.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/data_flow.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/orchestration/python_tracer.h>
#include <torch/csrc/profiler/perf.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>
#include <torch/csrc/utils/python_stub.h>

namespace torch {
namespace profiler {
namespace impl {

enum class EventType : uint8_t {
  TorchOp = 0,
  Backend,
  Vulkan,
  Allocation,
  OutOfMemory,
  PyCall,
  PyCCall,
  Kineto
};

// ============================================================================
// == Value (Tensor, Scalar) summary ==========================================
// ============================================================================
struct TORCH_API RawTensorMetadataBase {
  RawTensorMetadataBase() = default;
  explicit RawTensorMetadataBase(const at::Tensor& t);

  StorageImplData data_;
  c10::ScalarType dtype_;
  c10::Layout layout_;
  uint32_t dim_;
};

// Collected during profiling.
struct TORCH_API RawTensorMetadata : RawTensorMetadataBase {
  RawTensorMetadata() = default;
  RawTensorMetadata(const RawTensorMetadata&) = default;
  explicit RawTensorMetadata(const at::Tensor& t);

  // Wrap `weak_self_` in `c10::optional` and split device into components to
  // keep struct default constructable. (which the std::array initializer needs)
  c10::optional<WeakTensor> weak_self_;
  c10::DeviceType device_type_;
  c10::DeviceIndex device_index_;
};

// Used during post processing.
struct TORCH_API TensorMetadata : public RawTensorMetadataBase {
  TensorMetadata(
      const RawTensorMetadata& r,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides);

  TensorImplAddress impl() const {
    return weak_self_.get();
  }

  WeakTensor weak_self_;
  c10::Device device_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;

  // Set during `calculateUniqueTensorIDs`.
  c10::optional<TensorID> id_;
  c10::optional<AllocationID> allocation_id_;
};

using op_input_t = c10::variant<
    TensorMetadata,
    std::vector<TensorMetadata>,
    c10::IValue,
    c10::nullopt_t>;

// ============================================================================
// == ExtraFields =============================================================
// ============================================================================
template <EventType>
struct ExtraFields;

struct Result;

struct TorchOpBasicFields {
  int64_t sequence_number_{0};
  uint64_t forward_tid_{0};
  at::RecordScope scope_{};
  bool is_async_{false};
  int64_t debug_handle_{0};
  std::string name_;

  // Set in the exit callback.
  uint64_t end_tid_{0};
};

using jit_stack_t = std::vector<std::string>;
using jit_modules_t = std::vector<std::string>;
using extra_args_t = std::unordered_map<std::string, c10::IValue>;

struct FallbackPair {
  ProfilerEventStub cuda_event_start_ = nullptr;
  ProfilerEventStub cuda_event_end_ = nullptr;
};

template <>
struct ExtraFields<EventType::TorchOp> : TorchOpBasicFields {
  ExtraFields(
      TorchOpBasicFields&& f,
      uint64_t correlation_id,
      time_t end_time_ns,
      std::vector<op_input_t>&& inputs,
      std::vector<op_input_t>&& concrete_inputs,
      jit_stack_t&& jit_stack,
      jit_modules_t&& jit_modules,
      extra_args_t&& extra_args,
      FallbackPair&& gpu_fallback,
      bool allow_tf32_cublas,
      std::unique_ptr<perf_counters_t>&& perf_event_counters)
      : TorchOpBasicFields(std::move(f)),
        correlation_id_{correlation_id},
        end_time_ns_{end_time_ns},
        inputs_{std::move(inputs)},
        concrete_inputs_{std::move(concrete_inputs)},
        jit_stack_{std::move(jit_stack)},
        jit_modules_{std::move(jit_modules)},
        extra_args_{std::move(extra_args)},
        gpu_fallback_{std::move(gpu_fallback)},
        allow_tf32_cublas_{allow_tf32_cublas},
        perf_event_counters_{std::move(perf_event_counters)} {}
  uint64_t correlation_id_;
  time_t end_time_ns_;
  std::vector<op_input_t> inputs_;
  std::vector<op_input_t> concrete_inputs_;
  jit_stack_t jit_stack_;
  jit_modules_t jit_modules_;
  extra_args_t extra_args_;
  FallbackPair gpu_fallback_;
  bool allow_tf32_cublas_;
  std::unique_ptr<perf_counters_t> perf_event_counters_;
};

template <>
struct ExtraFields<EventType::Backend> {
  int64_t start_time_us_;
  int64_t end_time_us_;
  int64_t debug_handle_;
  at::RecordScope scope_;
  std::string name_;
  std::string backend_;
  jit_stack_t jit_stack_;
  jit_modules_t jit_modules_;
};

template <>
struct ExtraFields<EventType::Vulkan> {
  using raw_event_t = std::pair<approx_time_t, vulkan_id_t>;
  std::string name_;
  int64_t duration_ns_{0};
  // While building the event tree, we want to report a vulkan event's duration
  // as 0 so that its end time doesn't exceed that of its parent cpu op
  bool in_tree_building_{false};
};

struct RawAllocation {
  torch::profiler::impl::approx_time_t start_time_;
  void* ptr_;
  int64_t alloc_size_;
  size_t total_allocated_;
  size_t total_reserved_;
  c10::DeviceType device_type_;
  c10::DeviceIndex device_index_;
};

// For performance.
static_assert(c10::is_pod_v<RawAllocation>, "Non-POD member of RawAllocation.");

template <>
struct ExtraFields<EventType::Allocation> : RawAllocation {
  ExtraFields(const RawAllocation& allocation) : RawAllocation(allocation) {}

  c10::Device device() const {
    return {device_type_, device_index_};
  }

  c10::optional<TensorID> id_;
  c10::optional<AllocationID> allocation_id_;
};

template <>
struct ExtraFields<EventType::OutOfMemory> {
  torch::profiler::impl::approx_time_t start_time_;
  int64_t alloc_size_;
  size_t total_allocated_;
  size_t total_reserved_;
  c10::DeviceType device_type_;
  c10::DeviceIndex device_index_;
};

// For performance.
static_assert(
    c10::is_pod_v<ExtraFields<EventType::OutOfMemory>>,
    "Non-POD member of ExtraFields<EventType::OutOfMemory>.");

struct PyFrameState {
  int line_no_;
  at::StringView filename_;
  at::StringView funcname_;
};

template <typename T, typename Tag>
using strong_t = strong::
    type<T, Tag, strong::regular, strong::convertible_to<T>, strong::hashable>;

using PyModuleSelf = strong_t<PyObject*, struct PyModuleSelf_>;
using PyModuleCls = strong_t<PyObject*, struct PyModuleCls_>;
using PyMethod = strong_t</*PyMethodDef*/ void*, struct PyMethod_>;
using PyOptimizerSelf = strong_t<PyObject*, struct PyOptSelf_>;
using PyOptimizerCls = strong_t<PyObject*, struct PyOptimizer_>;

struct NNModuleInfo {
  struct ParameterInfo {
    std::string name_;
    TensorMetadata metadata_;
    c10::optional<TensorMetadata> grad_metadata_;
  };

  PyModuleSelf self_;
  PyModuleCls cls_;
  at::StringView cls_name_;

  std::vector<ParameterInfo> parameters_;
  // Indicates that `self_` is the kth instance of `cls_` observed.
  size_t id_{std::numeric_limits<size_t>::max()};
};

struct OptimizerInfo {
  struct ParameterInfo {
    TensorMetadata metadata_;
    c10::optional<TensorMetadata> grad_metadata_;
    std::vector<std::pair<std::string, TensorMetadata>> state_;
  };

  PyOptimizerSelf self_;
  PyOptimizerCls cls_;
  at::StringView cls_name_;

  std::vector<ParameterInfo> parameters_;
};

struct PyExtraFieldsBase {
  PyExtraFieldsBase(time_t end_time_ns, size_t python_tid, PyFrameState caller)
      : end_time_ns_{end_time_ns},
        python_tid_{python_tid},
        caller_{std::move(caller)} {}

  time_t end_time_ns_;
  size_t python_tid_;
  PyFrameState caller_;

  // kth python event observed. (Used by TensorBoard)
  size_t id_{std::numeric_limits<size_t>::max()};
};

template <>
struct ExtraFields<EventType::PyCall> : public PyExtraFieldsBase {
  struct args_t {
    PyFrameState frame_state_;
    c10::optional<NNModuleInfo> module_info_;
    c10::optional<OptimizerInfo> optimizer_info_;
  };

  ExtraFields(
      time_t end_time_ns,
      size_t python_tid,
      PyFrameState caller,
      args_t args)
      : PyExtraFieldsBase(end_time_ns, python_tid, caller),
        callsite_{args.frame_state_},
        module_{args.module_info_},
        optimizer_{args.optimizer_info_} {}

  PyFrameState callsite_;
  c10::optional<NNModuleInfo> module_;
  c10::optional<OptimizerInfo> optimizer_;
};

template <>
struct ExtraFields<EventType::PyCCall> : public PyExtraFieldsBase {
  using args_t = at::StringView;

  ExtraFields(
      time_t end_time_ns,
      size_t python_tid,
      PyFrameState caller,
      args_t args)
      : PyExtraFieldsBase(end_time_ns, python_tid, caller),
        function_name_{std::move(args)} {}

  at::StringView function_name_;
};

template <>
struct ExtraFields<EventType::Kineto> {
  // Mirrors `libkineto::GenericTraceActivity::Flow`. This information is used
  // during post processing to properly embed Kineto events into the broader
  // profiler tree structure. End users are not generally expected to use these
  // fields directly, but they are available for debugging.
  struct Flow {
    uint32_t id{0};
    uint32_t type{0};
    uint32_t start{0};
  };

  std::string name_;
  int64_t duration_us_{0};
  uint64_t correlation_id_{0};
  libkineto::ActivityType activity_type_;
  Flow flow;
  std::weak_ptr<Result> linked_activity_{};
};

struct TORCH_API Result : public std::enable_shared_from_this<Result> {
  template <typename... Args>
  [[nodiscard]] static std::shared_ptr<Result> create(Args... args) {
    return std::shared_ptr<Result>(new Result(std::forward<Args>(args)...));
  }

  template <typename T>
  decltype(auto) visit(T&& visitor) {
    return c10::visit(std::forward<T>(visitor), extra_fields_);
  }

  template <typename T>
  decltype(auto) visit(T&& visitor) const {
    return c10::visit(std::forward<T>(visitor), extra_fields_);
  }

  template <typename T, typename Fn>
  void visit_if_base(Fn&& fn) const {
    visit([&](const auto& extra_fields) {
      using extra_fields_t = typename std::remove_cv<
          typename std::remove_reference<decltype(extra_fields)>::type>::type;

      if constexpr (std::is_base_of_v<T, extra_fields_t>) {
        fn(extra_fields);
      }
    });
  }

  EventType tag() const {
    return visit([](const auto& i) { return deduceTag(i); });
  }

  std::string name() const;
  libkineto::ActivityType kinetoType() const;
  uint64_t correlationID() const;
  int64_t endTimeNS() const;
  uint64_t endTID() const;
  c10::DeviceType deviceType() const;

  int64_t start_time_ns_;
  uint64_t start_tid_;
  kineto::DeviceAndResource kineto_info_;
  c10::variant<
      ExtraFields<EventType::TorchOp>,
      ExtraFields<EventType::Backend>,
      ExtraFields<EventType::Vulkan>,
      ExtraFields<EventType::Allocation>,
      ExtraFields<EventType::OutOfMemory>,
      ExtraFields<EventType::PyCall>,
      ExtraFields<EventType::PyCCall>,
      ExtraFields<EventType::Kineto>>
      extra_fields_;

  std::weak_ptr<Result> parent_;
  std::vector<std::shared_ptr<Result>> children_;
  bool finished_{false};

  const torch::profiler::impl::kineto::activity_t* kineto_activity_{nullptr};

 private:
  template <EventType E>
  Result(
      int64_t start_time_ns,
      uint64_t start_tid,
      kineto::DeviceAndResource kineto_info,
      ExtraFields<E>&& extra_fields)
      : start_time_ns_{start_time_ns},
        start_tid_{start_tid},
        kineto_info_{kineto_info},
        extra_fields_{std::move(extra_fields)} {}

  template <EventType E>
  static EventType deduceTag(const ExtraFields<E>&) {
    return E;
  }
};

struct KinetoObserverContext : public at::ObserverContext {
  struct Event {
    TorchOpBasicFields basic_fields_;
    approx_time_t start_time_;

    // Set in the exit callback.
    approx_time_t end_time_{std::numeric_limits<approx_time_t>::min()};

    bool allow_tf32_cublas_;
    std::unique_ptr<perf_counters_t> counters_;
  };

  explicit KinetoObserverContext(Event* event) : event_{event} {}

  Event* event_;
  FallbackPair* fallback_{nullptr};
};

constexpr int IO_ENCODER_DEFAULT_BLOCK_SIZE = 1024;

constexpr int SCALAR_LIST_LENGTH_LIMIT = 30;

// InputOutputEncoder
// Stores each op_events' shapes and dtypes, and concrete values into a
// contiguous AppendOnlyList so that we no longer create vectors for shapes
// and dtypes on every op. Those vectors can be created during
// post-processing.
// It splits the data into two categories: input shapes and concrete inputs.
class InputOutputEncoder final {
 public:
  void push(c10::ArrayRef<const c10::IValue> values);

  // Used during post-processing to unpack the encoded data.
  // Each method returns a "supplier" lambda which takes no arguments;
  // invoking the lambda once will return a list of args that represent
  // the inputs for one op.
  // The data is split into two streams: "input shapes" and "concrete inputs".
  // Note: "auto" only works because these are only used in collection.cpp,
  // where they are implemented.
  auto getInputShapeGenerator();
  auto getConcreteInputGenerator();

  bool isSupportedScalarList(const c10::IValue& list_candidate);

  void clear();

  enum class Tag {
    Tensor = 0,
    UndefinedTensor,
    TensorListBegin, // TODO: generalize to other lists.
    ScalarList,
    Scalar,
    Other,
    TERMINATOR
  };

  enum class IOType { Shapes, ConcreteInputs, None };

 private:
  void push(const at::Tensor& t);

  // Implementation detail for getInputShapeGenerator and
  // getConcreteInputGenerator
  auto getIValueGenerator(const IOType& io_type);

  AppendOnlyList<Tag, IO_ENCODER_DEFAULT_BLOCK_SIZE> tags_;
  AppendOnlyList<RawTensorMetadata, IO_ENCODER_DEFAULT_BLOCK_SIZE>
      tensor_metadata_;
  AppendOnlyList<int64_t, IO_ENCODER_DEFAULT_BLOCK_SIZE> tensor_sizes_strides_;
  AppendOnlyList<c10::IValue, IO_ENCODER_DEFAULT_BLOCK_SIZE> ivalues_;
};

using perf_profiler_t = torch::profiler::impl::linux_perf::PerfProfiler;

class TORCH_API ThreadLocalSubqueue {
 public:
  ThreadLocalSubqueue(const uint64_t tid, const ProfilerConfig& config);

  std::unique_ptr<KinetoObserverContext> begin_op(const at::RecordFunction& fn);

  template <class... Args>
  void emplace_backend_event(Args&&... args) {
    backend_events_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_vulkan_event(Args&&... args) {
    vulkan_events_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_allocation_event(Args&&... args) {
    allocations_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_ooms_event(Args&&... args) {
    ooms_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_py_call(Args&&... args) {
    py_calls_.emplace_back(std::forward<Args>(args)...);
  }

  uint64_t tid() const {
    return tid_;
  }

  const kineto::DeviceAndResource& kineto_info() const {
    return kineto_info_;
  }

  inline void disable_perf_profiler(perf_counters_t& counters) const {
    perf_profiler_->Disable(counters);
  }

 private:
  uint64_t tid_;
  ProfilerConfig config_;
  kineto::DeviceAndResource kineto_info_;
  std::unique_ptr<perf_profiler_t> perf_profiler_;

  friend class RecordQueue;
  // See `containers.h` for block size benchmarks.
  static constexpr size_t BlockSize = 512;

  struct TorchOpStorage {
    // NB: This is a destructive operation.
    void materialize(
        std::vector<std::shared_ptr<Result>>& out,
        const std::function<time_t(approx_time_t)> time_converter,
        const uint64_t tid,
        const kineto::DeviceAndResource& kineto_info);

    template <typename T, size_t ChunkSize>
    class EventBlock : public std::array<T, ChunkSize> {
     public:
      EventBlock();
      uint64_t correlation_id(const T* ptr) const;

     private:
      uint64_t id_start_;
    };

    using event_t = KinetoObserverContext::Event;
    class OpList : public AppendOnlyList<event_t, BlockSize, EventBlock> {
     public:
      template <class... Args>
      std::pair<event_t*, uint64_t> emplace_back(Args&&... args);
      static uint64_t correlationID(const OpList::Iterator& e);
    } op_events_;

    // report_input_shapes
    InputOutputEncoder inputs_outputs_;

    // with_stack (JIT)
    AppendOnlyList<jit_stack_t, BlockSize> jit_stack_;

    // with_modules
    AppendOnlyList<jit_modules_t, BlockSize> jit_modules_;

    // with_flops
    AppendOnlyList<extra_args_t, BlockSize> extra_args_;

    // ProfilerState::KINETO_GPU_FALLBACK
    AppendOnlyList<FallbackPair, BlockSize> gpu_fallback_;
  } torch_ops_;

  // reportBackendEventToActiveKinetoProfiler
  AppendOnlyList<ExtraFields<EventType::Backend>, BlockSize> backend_events_;

  // _reportVulkanEventToProfiler
  AppendOnlyList<ExtraFields<EventType::Vulkan>::raw_event_t, BlockSize>
      vulkan_events_;

  // reportMemoryUsage
  AppendOnlyList<RawAllocation, BlockSize> allocations_;

  // reportOOMs
  AppendOnlyList<ExtraFields<EventType::OutOfMemory>, BlockSize> ooms_;

  // with_stack (Python)
  AppendOnlyList<std::pair<python_tracer::TraceKey, approx_time_t>, BlockSize>
      py_calls_;
};

class TORCH_API RecordQueue {
 public:
  RecordQueue(const ProfilerConfig& config, std::set<ActivityType> activities);

  bool tracePython() const;
  ThreadLocalSubqueue* getSubqueue();
  void stop();

  // NB: This is a destructive operation.
  std::pair<
      std::vector<std::shared_ptr<Result>>,
      std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>>
  getRecords(
      std::function<time_t(approx_time_t)> time_converter,
      uint64_t start_time_us,
      uint64_t end_time_us);

 private:
  uint32_t id_;
  ProfilerConfig config_;
  std::set<ActivityType> activities_;
  ska::flat_hash_map<uint64_t, std::unique_ptr<ThreadLocalSubqueue>>
      sub_queues_;
  std::mutex sub_queue_mutex_;
  std::unique_ptr<python_tracer::PythonTracerBase> python_tracer_;
};

TORCH_API bool get_record_concrete_inputs_enabled();
TORCH_API void set_record_concrete_inputs_enabled_fn(std::function<bool()>);
TORCH_API void set_record_concrete_inputs_enabled_val(bool);

TORCH_API bool get_fwd_bwd_enabled();
TORCH_API void set_fwd_bwd_enabled_fn(std::function<bool()>);
TORCH_API void set_fwd_bwd_enabled_val(bool);

} // namespace impl
} // namespace profiler
} // namespace torch
