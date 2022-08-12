#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <processthreadsapi.h>
#else
#include <unistd.h>
#endif // _WIN32

#include <fmt/format.h>
#include <chrono>
#include <fstream>
#include <map>
#include <mutex>
#include <stack>
#include <stdexcept>
#include <vector>

#include <ATen/core/TensorBody.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/stack.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/execution_graph_observer.h>

using namespace at;

namespace torch {
namespace {

//******************************************************************************
// JSON output utility functions. To be merged with PyTorch profiler.
//******************************************************************************
template <typename T>
inline std::string vectorToString(const std::vector<T>& v) {
  return fmt::format("[{}]", fmt::join(v, ","));
}

inline std::string getValueType(
    const c10::IValue& val,
    const bool baseType = true,
    const size_t maxArrayLen = 100) {
  std::string type = val.tagKind();

  if (val.isTensor()) {
    // Add tensor element data type.
    type += fmt::format("({})", std::string(val.toTensor().dtype().name()));
  } else if (val.isTuple()) {
    const auto& val_container = val.toTupleRef().elements();
    std::vector<std::string> str_array;
    for (const auto& t : val_container) {
      str_array.emplace_back(getValueType(t, false));
    }
    type += vectorToString(str_array);
  } else if (val.isList()) {
    const auto& val_list = val.toList();
    std::vector<std::string> str_array;
    str_array.reserve(val_list.size());
    for (const auto j : c10::irange(val_list.size())) {
      str_array.push_back(getValueType(val_list.get(j), false));
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << val_list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    type += vectorToString(str_array);
  }
  return baseType ? fmt::format("\"{}\"", type) : type;
}

inline std::string getValueShape(
    const c10::IValue& val,
    const size_t maxArrayLen = 100) {
  if (val.isTensor()) {
    auto& tensor = val.toTensor();
    if (tensor.defined()) {
      return vectorToString(tensor.sizes().vec());
    }
  } else if (val.isTuple()) {
    const auto& val_container = val.toTupleRef().elements();
    std::vector<std::string> str_array;
    for (const auto& t : val_container) {
      str_array.push_back(getValueShape(t));
    }
    return vectorToString(str_array);
  } else if (val.isList()) {
    const auto& val_list = val.toList();
    std::vector<std::string> str_array;
    str_array.reserve(val_list.size());
    for (const auto j : c10::irange(val_list.size())) {
      str_array.push_back(getValueShape(val_list.get(j)));
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << val_list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    return vectorToString(str_array);
  }
  return "[]";
}

inline std::string getScalarValue(const c10::IValue& val) {
  if (val.isDouble()) {
    return std::to_string(val.toDouble());
  } else if (val.isInt()) {
    return std::to_string(val.toInt());
  } else if (val.isBool()) {
    return val.toBool() ? "true" : "false";
  } else if (val.isString()) {
    constexpr int maxStringLen = 500;
    return fmt::format("\"{}\"", val.toStringRef().substr(0, maxStringLen));
  } else if (val.isDevice()) {
    return fmt::format("\"{}\"", val.toDevice().str());
  }
  return fmt::format("\"<{}>\"", val.tagKind());
}

inline int32_t processId() {
#ifndef _WIN32
  return static_cast<int32_t>(getpid());
#else
  return static_cast<int32_t>(GetCurrentProcessId());
#endif
}

//******************************************************************************
// Main ExecutionGraphObserver implementation.
//******************************************************************************

static CallbackHandle handle_{INVALID_CALLBACK_HANDLE};

// ExecutionGraphObserver contains all the states of the observer. Some of them
// are shared between the enter and exit RecordFunction call backs, some data
// like the `op_stack` may be accessed across different threads. So we should be
// careful about data races. A global mutex `g_mutex` is used avoid these races
// at the cost of performance in large number of threads situations. We may
// optimize this further to thread local, fine-grained locking, or use thread
// safe containers.
struct ExecutionGraphObserver {
  using ID = size_t;

  // Mapping of each thread to its own operator stack
  std::map<size_t, std::stack<ID>> op_stack{};
  // Uses the underlying TensorImpl object pointer as the key and map to its
  // unique id.
  std::map<void*, ID> object_id{};
  // Observer run state.
  enum class RunState { uninitialized, disabled, enabled };

 private:
  // Must use accessors to change this so that we can keep the
  // RecordFunction callback in sync with the state.
  RunState state{RunState::uninitialized};

 public:
  // Mutex for multithreaded access to the shared containers.
  std::mutex g_mutex{};
  // Stream to write output JSON.
  std::ofstream out{};

  // Full path to the output file.
  std::string file_name{};
  CallbackHandle op_observer_handle{};
  int32_t pid{-1};
  std::string record_time{};

  ExecutionGraphObserver() = default;

  // Returns a new unique ID.
  ID getNewID() {
    return id_++;
  }

  RunState getState() const {
    return state;
  }

  void setState(RunState newState) {
    if (state == RunState::uninitialized ||
        callbackShouldBeEnabled(state) != callbackShouldBeEnabled(newState)) {
      if (callbackShouldBeEnabled(newState)) {
        reenableCallback(handle_);
      } else {
        disableCallback(handle_);
      }
    }
    state = newState;
  }

 private:
  static bool callbackShouldBeEnabled(RunState run_state) {
    return run_state == ExecutionGraphObserver::RunState::enabled;
  }

  // All tensors and operators have an unique id assigned. Increment id for each
  // new tensor or operator node.
  // 0 -> unintialized
  // 1 -> root ID
  // 2 ... -> regular node ID
  std::atomic<ID> id_{2};
};

// Using a singleton pattern here to avoid global static variable initialization
// race.
ExecutionGraphObserver& observer() {
  static ExecutionGraphObserver _observer{};
  return _observer;
}

// Uninitialized node has id = 0
const ExecutionGraphObserver::ID uninitialized_id{0};
// Root node has id = 1
const ExecutionGraphObserver::ID root_id{1};

struct FunctionCallContext : public ObserverContext {
  std::string name;
  ExecutionGraphObserver::ID op_id{uninitialized_id};
  ExecutionGraphObserver::ID parent_id{uninitialized_id};
  ExecutionGraphObserver::ID fw_parent_id{uninitialized_id};
  std::vector<std::string> input_types;
  std::vector<std::string> input_shapes;
  std::vector<std::string> input_values;
};

// Opens the json file to write the execution graph.
std::ofstream openOutputFile(const std::string& name) {
  std::ofstream stream;
  stream.open(name, std::ofstream::out | std::ofstream::trunc);
  if (!stream) {
    LOG(ERROR) << "Failed to open '" << name << "'";
  } else {
    VLOG(1) << "Writing PyTorch execution graph to: " << name;
  }
  return stream;
}

void writeJsonNode(
    std::ofstream& out,
    const std::string& name,
    const uint64_t id,
    const uint64_t rf_id,
    const uint64_t parent,
    const uint64_t fw_parent,
    const int64_t seq_id,
    const uint64_t scope,
    const uint64_t tid,
    const uint64_t fw_tid,
    const std::string& inputs = "[]",
    const std::string& input_shapes = "[]",
    const std::string& input_types = "[]",
    const std::string& outputs = "[]",
    const std::string& output_shapes = "[]",
    const std::string& output_types = "[]",
    const std::string& operator_schema = "") {
  out << fmt::format(
      R"JSON(
    {{
      "name": "{}", "id": {}, "rf_id": {}, "parent": {}, "fw_parent": {}, "seq_id": {}, "scope": {}, "tid": {}, "fw_tid": {}, "op_schema": "{}",
      "inputs": {}, "input_shapes": {}, "input_types": {},
      "outputs": {}, "output_shapes": {}, "output_types": {}
    }})JSON",
      name,
      id,
      rf_id,
      parent,
      fw_parent,
      seq_id,
      scope,
      tid,
      fw_tid,
      operator_schema,
      inputs,
      input_shapes,
      input_types,
      outputs,
      output_shapes,
      output_types);
}

inline std::string timeString(const std::time_t timepoint) {
  std::ostringstream oss;
  oss << std::put_time(std::localtime(&timepoint), "%Y-%m-%d %X");
  return oss.str();
}

bool initExecutionGraphStart(ExecutionGraphObserver& ob) {
  ob.out = openOutputFile(ob.file_name);
  // If somehow the output stream failed to open, finish observer here.
  if (!ob.out) {
    LOG(WARNING) << "Failed to open output file: " << ob.file_name;
    return false;
  }

  // Wall clock time for the first op collection time.
  const auto current_time = std::chrono::system_clock::now();
  ob.record_time =
      timeString(std::chrono::system_clock::to_time_t(current_time));
  // Start timestamp using steady_clock for measurement.
  const auto timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();

  ob.out << fmt::format(
      R"JSON({{
  "schema": "1.0.0", "pid": {}, "time": "{}", "start_ts": {},
  "nodes": [)JSON",
      ob.pid,
      ob.record_time,
      timestamp);
  return true;
}

// Write out Execution Graph to file
void finalizeExecutionGraphOutput(ExecutionGraphObserver& ob) {
  writeJsonNode(
      ob.out,
      "[pytorch|profiler|execution_graph|process]",
      root_id,
      0, // rf_id
      root_id, // parent is self
      0, // fw_parent
      -1, // seq_id
      static_cast<std::underlying_type_t<RecordScope>>(RecordScope::USER_SCOPE),
      0, // tid
      0); // fw_tid

  // Finish timestamp using steady_clock for measurement.
  const auto timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  ob.out << fmt::format(
      R"JSON(
  ],
  "finish_ts": {}
}})JSON",
      timestamp);

  ob.out.close();
  VLOG(1) << "PyTorch execution graph is written to file: " << ob.file_name;
}

inline ExecutionGraphObserver::ID getObjectID(
    ExecutionGraphObserver& ob,
    void* t) {
  auto iter = ob.object_id.find(t);
  if (iter == ob.object_id.end()) {
    ExecutionGraphObserver::ID object_id = ob.getNewID();
    ob.object_id[t] = object_id;
    return object_id;
  }

  return iter->second;
}

inline std::string convertIValue(
    ExecutionGraphObserver& ob,
    const c10::IValue& val,
    const size_t maxArrayLen = 100) {
  if (val.isTensor()) {
    const auto t = val.toTensor().unsafeGetTensorImpl();
    ExecutionGraphObserver::ID tensor_id = getObjectID(ob, t);
    ExecutionGraphObserver::ID storage_id = 0;
    size_t offset = 0;
    size_t numel = 0;
    size_t itemsize = 0;
    if (t->has_storage()) {
      storage_id = getObjectID(ob, t->storage().data());
      offset = t->storage_offset();
      numel = t->numel();
      itemsize = t->itemsize();
    }
    return vectorToString(
        std::vector<size_t>{tensor_id, storage_id, offset, numel, itemsize});
  } else if (val.isTuple()) {
    std::vector<std::string> str_array;
    const auto& val_tuple = val.toTupleRef().elements();
    for (const auto j : c10::irange(val_tuple.size())) {
      str_array.push_back(convertIValue(ob, val_tuple[j]));
    }
    return vectorToString(str_array);
  } else if (val.isList()) {
    const auto& val_list = val.toList();
    std::vector<std::string> str_array;
    str_array.reserve(val_list.size());
    for (const auto j : c10::irange(val_list.size())) {
      str_array.push_back(convertIValue(ob, val_list.get(j)));
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << val_list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    return vectorToString(str_array);
  } else {
    return getScalarValue(val);
  }
}

inline void appendValueInfo(
    ExecutionGraphObserver& ob,
    const c10::IValue& val,
    std::vector<std::string>& values,
    std::vector<std::string>& types,
    std::vector<std::string>& shapes) {
  values.push_back(convertIValue(ob, val));
  types.push_back(getValueType(val));
  shapes.push_back(getValueShape(val));
}

void recordOperatorStart(
    ExecutionGraphObserver& ob,
    FunctionCallContext& fc,
    const RecordFunction& fn) {
  auto tid = fn.threadId();

  try {
    const std::lock_guard<std::mutex> lock(ob.g_mutex);

    // if current thread stack is empty, push the root node to the stack first
    if (ob.op_stack[tid].empty()) {
      auto thread_node_id = ob.getNewID();
      ob.op_stack[tid].push(thread_node_id);
      writeJsonNode(
          ob.out,
          "[pytorch|profiler|execution_graph|thread]",
          thread_node_id,
          0, // rf_id
          root_id,
          0, // fw_parent
          -1, // seq_id
          static_cast<std::underlying_type_t<RecordScope>>(
              RecordScope::USER_SCOPE),
          tid,
          0); // fw_tid
      ob.out << ",";
    }
    fc.name = fn.name();
    auto num_inputs = fn.num_inputs();
    const auto inputs = fn.inputs();

    VLOG(2) << "inputs: " << num_inputs << " " << inputs.size() << std::endl;
    // We have two cases: for unboxed kernel, we have num_inputs ==
    // inputs.size() for boxed kernel using stack, there could be more elements
    // on the stack from previous ops.
    // TORCH_INTERNAL_ASSERT(num_inputs <= inputs.size());
    if (num_inputs > inputs.size()) {
      LOG(WARNING) << "RecordFunction " << fc.name
                   << " expected num_inputs=" << num_inputs
                   << " > inputs.size()=" << inputs.size();
      return;
    }
    // need to account for Stack mode where the inputs are at the end.
    size_t input_start = inputs.size() - num_inputs;

    for (const auto i : c10::irange(input_start, inputs.size())) {
      appendValueInfo(
          ob, inputs[i], fc.input_values, fc.input_types, fc.input_shapes);
    }
    fc.parent_id = ob.op_stack[tid].top();
    // get parent id from the forward stack, this can be different for
    // autograd ops, which may execute on a different thread than the original
    // thread (which should have the parent op on the stack).
    auto fw_tid = fn.forwardThreadId();
    if (fw_tid != 0) {
      fc.fw_parent_id = ob.op_stack[fw_tid].top();
    }
    // all input nodes should have id > op_id
    fc.op_id = ob.getNewID();
    ob.op_stack[tid].push(fc.op_id);

  } catch (const std::exception& e) {
    LOG(WARNING) << "Exception in execution graph observer: " << e.what();
  }
}

std::unique_ptr<ObserverContext> onFunctionEnter(const RecordFunction& fn) {
  using RunState = ExecutionGraphObserver::RunState;
  auto& ob = observer();

  if (ob.getState() == RunState::enabled) {
    // record op
    auto fc_ptr = std::make_unique<FunctionCallContext>();
    recordOperatorStart(ob, *fc_ptr.get(), fn);
    return fc_ptr;
  } else {
    return nullptr;
  }
}

void onFunctionExit(const RecordFunction& fn, ObserverContext* ctx_ptr) {
  using RunState = ExecutionGraphObserver::RunState;
  auto& ob = observer();
  if (ctx_ptr == nullptr) {
    return;
  }
  if (ob.getState() == RunState::enabled) {
    auto fc_ptr = dynamic_cast<FunctionCallContext*>(ctx_ptr);
    // TORCH_INTERNAL_ASSERT(fc_ptr != nullptr);
    if (fc_ptr == nullptr) {
      LOG(WARNING) << "FunctionCallContext is nullptr.";
      return;
    }
    auto& fc = *fc_ptr;

    auto outputs = fn.outputs();
    auto num_outputs = fn.num_outputs();
    // We have two cases: for unboxed kernel, we have num_outputs ==
    // outputs.size() for boxed kernel using stack, there could be more elements
    // on the stack from previous ops.
    VLOG(2) << "outputs: " << num_outputs << " " << outputs.size() << std::endl;
    // TORCH_INTERNAL_ASSERT(num_outputs <= outputs.size());
    if (num_outputs > outputs.size()) {
      LOG(WARNING) << "RecordFunction " << fc.name
                   << " num_outputs=" << num_outputs
                   << " > outputs.size()=" << outputs.size();
      return;
    }
    // need to account for Stack mode where the outputs are at the end.
    size_t output_start = outputs.size() - num_outputs;

    std::vector<std::string> output_types;
    std::vector<std::string> output_shapes;
    std::vector<std::string> output_values;
    try {
      const std::lock_guard<std::mutex> lock(ob.g_mutex);
      // remove current op id from stack

      ob.op_stack[fn.threadId()].pop();
      for (const auto i : c10::irange(output_start, outputs.size())) {
        appendValueInfo(
            ob, outputs[i], output_values, output_types, output_shapes);
      }

      std::string op_schema_str{};
      const auto op_schema = fn.operator_schema();
      if (op_schema.has_value()) {
        op_schema_str = c10::toString(op_schema.value());
      }

      writeJsonNode(
          ob.out,
          fc.name,
          fc.op_id,
          fn.handle(),
          fc.parent_id,
          fc.fw_parent_id,
          fn.seqNr(),
          static_cast<std::underlying_type_t<RecordScope>>(fn.scope()),
          fn.threadId(),
          fn.forwardThreadId(),
          vectorToString(fc.input_values),
          vectorToString(fc.input_shapes),
          vectorToString(fc.input_types),
          vectorToString(output_values),
          vectorToString(output_shapes),
          vectorToString(output_types),
          op_schema_str);
      ob.out << ",";
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception in execution graph observer: [" << fc.name
                   << " (" << fc.op_id << ")] " << e.what();
    }
  }
}

} // namespace

namespace profiler {
namespace impl {
// Add execution graph observer callback functions to the RecordFunction global
// observers.
bool addExecutionGraphObserver(const std::string& output_file_path) {
  // Making this static local to ensure it's instantiated just once when it's
  // called the first time.
  if (handle_ == INVALID_CALLBACK_HANDLE) {
    auto& ob = observer();
    ob.pid = processId();
    // Set output
    ob.file_name = output_file_path;
    if (!initExecutionGraphStart(ob)) {
      return false;
    }

    handle_ = addGlobalCallback(
        RecordFunctionCallback(&onFunctionEnter, &onFunctionExit)
            .needsInputs(true)
            .needsOutputs(true)
            .needsIds(true));
    // Default to disabled.
    ob.setState(ExecutionGraphObserver::RunState::disabled);

    VLOG(1) << "Added PyTorch execution graph observer, output="
            << output_file_path;
  } else {
    LOG(WARNING) << "Execution graph observer is already registered.";
  }
  return handle_ != INVALID_CALLBACK_HANDLE;
}

void removeExecutionGraphObserver() {
  auto& ob = observer();
  if (ob.getState() != ExecutionGraphObserver::RunState::disabled) {
    disableExecutionGraphObserver();
  }

  if (handle_ != INVALID_CALLBACK_HANDLE) {
    finalizeExecutionGraphOutput(ob);
    removeCallback(handle_);
    handle_ = INVALID_CALLBACK_HANDLE;
    ob.setState(ExecutionGraphObserver::RunState::uninitialized);
    VLOG(1) << "Removed PyTorch execution graph observer";
  } else {
    LOG(WARNING) << "Execution graph observer was not registered.";
  }
}

void enableExecutionGraphObserver() {
  VLOG(1) << "enableExecutionGraphObserver() ";
  auto& ob = observer();
  // Make sure we are not already enabled.
  if (ob.getState() == ExecutionGraphObserver::RunState::enabled) {
    LOG(WARNING)
        << "Trying to enable Execution Graph Observer when it's already enabled.";
  } else {
    ob.setState(ExecutionGraphObserver::RunState::enabled);
  }
}

void disableExecutionGraphObserver() {
  VLOG(1) << "disableExecutionGraphObserver()";
  auto& ob = observer();
  if (ob.getState() != ExecutionGraphObserver::RunState::disabled) {
    ob.setState(ExecutionGraphObserver::RunState::disabled);
  } else {
    LOG(WARNING)
        << "Trying to disable Execution Graph Observer when it's already disabled.";
  }
}
} // namespace impl
} // namespace profiler
} // namespace torch
