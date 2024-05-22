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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stack>
#include <vector>

#include <ATen/core/TensorBody.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/stack.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/standalone/execution_trace_observer.h>
#include <torch/csrc/profiler/util.h>

#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#endif // USE_DISTRIBUTED

using namespace at;

// Collective property attributes
// https://github.com/pytorch/pytorch/issues/124674
#ifdef USE_DISTRIBUTED
constexpr auto kETCommsName = "collective_name";
constexpr auto kETInMsgNelems = "in_msg_nelems";
constexpr auto kETOutMsgNelems = "out_msg_nelems";
constexpr auto kETInSplit = "in_split_size";
constexpr auto kETOutSplit = "out_split_size";
constexpr auto kETGlobalRankStart = "global_rank_start";
constexpr auto kETGlobalRankStride = "global_rank_stride";
constexpr auto kETGroupSize = "pg_size";
constexpr auto kETProcessGroupName = "pg_name";
constexpr auto kETProcessGroupDesc = "pg_desc";
#endif // USE_DISTRIBUTED

namespace torch {
namespace profiler {
namespace impl {

//******************************************************************************
// JSON output utility functions. To be merged with PyTorch profiler.
//******************************************************************************
template <typename T>
inline std::string vectorToString(const std::vector<T>& v) {
  return fmt::format("[{}]", fmt::join(v, ","));
}

std::string json_str_escape(const std::string& str);

constexpr size_t maxNumElements = 4096;
constexpr size_t maxStrLength = 8192;

inline std::string getValueType(
    const c10::IValue& val,
    const bool baseType = true,
    const size_t maxArrayLen = maxNumElements) {
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
    const size_t maxArrayLen = maxNumElements) {
  if (val.isTensor()) {
    auto& tensor = val.toTensor();
    if (tensor.defined() &&
        !tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides()) {
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
    double d_val = val.toDouble();
    if (std::isinf(d_val) || std::isnan(d_val)) {
      return fmt::format("\"{}\"", std::to_string(d_val));
    } else {
      return std::to_string(d_val);
    }
  } else if (val.isInt()) {
    return std::to_string(val.toInt());
  } else if (val.isBool()) {
    return val.toBool() ? "true" : "false";
  } else if (val.isString()) {
    const std::string& str_val = val.toStringRef();
    if (str_val.size() > maxStrLength) {
      LOG(WARNING) << "string size=" << str_val.size()
                   << " exceeded maxStrLength=" << maxStrLength;
      return fmt::format(
          "\"{}\"", json_str_escape(str_val.substr(0, maxStrLength)));
    }

    return fmt::format("\"{}\"", json_str_escape(str_val));
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
// Main ExecutionTraceObserver implementation.
//******************************************************************************

// ExecutionTraceObserver contains all the states of the observer. Some of them
// are shared between the enter and exit RecordFunction call backs, some data
// like the `op_stack` may be accessed across different threads. So we should be
// careful about data races. A global mutex `g_mutex` is used avoid these races
// at the cost of performance in large number of threads situations. We may
// optimize this further to thread local, fine-grained locking, or use thread
// safe containers.
struct TORCH_API ExecutionTraceObserver {
  using ID = size_t;

  // Mapping of each thread to its own operator stack
  std::map<size_t, std::stack<ID>> op_stack{};
  // Uses the underlying TensorImpl object pointer as the key and map to its
  // unique id.
  std::map<const void*, ID> object_id{};
  // Observer run state.
  enum class RunState { uninitialized, disabled, enabled };

  // Mutex for multithreaded access to the shared containers.
  std::recursive_mutex g_mutex{};
  // Stream to write output JSON.
  std::ofstream out{};

  // Full path to the output file.
  std::string file_name{};

  // RecordFunction callback handle for this observer.
  CallbackHandle cb_handle{INVALID_CALLBACK_HANDLE};

  // Process ID.
  int32_t pid{-1};
  std::string record_time{};

  ExecutionTraceObserver() = default;

  // Returns a new unique ID.
  ID getNewID() {
    return id_++;
  }

  RunState getState() const {
    return state_;
  }

  void setState(RunState newState) {
    if (state_ == RunState::uninitialized ||
        callbackShouldBeEnabled(state_) != callbackShouldBeEnabled(newState)) {
      if (callbackShouldBeEnabled(newState)) {
        reenableCallback(cb_handle);
      } else {
        disableCallback(cb_handle);
      }
    }
    state_ = newState;
  }

 private:
  static bool callbackShouldBeEnabled(RunState run_state) {
    return run_state == ExecutionTraceObserver::RunState::enabled;
  }

  // Must use accessors to change this so that we can keep the
  // RecordFunction callback in sync with the state.
  RunState state_{RunState::uninitialized};

  // All tensors and operators have an unique id assigned. Increment id for each
  // new tensor or operator node.
  // 0 -> unintialized
  // 1 -> root ID
  // 2 ... -> regular node ID
  std::atomic<ID> id_{2};
};

// Using a singleton manager here to allow init and delete the observer object.
using ObserverManager = GlobalStateManager<ExecutionTraceObserver>;

// Uninitialized node has id = 0
const ExecutionTraceObserver::ID uninitialized_id{0};
// Root node has id = 1
const ExecutionTraceObserver::ID root_id{1};

struct FunctionCallContext : public ObserverContext {
  std::string name;
  std::string kernel_backend;
  std::string kernel_file;
  ExecutionTraceObserver::ID op_id{uninitialized_id};
  ExecutionTraceObserver::ID parent_id{uninitialized_id};
  ExecutionTraceObserver::ID fw_parent_id{uninitialized_id};
  std::vector<std::string> input_types;
  std::vector<std::string> input_shapes;
  std::vector<std::string> input_values;
};

// Opens the json file to write the execution trace.
static std::ofstream openOutputFile(const std::string& name) {
  std::ofstream stream;
  stream.open(name, std::ofstream::out | std::ofstream::trunc);
  if (!stream) {
    LOG(ERROR) << "Failed to open '" << name << "'";
  } else {
    VLOG(1) << "PyTorch Execution Trace: writing to " << name;
  }
  return stream;
}

static inline std::string getAttrJson(
    const std::string& name,
    const std::string& type,
    const std::string& value) {
  // note name and type are not quoted but value should be if it is a string.
  return fmt::format(
      R"JSON(
  {{"name": "{}", "type": "{}", "value": {}}})JSON",
      name,
      type,
      value);
}

static void writeJsonNode(
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
    const std::string& operator_schema = "",
    const std::string& kernel_backend = "",
    const std::string& kernel_file = "",
    const std::string& additiona_attrs = "") {
  out << fmt::format(
      R"JSON(
    {{
      "id": {}, "name": "{}", "ctrl_deps": {},
      "inputs": {{"values": {}, "shapes": {}, "types": {}}},
      "outputs": {{"values": {}, "shapes": {}, "types": {}}},
      "attrs": [{{"name": "rf_id", "type": "uint64", "value": {}}},{{"name": "fw_parent", "type": "uint64", "value": {}}},{{"name": "seq_id", "type": "int64", "value": {}}},{{"name": "scope", "type": "uint64", "value": {}}},{{"name": "tid", "type": "uint64", "value": {}}},{{"name": "fw_tid", "type": "uint64", "value": {}}},{{"name": "op_schema", "type": "string", "value": "{}"}},{{"name": "kernel_backend", "type": "string", "value": "{}"}},{{"name": "kernel_file", "type": "string", "value": "{}"}}{}]
    }})JSON",
      id,
      name,
      parent,
      inputs,
      input_shapes,
      input_types,
      outputs,
      output_shapes,
      output_types,
      rf_id,
      fw_parent,
      seq_id,
      scope,
      tid,
      fw_tid,
      operator_schema,
      kernel_backend,
      kernel_file,
      additiona_attrs);
}

inline std::string timeString(const std::time_t timepoint) {
  std::ostringstream oss;
  oss << std::put_time(std::localtime(&timepoint), "%Y-%m-%d %X");
  return oss.str();
}

static bool initExecutionTraceStart(ExecutionTraceObserver& ob) {
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
  "schema": "1.1.0-chakra.0.0.4", "pid": {}, "time": "{}", "start_ts": {},
  "nodes": [)JSON",
      ob.pid,
      ob.record_time,
      timestamp);
  return true;
}

// Write out Execution Trace to file
static void finalizeExecutionTraceOutput(ExecutionTraceObserver& ob) {
  writeJsonNode(
      ob.out,
      "[pytorch|profiler|execution_trace|process]",
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
  VLOG(1) << "PyTorch Execution Trace: written to file " << ob.file_name;
}

inline ExecutionTraceObserver::ID getObjectID(
    ExecutionTraceObserver& ob,
    const void* t) {
  auto iter = ob.object_id.find(t);
  if (iter == ob.object_id.end()) {
    ExecutionTraceObserver::ID object_id = ob.getNewID();
    ob.object_id[t] = object_id;
    return object_id;
  }

  return iter->second;
}

inline std::string convertIValue(
    ExecutionTraceObserver& ob,
    const c10::IValue& val,
    const size_t maxArrayLen = maxNumElements) {
  if (val.isTensor()) {
    const auto t = val.toTensor().unsafeGetTensorImpl();
    ExecutionTraceObserver::ID tensor_id = getObjectID(ob, t);
    ExecutionTraceObserver::ID storage_id = 0;
    size_t offset = 0;
    size_t numel = 0;
    size_t itemsize = 0;
    std::string device_str = "";
    // symbolic sizes/strides implies t->storage_offset() will fail
    if (t->has_storage() && !t->has_symbolic_sizes_strides()) {
      auto& t_storage = t->storage();
      storage_id = getObjectID(ob, t_storage.data());
      offset = t->storage_offset();
      numel = t->numel();
      itemsize = t->itemsize();
      device_str = t->device().str();
    }
    return fmt::format(
        "[{},{},{},{},{},\"{}\"]",
        tensor_id,
        storage_id,
        offset,
        numel,
        itemsize,
        device_str);
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
    ExecutionTraceObserver& ob,
    const c10::IValue& val,
    std::vector<std::string>& values,
    std::vector<std::string>& types,
    std::vector<std::string>& shapes) {
  values.push_back(convertIValue(ob, val));
  types.push_back(getValueType(val));
  shapes.push_back(getValueShape(val));
}

inline void handleKernelBackendInfo(
    FunctionCallContext& fc,
    const RecordFunction& fn) {
  // triton kernel related information are in kwinputs
  const auto& kwinputs = fn.kwinputs();
  if (kwinputs.find("kernel_backend") != kwinputs.end()) {
    fc.kernel_backend = kwinputs.at("kernel_backend").toStringRef();
    if (fc.kernel_backend == "triton") {
      fc.kernel_file = kwinputs.at("kernel_file").toStringRef();
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("kernel_file") != kwinputs.end(),
          "kernel file is missing in triton kernel");
      // Remove the path of the file name
      if (fc.kernel_file.find_last_of('/') != std::string::npos)
        fc.kernel_file =
            fc.kernel_file.substr(fc.kernel_file.find_last_of('/') + 1);

      // get grid information
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("grid") != kwinputs.end(),
          "grid is missing in triton kernel");
      fc.input_values.emplace_back(
          "\"" + kwinputs.at("grid").toStringRef() + "\"");
      fc.input_types.emplace_back("\"String\"");
      fc.input_shapes.emplace_back("[]");

      // get stream information
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("stream") != kwinputs.end(),
          "stream is missing in triton kernel");
      fc.input_values.emplace_back(
          std::to_string(kwinputs.at("stream").toInt()));
      fc.input_types.emplace_back("\"Int\"");
      fc.input_shapes.emplace_back("[]");
    }
  }
}

// Additional attributes for commounication collectives
inline std::string getCommsNodeAttrs(const RecordFunction& fn) {
  std::vector<std::string> attrs;

#ifdef USE_DISTRIBUTED
  // We rely on paramcommsdebug object that is available in thread local info
  auto debugInfo = dynamic_cast<ParamCommsDebugInfo*>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PARAM_COMMS_INFO));
  if (debugInfo == nullptr) {
    LOG(WARNING) << "ParamCommsDebugInfo not available for function: "
                 << fn.name();
    return ", " + getAttrJson("debug", "string", "\"missing comms info\"");
  }

  // get NcclMeta from record function, this used ParamCommsDebugInfo above
  auto meta = saveNcclMeta(fn, false /*truncate*/);

  auto addAttr =
      [&](const char* commsMetaName, const char* etMetaName, const char* type) {
        auto it = meta.find(commsMetaName);
        if (it != meta.end()) {
          attrs.push_back(getAttrJson(etMetaName, type, it->second));
        }
      };

  addAttr(kCommsName, kETCommsName, "string");
  addAttr(kDtype, kDtype, "string");

  addAttr(kInMsgNelems, kETInMsgNelems, "uint64");
  addAttr(kOutMsgNelems, kETOutMsgNelems, "uint64");

  // following two metadata are lists.
  addAttr(kInSplit, kETInSplit, "string");
  addAttr(kOutSplit, kETOutSplit, "string");

  addAttr(kGlobalRankStart, kETGlobalRankStart, "uint64");
  addAttr(kGlobalRankStride, kETGlobalRankStride, "uint64");

  // pg_name is a string.
  addAttr(kProcessGroupName, kETProcessGroupName, "string");
  addAttr(kProcessGroupDesc, kETProcessGroupDesc, "string");

  addAttr(kGroupSize, kETGroupSize, "uint64");

#endif // USE_DISTRIBUTED

  // XXX consider using as string stream?
  return attrs.size() == 0 ? "" : fmt::format(", {}", fmt::join(attrs, ", "));
}

static void recordOperatorStart(
    ExecutionTraceObserver& ob,
    FunctionCallContext& fc,
    const RecordFunction& fn) {
  auto tid = fn.threadId();

  try {
    const std::lock_guard<std::recursive_mutex> lock(ob.g_mutex);

    // if current thread stack is empty, push the root node to the stack first
    if (ob.op_stack[tid].empty()) {
      auto thread_node_id = ob.getNewID();
      ob.op_stack[tid].push(thread_node_id);
      writeJsonNode(
          ob.out,
          "[pytorch|profiler|execution_trace|thread]",
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

    handleKernelBackendInfo(fc, fn);

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
    LOG(WARNING) << "Exception in execution trace observer: " << e.what();
  }
}

static std::unique_ptr<ObserverContext> onFunctionEnter(
    const RecordFunction& fn) {
  using RunState = ExecutionTraceObserver::RunState;
  auto ob = ObserverManager::get();
  if (ob != nullptr && ob->getState() == RunState::enabled) {
    // record op
    auto fc_ptr = std::make_unique<FunctionCallContext>();
    recordOperatorStart(*ob, *fc_ptr.get(), fn);
    return fc_ptr;
  }
  return nullptr;
}

inline std::string json_str_escape(const std::string& str) {
  std::ostringstream ostream;
  for (char ch : str) {
    if (ch == '"') {
      ostream << "\\\"";
    } else if (ch == '\\') {
      ostream << "\\\\";
    } else if (ch == '\b') {
      ostream << "\\b";
    } else if (ch == '\f') {
      ostream << "\\f";
    } else if (ch == '\n') {
      ostream << "\\n";
    } else if (ch == '\r') {
      ostream << "\\r";
    } else if (ch == '\t') {
      ostream << "\\t";
    } else if ('\x00' <= ch && ch <= '\x1f') {
      ostream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch);
    } else {
      ostream << ch;
    }
  }
  return ostream.str();
}

static void onFunctionExit(const RecordFunction& fn, ObserverContext* ctx_ptr) {
  using RunState = ExecutionTraceObserver::RunState;
  auto ob = ObserverManager::get();
  if (ob == nullptr || ctx_ptr == nullptr) {
    return;
  }
  if (ob->getState() == RunState::enabled) {
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
      const std::lock_guard<std::recursive_mutex> lock(ob->g_mutex);
      // remove current op id from stack

      ob->op_stack[fn.threadId()].pop();
      for (const auto i : c10::irange(output_start, outputs.size())) {
        appendValueInfo(
            *ob, outputs[i], output_values, output_types, output_shapes);
      }

      std::string op_schema_str{};
      const auto op_schema = fn.operator_schema();
      if (op_schema.has_value()) {
        op_schema_str = json_str_escape(c10::toString(op_schema.value()));
      }

      const std::string additiona_attrs =
          fn.isNcclMeta() ? getCommsNodeAttrs(fn) : "";

      writeJsonNode(
          ob->out,
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
          op_schema_str,
          fc.kernel_backend,
          fc.kernel_file,
          additiona_attrs);
      ob->out << ",";
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception in execution trace observer: [" << fc.name
                   << " (" << fc.op_id << ")] " << e.what();
    }
  }
}

// Add execution trace observer callback functions to the RecordFunction global
// observers.
bool addExecutionTraceObserver(const std::string& output_file_path) {
  // Check if the observer is already initialized.
  if (ObserverManager::get() == nullptr) {
    ObserverManager::push(std::make_shared<ExecutionTraceObserver>());
    auto& ob = *ObserverManager::get();
    ob.pid = processId();
    // Set output
    ob.file_name = output_file_path;
    if (!initExecutionTraceStart(ob)) {
      return false;
    }

    ob.cb_handle = addGlobalCallback(
        RecordFunctionCallback(&onFunctionEnter, &onFunctionExit)
            .needsInputs(true)
            .needsOutputs(true)
            .needsIds(true));
    // Default to disabled.
    ob.setState(ExecutionTraceObserver::RunState::disabled);

    VLOG(1) << "PyTorch Execution Trace: added observer, output="
            << output_file_path;
  } else if (ObserverManager::get()->cb_handle != INVALID_CALLBACK_HANDLE) {
    LOG(WARNING) << "Execution trace observer is already registered.";
  }
  return true;
}

void removeExecutionTraceObserver() {
  auto ob = ObserverManager::get();
  if (ob != nullptr) {
    if (ob->getState() != ExecutionTraceObserver::RunState::disabled) {
      disableExecutionTraceObserver();
    }

    if (ob->cb_handle != INVALID_CALLBACK_HANDLE) {
      finalizeExecutionTraceOutput(*ob);
      removeCallback(ob->cb_handle);
      ob->cb_handle = INVALID_CALLBACK_HANDLE;
      // Release the current ET observer object and reset.
      TORCH_INTERNAL_ASSERT(
          ObserverManager::pop() != nullptr,
          "Global state ptr cannot be null before resetting");
      VLOG(1) << "PyTorch Execution Trace: removed observer";
    } else {
      LOG(WARNING) << "Execution trace observer was not registered.";
    }
  } else {
    LOG(WARNING) << "Execution trace observer was not initialized.";
  }
}

void enableExecutionTraceObserver() {
  LOG(WARNING) << "Enabling Execution Trace Observer";
  auto& ob = *ObserverManager::get();
  // Make sure we are not already enabled.
  if (ob.getState() == ExecutionTraceObserver::RunState::enabled) {
    LOG(WARNING)
        << "Trying to enable Execution Trace Observer when it's already enabled.";
  } else {
    ob.setState(ExecutionTraceObserver::RunState::enabled);
  }
}

void disableExecutionTraceObserver() {
  LOG(WARNING) << "Disabling Execution Trace Observer";
  auto& ob = *ObserverManager::get();
  if (ob.getState() != ExecutionTraceObserver::RunState::disabled) {
    ob.setState(ExecutionTraceObserver::RunState::disabled);
  } else {
    LOG(WARNING)
        << "Trying to disable Execution Trace Observer when it's already disabled.";
  }
}
} // namespace impl
} // namespace profiler
} // namespace torch
