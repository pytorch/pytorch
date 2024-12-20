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
#include <fmt/ranges.h>
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

namespace torch::profiler::impl {

//******************************************************************************
// JSON output utility functions. To be merged with PyTorch profiler.
//******************************************************************************
template <typename T>
static std::string vectorToString(const std::vector<T>& v) {
  return fmt::format("[{}]", fmt::join(v, ","));
}

static std::string json_str_escape(const std::string& str);

constexpr size_t kMaxNumElements = 4096;

static std::string getScalarValue(const c10::IValue& val) {
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
    return fmt::format("\"{}\"", json_str_escape(str_val));
  } else if (val.isDevice()) {
    return fmt::format("\"{}\"", val.toDevice().str());
  }
  return fmt::format("\"<{}>\"", val.tagKind());
}

static int32_t processId() {
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
// like the `opStack` may be accessed across different threads. So we should be
// careful about data races. A global mutex `gMutex` is used avoid these races
// at the cost of performance in large number of threads situations. We may
// optimize this further to thread local, fine-grained locking, or use thread
// safe containers.
struct TORCH_API ExecutionTraceObserver { // NOLINT
  using ID = size_t;

  // Mapping of each thread to its own operator stack
  std::map<size_t, std::stack<ID>> opStack{};
  // Uses the underlying TensorImpl object pointer as the key and map to its
  // unique id.
  std::map<const void*, ID> objectId{};
  // Observer run state.
  enum class RunState { uninitialized, disabled, enabled };

  // Mutex for multithreaded access to the shared containers.
  std::recursive_mutex gMutex{};
  // Stream to write output JSON.
  std::ofstream out{};

  // Full path to the output file.
  std::string fileName{};

  // RecordFunction callback handle for this observer.
  CallbackHandle cbHandle{INVALID_CALLBACK_HANDLE};

  // Process ID.
  int32_t pid{-1};
  std::string recordTime{};

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
        reenableCallback(cbHandle);
      } else {
        disableCallback(cbHandle);
      }
    }
    state_ = newState;
  }

  bool record_integral_tensor_range{false};

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
const ExecutionTraceObserver::ID kUninitializedId{0};
// Root node has id = 1
const ExecutionTraceObserver::ID kRootId{1};

struct FunctionCallContext : public ObserverContext { // NOLINT
  std::string name;
  std::string kernelBackend;
  std::string kernelFile;
  ExecutionTraceObserver::ID opId{kUninitializedId};
  ExecutionTraceObserver::ID parentId{kUninitializedId};
  ExecutionTraceObserver::ID fwParentId{kUninitializedId};
  std::vector<std::string> inputTypes;
  std::vector<std::string> inputShapes;
  std::vector<std::string> inputStrides;
  std::vector<std::string> inputValues;
  std::map<int, std::pair<long, long>> tensor_index_min_max_map;

  std::string get_string_for_tensor_range() {
    if (tensor_index_min_max_map.empty()) {
      return "";
    }

    std::string result = "{";
    unsigned int i = 0;
    for (auto const& [key, value] : tensor_index_min_max_map) {
      if (i == tensor_index_min_max_map.size() - 1) {
        result += json_str_escape(
            fmt::format("\"{}\":[{},{}]", key, value.first, value.second));
      } else {
        result += json_str_escape(
            fmt::format("\"{}\":[{},{}],", key, value.first, value.second));
      }
      i++;
    }
    result += "}";
    return result;
  }
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

#ifdef USE_DISTRIBUTED
static std::string getAttrJson(
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
#endif

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
    const std::string& inputShapes = "[]",
    const std::string& inputStrides = "[]",
    const std::string& inputTypes = "[]",
    const std::string& outputs = "[]",
    const std::string& output_shapes = "[]",
    const std::string& output_strides = "[]",
    const std::string& output_types = "[]",
    const std::string& operator_schema = "",
    const std::string& kernelBackend = "",
    const std::string& kernelFile = "",
    const std::string& tensor_range = "",
    const std::string& additiona_attrs = "") {
  out << fmt::format(
      R"JSON(
    {{
      "id": {}, "name": "{}", "ctrl_deps": {},
      "inputs": {{"values": {}, "shapes": {}, "types": {}, "strides": {}}},
      "outputs": {{"values": {}, "shapes": {}, "types": {}, "strides": {}}},
      "attrs": [{{"name": "rf_id", "type": "uint64", "value": {}}},{{"name": "fw_parent", "type": "uint64", "value": {}}},{{"name": "seq_id", "type": "int64", "value": {}}},{{"name": "scope", "type": "uint64", "value": {}}},{{"name": "tid", "type": "uint64", "value": {}}},{{"name": "fw_tid", "type": "uint64", "value": {}}},{{"name": "op_schema", "type": "string", "value": "{}"}},{{"name": "kernel_backend", "type": "string", "value": "{}"}},{{"name": "kernel_file", "type": "string", "value": "{}"}},{{"name": "tensor_range", "type": "string", "value": "{}"}}{}]
    }})JSON",
      id,
      name,
      parent,
      inputs,
      inputShapes,
      inputTypes,
      inputStrides,
      outputs,
      output_shapes,
      output_types,
      output_strides,
      rf_id,
      fw_parent,
      seq_id,
      scope,
      tid,
      fw_tid,
      operator_schema,
      kernelBackend,
      kernelFile,
      tensor_range,
      additiona_attrs);
}

static std::string timeString(const std::time_t timepoint) {
  std::ostringstream oss;
  oss << std::put_time(std::localtime(&timepoint), "%Y-%m-%d %X"); // NOLINT
  return oss.str();
}

static bool initExecutionTraceStart(ExecutionTraceObserver& ob) {
  ob.out = openOutputFile(ob.fileName);
  // If somehow the output stream failed to open, finish observer here.
  if (!ob.out) {
    LOG(WARNING) << "Failed to open output file: " << ob.fileName;
    return false;
  }

  // Wall clock time for the first op collection time.
  const auto current_time = std::chrono::system_clock::now();
  ob.recordTime =
      timeString(std::chrono::system_clock::to_time_t(current_time));
  // Start timestamp using steady_clock for measurement.
  const auto timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();

  ob.out << fmt::format(
      R"JSON({{
  "schema": "1.1.1-chakra.0.0.4", "pid": {}, "time": "{}", "start_ts": {},
  "nodes": [)JSON",
      ob.pid,
      ob.recordTime,
      timestamp);
  return true;
}

// Write out Execution Trace to file
static void finalizeExecutionTraceOutput(ExecutionTraceObserver& ob) {
  writeJsonNode(
      ob.out,
      "[pytorch|profiler|execution_trace|process]",
      kRootId,
      0, // rf_id
      kRootId, // parent is self
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
  VLOG(1) << "PyTorch Execution Trace: written to file " << ob.fileName;
}

static ExecutionTraceObserver::ID getObjectID(
    ExecutionTraceObserver& ob,
    const void* t) {
  const std::lock_guard<std::recursive_mutex> lock(ob.gMutex);

  auto iter = ob.objectId.find(t);
  if (iter == ob.objectId.end()) {
    ExecutionTraceObserver::ID objectId = ob.getNewID();
    ob.objectId[t] = objectId;
    return objectId;
  }

  return iter->second;
}

static std::tuple<std::string, std::string, std::string, std::string>
convertIValue(
    ExecutionTraceObserver& ob,
    int& tensorIndex,
    std::map<int, std::pair<long, long>>& tensor_index_min_max_map,
    bool isInput,
    const c10::IValue& val,
    const bool baseType = true,
    const size_t maxArrayLen = kMaxNumElements) {
  std::string type = val.tagKind();
  if (val.isTensor()) {
    std::string tensor_shape, tensor_stride, tensor_type, tensor_value;

    const auto& tensor = val.toTensor();
    const auto tensor_impl = tensor.unsafeGetTensorImpl();
    if (tensor.defined() && !tensor_impl->has_symbolic_sizes_strides()) {
      // tensor shape
      tensor_shape = vectorToString(tensor.sizes().vec());
      // tensor strides
      tensor_stride = vectorToString(tensor.strides().vec());
    } else {
      tensor_shape = "[]";
      tensor_stride = "[]";
    }
    // tensor dtype
    type = type + fmt::format("({})", std::string(tensor.dtype().name()));
    tensor_type = baseType ? fmt::format("\"{}\"", type) : type;

    ExecutionTraceObserver::ID tensor_id = getObjectID(ob, tensor_impl);
    ExecutionTraceObserver::ID storage_id = 0;
    size_t offset = 0;
    size_t numel = 0;
    size_t itemsize = 0;
    std::string device_str = "";
    // symbolic sizes/strides implies t->storage_offset() will fail
    if (tensor_impl->has_storage() &&
        !tensor_impl->has_symbolic_sizes_strides()) {
      auto& t_storage = tensor_impl->storage();
      storage_id = getObjectID(ob, t_storage.data());
      offset = tensor_impl->storage_offset();
      numel = tensor_impl->numel();
      itemsize = tensor_impl->itemsize();
      device_str = tensor_impl->device().str();

      if (ob.record_integral_tensor_range && isInput &&
          at::isIntegralType(tensor.scalar_type(), false) &&
          tensor.numel() != 0) {
        enableRecordFunction(false);
        long min = tensor.min().item().toLong();
        long max = tensor.max().item().toLong();
        enableRecordFunction(true);
        tensor_index_min_max_map[tensorIndex] = std::make_pair(min, max);
      }
    }
    tensorIndex++;
    tensor_value = fmt::format(
        "[{},{},{},{},{},\"{}\"]",
        tensor_id,
        storage_id,
        offset,
        numel,
        itemsize,
        device_str);
    return std::make_tuple(
        tensor_shape, tensor_stride, tensor_type, tensor_value);
  } else if (val.isTuple()) {
    const auto& val_tuple = val.toTupleRef().elements();
    size_t tuple_size = val_tuple.size();
    std::vector<std::string> shape_array;
    std::vector<std::string> stride_array;
    std::vector<std::string> type_array;
    std::vector<std::string> value_array;
    for (const auto j : c10::irange(tuple_size)) {
      auto tuple = convertIValue(
          ob,
          tensorIndex,
          tensor_index_min_max_map,
          isInput,
          val_tuple[j],
          false,
          maxArrayLen);
      shape_array.push_back(std::get<0>(tuple));
      stride_array.push_back(std::get<1>(tuple));
      type_array.push_back(std::get<2>(tuple));
      value_array.push_back(std::get<3>(tuple));
    }
    type = type + vectorToString(type_array);
    std::string tensor_type = baseType ? fmt::format("\"{}\"", type) : type;
    return std::make_tuple(
        vectorToString(shape_array),
        vectorToString(stride_array),
        tensor_type,
        vectorToString(value_array));
  } else if (val.isList()) {
    const auto& val_list = val.toList();
    size_t list_size = val_list.size();
    std::vector<std::string> shape_array;
    std::vector<std::string> stride_array;
    std::vector<std::string> type_array;
    std::vector<std::string> value_array;
    for (const auto j : c10::irange(list_size)) {
      auto tuple = convertIValue(
          ob,
          tensorIndex,
          tensor_index_min_max_map,
          isInput,
          val_list.get(j),
          false,
          maxArrayLen);
      shape_array.push_back(std::get<0>(tuple));
      stride_array.push_back(std::get<1>(tuple));
      type_array.push_back(std::get<2>(tuple));
      value_array.push_back(std::get<3>(tuple));
      if (j >= maxArrayLen) {
        LOG(WARNING) << "list size=" << val_list.size()
                     << " exceeded maxArrayLen=" << maxArrayLen;
        break;
      }
    }
    type = type + vectorToString(type_array);
    std::string tensor_type = baseType ? fmt::format("\"{}\"", type) : type;
    return std::make_tuple(
        vectorToString(shape_array),
        vectorToString(stride_array),
        tensor_type,
        vectorToString(value_array));
  } else {
    std::string tensor_shape = "[]";
    std::string tensor_stride = "[]";
    std::string tensor_type = baseType ? fmt::format("\"{}\"", type) : type;
    std::string tensor_value = getScalarValue(val);

    return std::make_tuple(
        tensor_shape, tensor_stride, tensor_type, tensor_value);
  }
}

static void appendValueInfo(
    ExecutionTraceObserver& ob,
    int& tensorIndex,
    std::map<int, std::pair<long, long>>& tensor_index_min_max_map,
    bool isInput,
    const c10::IValue& val,
    std::vector<std::string>& shapes,
    std::vector<std::string>& strides,
    std::vector<std::string>& types,
    std::vector<std::string>& values) {
  auto tuple = convertIValue(
      ob, tensorIndex, tensor_index_min_max_map, isInput, val, true);
  shapes.push_back(std::get<0>(tuple));
  strides.push_back(std::get<1>(tuple));
  types.push_back(std::get<2>(tuple));
  values.push_back(std::get<3>(tuple));
}

static void handleKernelBackendInfo(
    FunctionCallContext& fc,
    const RecordFunction& fn) {
  // triton kernel related information are in kwinputs
  const auto& kwinputs = fn.kwinputs();
  if (kwinputs.find("kernel_backend") != kwinputs.end()) {
    fc.kernelBackend = kwinputs.at("kernel_backend").toStringRef();
    if (fc.kernelBackend == "triton") {
      fc.kernelFile = kwinputs.at("kernel_file").toStringRef();
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("kernel_file") != kwinputs.end(),
          "kernel file is missing in triton kernel");
      // Remove the path of the file name
      if (fc.kernelFile.find_last_of('/') != std::string::npos) {
        fc.kernelFile =
            fc.kernelFile.substr(fc.kernelFile.find_last_of('/') + 1);
      }

      // get grid information
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("grid") != kwinputs.end(),
          "grid is missing in triton kernel");
      fc.inputValues.emplace_back(
          "\"" + kwinputs.at("grid").toStringRef() + "\"");
      fc.inputTypes.emplace_back("\"String\"");
      fc.inputShapes.emplace_back("[]");

      // get stream information
      TORCH_INTERNAL_ASSERT(
          kwinputs.find("stream") != kwinputs.end(),
          "stream is missing in triton kernel");
      fc.inputValues.emplace_back(
          std::to_string(kwinputs.at("stream").toInt()));
      fc.inputTypes.emplace_back("\"Int\"");
      fc.inputShapes.emplace_back("[]");
    }
  }
}

// Additional attributes for commounication collectives
inline std::string getCommsNodeAttrs(const RecordFunction& fn) { // NOLINT
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
  // since we currently have this read called in onFunctionExit flow, we
  // should only introspect output tensors to prevent an INTERNAL ASSERT
  // FAILED in RecordFunction when we try to read input in RecordFunction exit
  // methods.
  auto meta = saveNcclMeta(fn, SaveNcclMetaConfig(false, true, false, true));

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
  return attrs.empty() ? "" : fmt::format(", {}", fmt::join(attrs, ", "));
}

static void recordOperatorStart(
    ExecutionTraceObserver& ob,
    FunctionCallContext& fc,
    const RecordFunction& fn) {
  auto tid = fn.threadId();

  try {
    {
      const std::lock_guard<std::recursive_mutex> lock(ob.gMutex);

      // if current thread stack is empty, push the root node to the stack
      // first
      if (ob.opStack[tid].empty()) {
        auto thread_node_id = ob.getNewID();
        ob.opStack[tid].push(thread_node_id);
        writeJsonNode(
            ob.out,
            "[pytorch|profiler|execution_trace|thread]",
            thread_node_id,
            0, // rf_id
            kRootId,
            0, // fw_parent
            -1, // seq_id
            static_cast<std::underlying_type_t<RecordScope>>(
                RecordScope::USER_SCOPE),
            tid,
            0); // fw_tid
        ob.out << ",";
      }
    }

    fc.name = fn.name();
    if (!checkFunctionInputsForLogging(fn)) {
      return;
    }
    auto num_inputs = fn.num_inputs();
    const auto inputs = fn.inputs();
    // need to account for Stack mode where the inputs are at the end.
    size_t input_start = inputs.size() - num_inputs;
    // tensor_index is the index of the flattened tensor list for all input
    // tensors
    int tensor_index = 0;
    for (const auto i : c10::irange(input_start, inputs.size())) {
      appendValueInfo(
          ob,
          tensor_index,
          fc.tensor_index_min_max_map,
          true,
          inputs[i],
          fc.inputShapes,
          fc.inputStrides,
          fc.inputTypes,
          fc.inputValues);
    }

    handleKernelBackendInfo(fc, fn);

    {
      const std::lock_guard<std::recursive_mutex> lock(ob.gMutex);

      fc.parentId = ob.opStack[tid].top();
      // get parent id from the forward stack, this can be different for
      // autograd ops, which may execute on a different thread than the
      // original thread (which should have the parent op on the stack).
      auto fw_tid = fn.forwardThreadId();
      if (fw_tid != 0) {
        fc.fwParentId = ob.opStack[fw_tid].top();
      }
      // all input nodes should have id > opId
      fc.opId = ob.getNewID();
      ob.opStack[tid].push(fc.opId);
    }

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

static std::string json_str_escape(const std::string& str) {
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
    } else if (ch <= '\x1f') {
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
    if (!checkFunctionOutputsForLogging(fn)) {
      return;
    }
    auto outputs = fn.outputs();
    auto num_outputs = fn.num_outputs();
    // need to account for Stack mode where the outputs are at the end.
    size_t output_start = outputs.size() - num_outputs;

    std::vector<std::string> output_types;
    std::vector<std::string> output_strides;
    std::vector<std::string> output_shapes;
    std::vector<std::string> output_values;
    try {
      int tensor_index = 0;
      for (const auto i : c10::irange(output_start, outputs.size())) {
        appendValueInfo(
            *ob,
            tensor_index,
            fc.tensor_index_min_max_map,
            false,
            outputs.at(i),
            output_shapes,
            output_strides,
            output_types,
            output_values);
      }

      std::string op_schema_str{};
      const auto op_schema = fn.operator_schema();
      if (op_schema.has_value()) {
        op_schema_str = json_str_escape(c10::toString(op_schema.value()));
      }

      const std::string additiona_attrs =
          fn.isNcclMeta() ? getCommsNodeAttrs(fn) : "";
      {
        const std::lock_guard<std::recursive_mutex> lock(ob->gMutex);

        // remove current op id from stack
        ob->opStack[fn.threadId()].pop();

        writeJsonNode(
            ob->out,
            fc.name,
            fc.opId,
            fn.handle(),
            fc.parentId,
            fc.fwParentId,
            fn.seqNr(),
            static_cast<std::underlying_type_t<RecordScope>>(fn.scope()),
            fn.threadId(),
            fn.forwardThreadId(),
            vectorToString(fc.inputValues),
            vectorToString(fc.inputShapes),
            vectorToString(fc.inputStrides),
            vectorToString(fc.inputTypes),
            vectorToString(output_values),
            vectorToString(output_shapes),
            vectorToString(output_strides),
            vectorToString(output_types),
            op_schema_str,
            fc.kernelBackend,
            fc.kernelFile,
            fc.get_string_for_tensor_range(),
            additiona_attrs);
        ob->out << ",";
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Exception in execution trace observer: [" << fc.name
                   << " (" << fc.opId << ")] " << e.what();
    }
  }
}

// Add execution trace observer callback functions to the RecordFunction
// global observers.
bool addExecutionTraceObserver(const std::string& output_file_path) {
  // Check if the observer is already initialized.
  if (ObserverManager::get() == nullptr) {
    ObserverManager::push(std::make_shared<ExecutionTraceObserver>());
    auto& ob = *ObserverManager::get();
    ob.pid = processId();
    // Set output
    ob.fileName = output_file_path;
    if (!initExecutionTraceStart(ob)) {
      return false;
    }

    // check if the environment variable is set to force recording integer
    // tensors
    auto env_variable =
        getenv("ENABLE_PYTORCH_EXECUTION_TRACE_INTEGRAL_TENSOR_RANGE");
    if (env_variable != nullptr) {
      ob.record_integral_tensor_range = true;
    }
    ob.cbHandle = addGlobalCallback(
        RecordFunctionCallback(&onFunctionEnter, &onFunctionExit)
            .needsInputs(true)
            .needsOutputs(true)
            .needsIds(true));
    // Default to disabled.
    ob.setState(ExecutionTraceObserver::RunState::disabled);

    VLOG(1) << "PyTorch Execution Trace: added observer, output="
            << output_file_path;
  } else if (ObserverManager::get()->cbHandle != INVALID_CALLBACK_HANDLE) {
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

    if (ob->cbHandle != INVALID_CALLBACK_HANDLE) {
      finalizeExecutionTraceOutput(*ob);
      removeCallback(ob->cbHandle);
      ob->cbHandle = INVALID_CALLBACK_HANDLE;
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
} // namespace torch::profiler::impl
