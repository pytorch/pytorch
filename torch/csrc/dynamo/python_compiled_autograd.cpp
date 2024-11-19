#include <torch/csrc/dynamo/python_compiled_autograd.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/dynamo/compiled_autograd.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*
[Note: Compiled Autograd]

Compiled autograd replaces the standard autograd engine by converting
the autograd graph to an FX graph that can be torch.compiled. It caches
this conversion using a shadow graph. We compare the new graph to the
shadow graph by walking the two graphs simultaneously and computing a
CacheKey for each original node to find the next edge in the shadow graph.
Two different graphs might have a shared common prefix in the shadow
graph, but then diverge at the first difference. Tensors, SavedVariables,
and SymInt found stored on the nodes in the autograd graph are lifted to
become inputs to the graph. All other properties (ints, floats, types,
etc.) are specialized using the CacheKey and will result in landing on
a different cache node in the shadow graph if some property differs.

To interact with the (hundreds) of different autograd::Node types,
we use a visitor pattern that walks each Node structure recursively.

- The first pass, compiled_args/collect, extracts all the inputs to the
graph and builds a CacheKey for us to specialize on.  On a cache hit,
we stop here and this is the only pass.

- On a cache miss, a second pass kicks in to extract the FX graph using
apply_with_saved, which uses another visitor pattern.  The before()
visitor swaps out all the Tensors, SavedVariables, and SymInt for
fake/symbolic versions to allow tracing.  We then run the standard apply()
method, and after() restores things to how we found them.

When we see tensor hooks, we record them directly in the output graph
without tracing into them.  We do this to avoid executing unsafe code
at trace time.

Notes:
  - We require hooks to not change shapes of tensors.
  - We require non-hook autograd nodes to be tracable.
*/

namespace torch::dynamo::autograd {
using c10::SymInt;

static PyObject* wrap_int_list(const std::vector<int64_t>& inputs) {
  PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(inputs.size()));
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, PyLong_FromSsize_t(inputs[i]));
  }
  return pyinput;
}

static PyObject* convert_hook_list(std::vector<c10::SafePyObject>& inputs) {
  // inplace, consumes the input hooks
  PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(inputs.size()));
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, inputs[i].release());
  }
  return pyinput;
}

// see https://github.com/pytorch/pytorch/pull/34845
static void throw_python_error() {
  python_error err;
  err.persist();
  // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
  throw err;
}

static PyObject* check(PyObject* pyresult) {
  if (C10_UNLIKELY(pyresult == nullptr)) {
    throw_python_error();
  }
  return pyresult;
}

static void check(bool result) {
  if (C10_UNLIKELY(!result))
    check(nullptr);
}

// snapshot of python verbose logging toggle
static PyObject* python_verbose_logger = nullptr;

struct PythonLogger {
  PythonLogger() = delete;
  explicit PythonLogger(PyObject* logger) : logger_(logger) {
    TORCH_INTERNAL_ASSERT(logger_ != nullptr);
  }

  enum Level : unsigned int {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4,
    COUNT // Keep this as the last enum
  };

  // must be called while GIL is held
  void log(Level level, std::string_view msg) const {
    THPObjectPtr pymethod(PyUnicode_FromString(levelNames_[level].data()));
    TORCH_INTERNAL_ASSERT(pymethod != nullptr);
    THPObjectPtr pyfunc(PyObject_GetAttr(logger_, pymethod.get()));
    if (pyfunc == nullptr) {
      throw_python_error();
    }
    PyObject* result = PyObject_CallFunction(pyfunc.get(), "s", msg.data());
    if (result == nullptr) {
      throw_python_error();
    }
  }

 private:
  static constexpr std::array<std::string_view, COUNT> levelNames_ = {
      "debug", // Level::DEBUG
      "info", // Level::INFO
      "warning", // Level::WARNING
      "error", // Level::ERROR
      "critical" // Level::CRITICAL
  };

  // Note: logger_ must stay valid for the lifetime of this object
  PyObject* logger_;
};

struct VerboseLogger : public PythonLogger {
  static std::optional<VerboseLogger> maybe_create() {
    if (python_verbose_logger == nullptr) {
      return std::nullopt;
    }
    return VerboseLogger(python_verbose_logger);
  }

  VerboseLogger(PyObject* vlogger) : PythonLogger(vlogger) {}

  void log_node_check(
      const Node& fn,
      size_t size_inputs_num,
      std::unordered_set<CacheKey> cached_keys,
      const CacheKey& key,
      size_t node_idx) {
    std::string node_name =
        fn.name() + " (NodeCall " + std::to_string(node_idx) + ")";

    cumulative_sizes_per_node[size_inputs_num] = node_name;

    if (!logged_node_miss && cached_keys.find(key) == cached_keys.end()) {
      _log_node_miss(typeid(fn), cached_keys, key, node_name);
      logged_node_miss = true;
    }
  }

  void _log_node_miss(
      const std::type_info& node_type,
      std::unordered_set<CacheKey> cached_keys,
      const CacheKey& key,
      const std::string& node_name) const {
    std::ostringstream oss;
    oss << "Cache miss due to new autograd node: " << node_name
        << " with key size " << std::to_string(key.key_size)
        << ", previous key sizes=[";

    for (auto it = cached_keys.begin(); it != cached_keys.end(); it++) {
      if (it->node_type != node_type) {
        continue;
      }
      oss << it->key_size;
      if (std::next(it) != cached_keys.end()) {
        oss << ",";
      }
    }
    oss << "]";
    log(PythonLogger::DEBUG, oss.str());
  }

  void log_dynamic_shapes_check(size_t size_idx) const {
    if (cumulative_sizes_per_node.empty()) {
      return;
    }

    auto it = cumulative_sizes_per_node.lower_bound(size_idx);
    TORCH_CHECK(it != cumulative_sizes_per_node.end());
    size_t start_idx =
        it == cumulative_sizes_per_node.begin() ? 0 : std::prev(it)->first;
    log(PythonLogger::DEBUG,
        "Cache miss due to changed shapes: marking size idx " +
            std::to_string(size_idx - start_idx) + " of " + it->second +
            " as dynamic");
  }

  // track which size index belongs to which node
  std::map<size_t, std::string> cumulative_sizes_per_node;
  // only log cache miss due to node key once
  bool logged_node_miss = false;
};

struct CacheNode {
  // A node in the shadow graph, we follow next edges until we reach the end of
  // the graph
  static CacheNode* root() {
    static CacheNode _root;
    return &_root;
  }

  CacheNode* lookup(const CacheKey& key, bool create = true) {
    auto it = next.find(key);
    if (it == next.end()) {
      if (!create)
        return nullptr;
      // caller's key is in temporary memory, must copy it
      CacheKeyBuffer buffer(key.key, key.key_size);
      CacheKey key_with_storage(key.node_type, buffer.get(), key.key_size);
      it = next.emplace(key_with_storage, std::make_unique<CacheNode>()).first;
      key_storage.emplace_back(std::move(buffer));
    }
    return it->second.get();
  }

  void clear() {
    next.clear();
    key_storage.clear();
    expected_sizes.clear();
    runtime_wrapper = nullptr;
    compiled_fn = nullptr;
  }

  bool is_empty() const {
    return next.empty() && !compiled_fn;
  }

  CacheNode() : runtime_wrapper(nullptr), compiled_fn(nullptr) {}
  ~CacheNode() {
    if (!Py_IsInitialized()) {
      // leak on shutdown
      runtime_wrapper.release();
      compiled_fn.release();
    }
  }
  CacheNode(CacheNode&&) = delete;
  CacheNode(const CacheNode&) = delete;
  CacheNode& operator=(const CacheNode&) = delete;
  CacheNode& operator=(CacheNode&&) = delete;

  bool check_dynamic_sizes(
      AutogradCompilerCall& call,
      const std::optional<VerboseLogger>& vlogger) {
    /*
    We start off by assuming everything is static, then we mark things
    as dynamic when we see them change.  This function:
      1) Checks for a cache hit
      2) Updates expected_sizes to track what is dynamic
      3) Populates call.dyn_size_inputs by filtering call.all_size_inputs
    */
    bool cache_hit = compiled_fn.get() != nullptr;
    auto len = call.all_size_inputs.size();
    const SizeInput* data = call.all_size_inputs.data();
    if (expected_sizes.empty()) {
      expected_sizes.reserve(len);
      for (const auto i : c10::irange(len)) {
        expected_sizes.emplace_back(data[i]);
      }
    }

    TORCH_INTERNAL_ASSERT(expected_sizes.size() == call.all_size_inputs.size());
    if (!call.size_input_origins.empty()) {
      TORCH_INTERNAL_ASSERT(
          call.all_size_inputs.size() == call.size_input_origins.size());
    }
    std::vector<uint32_t> dynamic_size_input_origins;
    dynamic_size_input_origins.reserve(len);
    for (const auto i : c10::irange(len)) {
      auto& expected = expected_sizes[i];
      bool was_dynamic = expected.dyn_type == SizeInput::DYNAMIC;
      bool changed_value = expected.value != data[i].value;
      if (changed_value) {
        if (!was_dynamic) {
          cache_hit = false;
          if (vlogger.has_value()) {
            vlogger->log_dynamic_shapes_check(i);
          }
        }
        expected = SizeInput(SizeInput::DYNAMIC, data[i].value);
      }

      if (changed_value || was_dynamic) {
        if (call.dyn_size_inputs.empty()) {
          call.dyn_size_inputs.reserve(len);
        }
        call.dyn_size_inputs.emplace_back(data[i].value);
        if (!call.size_input_origins.empty()) {
          dynamic_size_input_origins.emplace_back(call.size_input_origins[i]);
        }
      }
    }
    call.size_input_origins = std::move(dynamic_size_input_origins);

    if (!cache_hit) {
      // we missed cache because static size inputs didn't match; force
      // recompilation with the varying size input as dynamic
      runtime_wrapper = nullptr;
      compiled_fn = nullptr;
    }
    return cache_hit;
  }

  PyObject* wrap_dynamic_inputs() const {
    size_t dynamic_count = 0;
    size_t idx = 0;
    for (const auto& i : expected_sizes) {
      if (i.dyn_type == SizeInput::DYNAMIC) {
        ++dynamic_count;
      }
    }
    PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(dynamic_count));
    for (const auto& i : expected_sizes) {
      if (i.dyn_type == SizeInput::DYNAMIC) {
        PyTuple_SET_ITEM(pyinput, idx++, PyLong_FromSsize_t(i.value));
      }
    }
    TORCH_INTERNAL_ASSERT(idx == dynamic_count);
    return pyinput;
  }

  std::vector<std::optional<SymInt>> unwrap_dynamic_inputs(
      PyObject* pyresult) const {
    TORCH_INTERNAL_ASSERT(PyList_CheckExact(pyresult));
    size_t idx = 0;
    size_t result_len = PyList_GET_SIZE(pyresult);
    std::vector<std::optional<SymInt>> result;
    result.reserve(expected_sizes.size());
    for (const auto& i : expected_sizes) {
      if (i.dyn_type == SizeInput::DYNAMIC) {
        TORCH_INTERNAL_ASSERT(idx < result_len);
        result.emplace_back(
            py::cast<c10::SymInt>(PyList_GET_ITEM(pyresult, idx++)));
      } else {
        result.emplace_back();
      }
    }
    TORCH_INTERNAL_ASSERT(
        idx == result_len && result.size() == expected_sizes.size());
    return result;
  }

  std::unordered_map<CacheKey, std::unique_ptr<CacheNode>> next;
  std::vector<CacheKeyBuffer> key_storage;
  std::vector<SizeInput> expected_sizes;

  THPObjectPtr runtime_wrapper;
  THPObjectPtr compiled_fn;
};

struct InputBuffers : public std::unordered_map<Node*, InputBuffer> {
  InputBuffer& lookup(Node* function) {
    auto it = emplace(function, InputBuffer(function->num_inputs())).first;
    return it->second;
  }
};

static PyObject* the_autograd_compiler = nullptr;
static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args);

static PyObject* clear_cache(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS;
  CacheNode::root()->clear();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS;
}

static PyObject* is_cache_empty(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS;
  if (CacheNode::root()->is_empty()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS;
}

static PyObject* set_verbose_logger(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS;
  PyObject* logger = nullptr;
  if (!PyArg_ParseTuple(args, "O", &logger)) {
    throw_python_error();
  }

  if (logger == Py_None) {
    python_verbose_logger = nullptr;
  } else {
    python_verbose_logger = logger;
  }
  Py_RETURN_TRUE;
  END_HANDLE_TH_ERRORS;
}

// NOLINTNEXTLINE(*array*)
static PyMethodDef _methods[] = {
    {"set_autograd_compiler", set_autograd_compiler, METH_VARARGS, nullptr},
    {"clear_cache", clear_cache, METH_NOARGS, nullptr},
    {"is_cache_empty", is_cache_empty, METH_NOARGS, nullptr},
    {"set_verbose_logger", set_verbose_logger, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.autograd_compiler",
    "Hooks for compiling autograd",
    -1,
    _methods};

PyObject* wrap_lifted_ivalue_args(
    const std::vector<LiftedIValueArg>& lifted_ivalue_args) {
  PyObject* pyivalueargs =
      PyList_New(static_cast<Py_ssize_t>(lifted_ivalue_args.size()));
  size_t idx = 0;
  for (const auto& arg : lifted_ivalue_args) {
    if (arg.actual_ptr->isInt() || arg.actual_ptr->isSymInt()) {
      PyList_SET_ITEM(
          pyivalueargs, idx++, PyLong_FromSsize_t(arg.actual_ptr->toInt()));
    } else if (arg.actual_ptr->isDouble() || arg.actual_ptr->isSymFloat()) {
      PyList_SET_ITEM(
          pyivalueargs, idx++, PyFloat_FromDouble(arg.actual_ptr->toDouble()));
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected lifted ivalue type");
    }
  }
  return pyivalueargs;
}

PyObject* wrap_node_origins(
    const AutogradCompilerCall& compiler,
    size_t dynamic_sizes) {
  TORCH_INTERNAL_ASSERT(
      compiler.tensor_args.input_origins.empty() ||
      (compiler.tensor_args.input_origins.size() ==
       compiler.tensor_args.inputs.size()));
  TORCH_INTERNAL_ASSERT(
      compiler.size_input_origins.empty() ||
      (compiler.size_input_origins.size() == dynamic_sizes));
  TORCH_INTERNAL_ASSERT(
      compiler.lifted_ivalue_args.args_origins.empty() ||
      (compiler.lifted_ivalue_args.args_origins.size() ==
       compiler.lifted_ivalue_args.args.size()));
  PyObject* pyallorigins = PyList_New(3);
  size_t next = 0;
  for (const std::vector<uint32_t>& vec :
       {compiler.tensor_args.input_origins,
        compiler.size_input_origins,
        compiler.lifted_ivalue_args.args_origins}) {
    PyObject* pyorigins = PyList_New(static_cast<Py_ssize_t>(vec.size()));
    for (const auto i : c10::irange(vec.size())) {
      uint32_t node_id = vec[i];
      PyObject* pyorigin = PyTuple_Pack(
          2,
          THPUtils_packUInt32(node_id),
          PyUnicode_FromString(
              compiler.node_calls.lookup(node_id).node->name().c_str()));
      PyList_SET_ITEM(pyorigins, i, pyorigin);
    }
    PyList_SET_ITEM(pyallorigins, next++, pyorigins);
  }
  return pyallorigins;
}

void set_ivalue_proxies(
    PyObject* fake_ivalue_args,
    std::vector<LiftedIValueArg>& lifted_ivalue_args) {
  TORCH_INTERNAL_ASSERT(PyList_Check(fake_ivalue_args));
  TORCH_INTERNAL_ASSERT(
      static_cast<size_t>(PyList_Size(fake_ivalue_args)) ==
      lifted_ivalue_args.size());

  for (const auto& i : c10::irange(lifted_ivalue_args.size())) {
    auto& arg = lifted_ivalue_args[i];
    if (arg.actual_ptr->isInt() || arg.actual_ptr->isSymInt()) {
      arg.proxy = at::IValue(
          py::cast<c10::SymInt>(PyList_GET_ITEM(fake_ivalue_args, i)));
      TORCH_INTERNAL_ASSERT(arg.proxy.isSymInt());
    } else if (arg.actual_ptr->isDouble() || arg.actual_ptr->isSymFloat()) {
      arg.proxy = at::IValue(
          py::cast<c10::SymFloat>(PyList_GET_ITEM(fake_ivalue_args, i)));
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected lifted ivalue type");
    }
  }
}

static TraceState call_begin_capture(
    PyObject* self,
    CacheNode& cache,
    AutogradCompilerCall& compiler_call,
    size_t num_outputs) {
  static PyObject* method_name = PyUnicode_InternFromString("begin_capture");
  THPObjectPtr pyinput(THPVariable_WrapList(compiler_call.tensor_args.inputs));
  THPObjectPtr pysizeinput(cache.wrap_dynamic_inputs());
  THPObjectPtr pyivalueargsinput(
      wrap_lifted_ivalue_args(compiler_call.lifted_ivalue_args.args));
  THPObjectPtr pynodeorigins(
      wrap_node_origins(compiler_call, PyTuple_GET_SIZE(pysizeinput.get())));
  THPObjectPtr pyresult(check(PyObject_CallMethodObjArgs(
      self,
      method_name,
      pyinput.get(),
      pysizeinput.get(),
      pyivalueargsinput.get(),
      pynodeorigins.get(),
      nullptr)));

  PyObject *fake_inputs{nullptr}, *fake_sizes{nullptr},
      *fake_ivalue_args{nullptr};
  check(PyArg_ParseTuple(
      pyresult.get(), "OOO", &fake_inputs, &fake_sizes, &fake_ivalue_args));

  variable_list proxy_inputs = THPVariable_UnpackList(fake_inputs);
  TORCH_INTERNAL_ASSERT(
      proxy_inputs.size() == compiler_call.tensor_args.inputs.size());
  for (const auto i : c10::irange(proxy_inputs.size())) {
    TensorArg& arg =
        compiler_call.tensor_args.lookup(compiler_call.tensor_args.inputs[i]);
    arg.proxy_tensor = proxy_inputs[i];
  }

  set_ivalue_proxies(fake_ivalue_args, compiler_call.lifted_ivalue_args.args);
  return TraceState(cache.unwrap_dynamic_inputs(fake_sizes), num_outputs);
}

static PyObject* call_end_capture(PyObject* self, const variable_list& inputs) {
  static PyObject* method_name = PyUnicode_InternFromString("end_capture");
  THPObjectPtr pyinput(THPVariable_WrapList(inputs));
  return check(
      PyObject_CallMethodObjArgs(self, method_name, pyinput.get(), nullptr));
}

struct ClosingTHPObjectPtr : public THPObjectPtr {
  ClosingTHPObjectPtr(PyObject* o) : THPObjectPtr(o) {}
  ClosingTHPObjectPtr(ClosingTHPObjectPtr&& other) = default;
  ClosingTHPObjectPtr(const ClosingTHPObjectPtr&) = delete;
  ClosingTHPObjectPtr& operator=(const ClosingTHPObjectPtr&) = delete;
  ClosingTHPObjectPtr& operator=(ClosingTHPObjectPtr&&) = default;
  ~ClosingTHPObjectPtr() {
    if (PyErr_Occurred()) {
      // do nothing, do not attempt to close
      return;
    }
    static PyObject* method_name = PyUnicode_InternFromString("close");
    if (PyObject_CallMethodObjArgs(get(), method_name, nullptr) == nullptr) {
      PyErr_WriteUnraisable(get());
      PyErr_Clear();
    }
  }
};

// Only call this function while holding GIL
CacheNode* _compiled_autograd_impl(
    const std::shared_ptr<Node>& graph_root,
    GraphTask& graph_task,
    bool accumulate_grad,
    const edge_list& output_edges,
    THPObjectPtr* graph_arg_inputs,
    THPObjectPtr* graph_arg_sizes,
    THPObjectPtr* graph_arg_ivalue_args,
    THPObjectPtr* graph_arg_hooks) {
  std::unordered_map<Node*, int>& dependencies = graph_task.dependencies_;
  std::vector<std::shared_ptr<Node>> worklist{graph_root};
  AutogradCompilerCall compiler_call;

  for (const auto i : c10::irange(output_edges.size())) {
    compiler_call.node_calls
        .lookup(output_edges[i].function)
        // NOLINTNEXTLINE(*-narrowing-conversions)
        .mark_output(output_edges[i].input_nr, i);
  }
  const bool check_exec_info = !graph_task.exec_info_.empty();
  CacheNode* cache = CacheNode::root();
  std::vector<NodeCall*> calls;
  calls.reserve(
      check_exec_info ? graph_task.exec_info_.size() : dependencies.size() + 1);

  int i = 0;
  std::optional<VerboseLogger> vlogger = VerboseLogger::maybe_create();
  while (!worklist.empty()) {
    std::shared_ptr<Node> fn = std::move(worklist.back());
    worklist.pop_back();
    NodeCall& call = compiler_call.node_calls.lookup(fn);
    calls.emplace_back(&call);

    { // update cache and gather args into `compiler_call`
      CompiledNodeArgs node_args(compiler_call, call);
      node_args.collect(call);
      if (vlogger.has_value()) {
        compiler_call.set_active_node_call_idx(i);
      }
      if (node_args.cond(call.needed)) {
        fn->compiled_args(node_args);
        node_args.collect(call.node->next_edges());
      }
      CacheKey key = node_args.key();
      if (vlogger.has_value()) {
        std::unordered_set<CacheKey> cached_keys;
        for (const auto& [k, _] : cache->next) {
          cached_keys.emplace(k);
        }
        vlogger->log_node_check(
            *fn,
            compiler_call.all_size_inputs.size(),
            std::move(cached_keys),
            key,
            i);
      }
      cache = cache->lookup(key);
    }

    for (const auto& edge : fn->next_edges()) {
      if (!edge.is_valid()) {
        continue;
      }
      if (check_exec_info) {
        auto it = graph_task.exec_info_.find(edge.function.get());
        if (it == graph_task.exec_info_.end() || !it->second.should_execute()) {
          continue;
        }
        if (!it->second.needed_) {
          compiler_call.node_calls.lookup(edge.function).needed = false;
        }
      }
      auto it = dependencies.find(edge.function.get());
      TORCH_INTERNAL_ASSERT(it != dependencies.end());
      if (--it->second == 0) {
        dependencies.erase(it);
        worklist.emplace_back(edge.function);
      }
    }
    i++;
  }

  // TODO(jansel): some dynamic sizes seem to be ints not symints
  if (!cache->check_dynamic_sizes(compiler_call, vlogger)) {
    // cache miss, need to capture FX graph
    ClosingTHPObjectPtr py_compiler(
        check(PyObject_CallNoArgs((the_autograd_compiler))));

    TraceState state = call_begin_capture(
        py_compiler, *cache, compiler_call, output_edges.size());
    InputBuffers input_buffers;

    for (size_t i = 0; i < calls.size(); i++) {
      NodeCall& call = *calls[i];

      std::string _node_name = call.node->name();
      THPObjectPtr node_name(PyUnicode_FromString(_node_name.data()));
      TORCH_INTERNAL_ASSERT(node_name != nullptr);
      THPObjectPtr set_node_origin(
          PyObject_GetAttrString(py_compiler.get(), "set_node_origin"));
      PyObject* pyobj = Py_None;
      if (auto pynode = std::dynamic_pointer_cast<PyNode>(call.node)) {
        pyobj = pynode->obj;
      }
      check(PyObject_CallFunction(
          set_node_origin, "OIO", node_name.get(), i, pyobj, nullptr));

      // TODO(jansel): consider adding some of this stuff:
      // guard(local_graph_task); NodeGuard ndguard(task.fn_); const auto
      // opt_parent_stream = (*func).stream(c10::DeviceType::CUDA);
      // c10::OptionalStreamGuard parent_stream_guard{opt_parent_stream};
      // CheckpointValidGuard cpvguard(graph_task);
      // at::getStepCallbacksUnlessEmpty(at::RecordScope::BACKWARD_FUNCTION);
      // if (C10_UNLIKELY(step_callbacks.has_value())) { ... }

      variable_list inputs =
          std::move(input_buffers.lookup(call.node.get()).buffer);
      input_buffers.erase(call.node.get());

      if (!call.tensor_pre_hooks.empty()) {
        THPObjectPtr pyinputs(THPVariable_WrapList(inputs));
        for (const auto& hook : call.tensor_pre_hooks) {
          pyinputs = check(PyObject_CallMethod(
              py_compiler,
              "tensor_pre_hook",
              "Oii",
              pyinputs.get(),
              hook.first,
              hook.second));
        }
        inputs = THPVariable_UnpackList(pyinputs);
      }
      for (const auto& graph_output : call.graph_output) {
        int input_nr = graph_output.first;
        int output_index = graph_output.second;
        TORCH_INTERNAL_ASSERT(
            output_index < static_cast<int>(state.outputs.size()));
        TORCH_INTERNAL_ASSERT(!state.outputs[output_index].defined());
        state.outputs[output_index] = inputs[input_nr];
      }
      if (!call.needed) {
        continue;
      }
      if (!call.pre_hooks.empty()) {
        THPObjectPtr pyinputs(THPVariable_WrapList(inputs));
        for (const auto hook : call.pre_hooks) {
          pyinputs = check(PyObject_CallMethod(
              py_compiler.get(), "pre_hook", "Oi", pyinputs.get(), hook));
        }
        inputs = THPVariable_UnpackList(pyinputs);
      }

      SwapSavedVariables saved(compiler_call, state, py_compiler.get(), call);
      variable_list outputs = call.node->apply_with_saved(inputs, saved);

      saved.debug_asserts();
      saved.before(call.node->next_edges());
      validate_outputs(
          call.node->next_edges(), outputs, [&](const std::string& msg) {
            std::ostringstream ss;
            ss << "[Compiled Autograd Tracing: " << call.node->name() << "] "
               << msg;
            return ss.str();
          });
      saved.after(call.node->next_edges());
      saved.debug_asserts();

      if (!call.post_hooks.empty()) {
        THPObjectPtr pyinputs(THPVariable_WrapList(inputs));
        THPObjectPtr pyoutputs(THPVariable_WrapList(outputs));
        for (const auto hook : call.post_hooks) {
          pyoutputs = check(PyObject_CallMethod(
              py_compiler.get(),
              "post_hook",
              "OOi",
              pyoutputs.get(),
              pyinputs.get(),
              hook));
        }
        outputs = THPVariable_UnpackList(pyoutputs);
      }
      for (const auto i : c10::irange(outputs.size())) {
        auto& output = outputs[i];
        const auto& next = call.node->next_edge(i);
        if (next.is_valid() && output.defined()) {
          input_buffers.lookup(next.function.get())
              .add(
                  next.input_nr, std::move(output), std::nullopt, std::nullopt);
        }
      }
    }

    PyObject* res = check(call_end_capture(py_compiler, state.outputs));
    TORCH_CHECK(PyTuple_Check(res), "Expected end_capture to return tuple");
    TORCH_CHECK(
        PyTuple_Size(res) == 2,
        "Expected end_capture to return tuple of size 2");
    cache->runtime_wrapper = Py_NewRef(PyTuple_GetItem(res, 0));
    TORCH_CHECK(
        PyCallable_Check(cache->runtime_wrapper),
        "Expected end_capture to return runtime_wrapper");
    cache->compiled_fn = Py_NewRef(PyTuple_GetItem(res, 1));
    TORCH_CHECK(
        PyCallable_Check(cache->compiled_fn),
        "Expected end_capture to return compiled_fn");
    state.debug_asserts();
  } // End cache miss region

  // TODO(jansel): clear grads we will overwrite below
  if (!graph_task.keep_graph_) {
    for (auto& call : calls) {
      call->node->release_variables();
    }
  }

  *graph_arg_inputs = THPVariable_WrapList(compiler_call.tensor_args.inputs);
  *graph_arg_sizes = wrap_int_list(compiler_call.dyn_size_inputs);
  *graph_arg_ivalue_args =
      wrap_lifted_ivalue_args(compiler_call.lifted_ivalue_args.args);
  *graph_arg_hooks = convert_hook_list(compiler_call.hooks);
  return cache;
}

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct LockGuardWithErrorLogs {
  LockGuardWithErrorLogs(std::mutex& mtx) : mtx_(mtx) {
    // Note: the standard allows try_lock to fail spuriously during races for
    // performance reasons, but it shouldn't happen here since we:
    // 1. disable multithreaded autograd
    // 2. plenty of latency between backward calls
    TORCH_INTERNAL_ASSERT(
        mtx_.try_lock(),
        "Trying to run compiled autograd within another compiled autograd call (e.g. reentrant checkpointing), this is not supported yet.");
  }

  ~LockGuardWithErrorLogs() {
    mtx_.unlock();
  }

  std::mutex& mtx_;
};

variable_list compiled_autograd(
    const std::shared_ptr<Node>& graph_root,
    GraphTask& graph_task,
    bool accumulate_grad,
    const edge_list& output_edges) {
  TORCH_CHECK(
      c10::impl::TorchDispatchModeTLS::stack_len() == 0,
      "TorchDispatchMode not yet implemented for compiled autograd")
  static std::mutex mtx;
  LockGuardWithErrorLogs lock_guard(mtx);
  pybind11::gil_scoped_acquire gil;
  at::ThreadLocalStateGuard tls_guard(graph_task.thread_locals_);

  THPObjectPtr inputs;
  THPObjectPtr sizes;
  THPObjectPtr ivalue_args;
  THPObjectPtr hooks;
  CacheNode* cache = _compiled_autograd_impl(
      graph_root,
      graph_task,
      accumulate_grad,
      output_edges,
      &inputs,
      &sizes,
      &ivalue_args,
      &hooks);

  THPObjectPtr pyresult(check(PyObject_CallFunctionObjArgs(
      cache->runtime_wrapper.get(),
      cache->compiled_fn.get(),
      inputs.get(),
      sizes.get(),
      ivalue_args.get(),
      hooks.get(),
      NULL)));
  variable_list outputs = THPVariable_UnpackList(pyresult);
  TORCH_INTERNAL_ASSERT(outputs.size() == output_edges.size());
  return outputs;
}

static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS;
  PyObject* obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }

  PyObject* prior = the_autograd_compiler;
  if (obj == Py_None) { // disable
    the_autograd_compiler = nullptr; // decref not needed due to `prior`
    Engine::set_compiled_autograd(nullptr);
  } else { // enable
    Py_INCREF(obj);
    the_autograd_compiler = obj;
    Engine::set_compiled_autograd(&compiled_autograd);
  }

  if (prior == nullptr) {
    Py_RETURN_NONE;
  } else {
    return prior;
  }
  END_HANDLE_TH_ERRORS;
}

PyObject* torch_c_dynamo_compiled_autograd_init() {
  PyObject* mod = PyModule_Create(&_module);
  if (mod == nullptr) {
    return nullptr;
  }

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(mod, Py_MOD_GIL_NOT_USED);
#endif
  return mod;
}

} // namespace torch::dynamo::autograd
