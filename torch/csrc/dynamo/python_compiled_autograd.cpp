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

namespace {
PyObject* the_autograd_compiler = nullptr;
int default_dyn_type_int = 0;
PyObject* python_verbose_logger = nullptr;
} // namespace

// see https://github.com/pytorch/pytorch/pull/34845
static void throw_python_error() {
  python_error err;
  err.persist();
  throw std::move(err);
}

// RuntimeState contains arbitrary callables created during the forward pass.
// e.g. .retains_grad(). It is created during the compiled_args stage, and is
// used at runtime.  The lifetime of RuntimeState is a single backward pass.
struct RuntimeState {
  at::TensorBase call_cpp_tensor_pre_hooks(
      size_t idx,
      const at::TensorBase& grad) {
    TORCH_INTERNAL_ASSERT(
        cpp_tensor_pre_hooks.size() > static_cast<size_t>(idx));
    return cpp_tensor_pre_hooks[idx](grad);
  }

  std::vector<std::function<at::TensorBase(const at::TensorBase&)>>
      cpp_tensor_pre_hooks;
  size_t next_id = 0;
};

static RuntimeState* active_rstate;
struct RuntimeStateGuard {
  RuntimeStateGuard() : _state(std::make_unique<RuntimeState>()) {
    active_rstate = _state.get();
  }
  RuntimeStateGuard(const RuntimeStateGuard&) = delete;
  RuntimeStateGuard& operator=(const RuntimeStateGuard&) = delete;
  RuntimeStateGuard(RuntimeStateGuard&&) = delete;
  RuntimeStateGuard& operator=(RuntimeStateGuard&&) = delete;

  ~RuntimeStateGuard() {
    active_rstate = nullptr;
  }

  std::unique_ptr<RuntimeState> _state;
};

static PyObject* call_cpp_tensor_pre_hooks(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS;
  int idx = -1;
  PyObject* grad = nullptr;
  if (!PyArg_ParseTuple(args, "iO", &idx, &grad)) {
    throw_python_error();
  }
  TORCH_INTERNAL_ASSERT(idx > -1);
  TORCH_INTERNAL_ASSERT(grad != nullptr);
  TORCH_INTERNAL_ASSERT(active_rstate != nullptr);
  auto res = active_rstate->call_cpp_tensor_pre_hooks(
      static_cast<size_t>(idx), THPVariable_Unpack(grad));
  return THPVariable_Wrap(res);
  END_HANDLE_TH_ERRORS;
}

// List[Optional[Tensor]] in Python can't be directly parsed into a
// List[Tensor], so we need to do this conversion manually.
static std::vector<at::Tensor> toTensorList(
    const std::vector<std::optional<at::Tensor>>& inputs) {
  std::vector<at::Tensor> result;
  result.reserve(inputs.size());
  for (const auto& inp : inputs) {
    if (inp.has_value()) {
      result.emplace_back(*inp);
    } else {
      result.emplace_back();
    }
  }
  return result;
}

// Binds a function (that represents some backward computation) to Python.
// All of these functions have a common signature, which is
// (in C++) (vector<Tensor>, vector<ivalue>) -> vector<Tensor>
// (in Python) (List[Optional[Tensor]], *packed_args: IValue) ->
// List[Optional[Tensor]]
//
// The vector<Tensor> are the list of gradient Tensors, each of which may be
// undefined (in C++) which corresponds to None (in Python).
static std::string bind_function(
    PyObject* py_compiler,
    const std::string& fn_name,
    functional_apply_t fn,
    std::vector<at::TypePtr> packed_args_schema,
    bool is_custom_function,
    bool is_traceable) {
  // This is the function that can be called from Python.
  auto py_func = py::cpp_function(
      [packed_args_schema = std::move(packed_args_schema), fn = std::move(fn)](
          std::vector<std::optional<at::Tensor>>& inputs,
          const py::args& py_args) -> py::object {
        // py_args is a tuple of PyObject*.
        // We need to reconstruct a vector<IValue> to invoke `fn`.
        // To do so, we use the packed_args_schema to convert each PyObject*
        // to its corresponding C++ type that can be stored into IValue.
        TORCH_INTERNAL_ASSERT(py_args.size() == packed_args_schema.size());
        std::vector<at::IValue> args;
        args.reserve(py_args.size());
        auto tuple_args = jit::tuple_slice(py_args);
        for (uint64_t idx = 0; idx < packed_args_schema.size(); idx++) {
          if (packed_args_schema[idx]->isSubtypeOf(
                  *at::ListType::ofTensors())) {
            // List[Tensor] might have Nones, not handled in jit::toIValue
            auto tmp = py::cast<std::vector<std::optional<at::Tensor>>>(
                tuple_args[idx]);
            args.emplace_back(toTensorList(tmp));
          } else {
            args.emplace_back(jit::toIValue(
                tuple_args[idx], packed_args_schema[idx], std::nullopt));
          }
        }
        // None in Python corresponds to undefined Tensor in C++
        auto inputs_ = toTensorList(inputs);
        auto outputs = fn(inputs_, args);
        return jit::toPyObject(at::IValue(outputs));
      });
  py::handle handle(py_compiler);
  auto result = handle.attr("bind_function")(
      fn_name, py_func, is_custom_function, is_traceable);
  return result.cast<std::string>();
}

// Invokes py_compiler.method_name(fn_name, inputs, packed_args,
// output_metadata)
static variable_list call_function(
    PyObject* py_compiler,
    const char* method_name,
    const std::string& fn_name,
    const variable_list& inputs,
    const ivalue_list& packed_args,
    const c10::IValue& output_metadata) {
  // convert ivalue_list -> PyObject*
  PyObject* py_packed_args =
      PyTuple_New(static_cast<Py_ssize_t>(packed_args.size()));
  for (const auto i : c10::irange(packed_args.size())) {
    py::object obj = jit::toPyObject(packed_args[i]);
    Py_INCREF(obj.ptr());
    PyTuple_SET_ITEM(py_packed_args, i, obj.ptr());
  }

  // call the corresponding method on the py_compiler
  py::handle handle(py_compiler);
  py::object stuff = handle.attr(method_name)(
      fn_name,
      inputs,
      py::handle(py_packed_args),
      jit::toPyObject(output_metadata));

  // Convert the output from PyObject* to vector<Tensor>
  auto tmp = py::cast<std::vector<std::optional<at::Tensor>>>(std::move(stuff));
  return toTensorList(tmp);
}

struct PyCompilerInterfaceImpl : PyCompilerInterface {
  std::string bind_function(
      PyObject* py_compiler,
      const std::string& fn_name,
      functional_apply_t fn,
      std::vector<at::TypePtr> packed_args_schema,
      bool is_custom_function = false,
      bool is_traceable = true) const override {
    return torch::dynamo::autograd::bind_function(
        py_compiler,
        fn_name,
        std::move(fn),
        std::move(packed_args_schema),
        is_custom_function,
        is_traceable);
  }
  variable_list call_function(
      PyObject* py_compiler,
      const char* method_name,
      const std::string& fn_name,
      const variable_list& inputs,
      const ivalue_list& packed_args,
      const c10::IValue& output_metadata) const override {
    return torch::dynamo::autograd::call_function(
        py_compiler,
        method_name,
        fn_name,
        inputs,
        packed_args,
        output_metadata);
  }
  variable_list call_copy_slices_prologue(
      PyObject* py_compiler,
      const variable_list& inputs,
      const at::TensorGeometry& base,
      const at::TensorGeometry& view) const override {
    py::handle handle(py_compiler);
    py::object stuff = handle.attr("call_copy_slices_prologue")(
        inputs,
        base.sym_sizes(),
        base.sym_strides(),
        base.sym_storage_offset(),
        view.sym_sizes(),
        view.sym_strides(),
        view.sym_storage_offset());
    return py::cast<std::vector<at::Tensor>>(std::move(stuff));
  }
  variable_list call_copy_slices_epilogue(
      PyObject* py_compiler,
      const std::vector<bool>& needs_input_grad,
      const at::Tensor& result,
      const variable_list& res,
      const at::Tensor& grad_slice) const override {
    py::handle handle(py_compiler);
    py::object stuff = handle.attr("call_copy_slices_epilogue")(
        needs_input_grad, result, res, grad_slice);
    auto output =
        py::cast<std::vector<std::optional<at::Tensor>>>(std::move(stuff));
    return toTensorList(output);
  }
  at::Tensor call_unpack(
      PyObject* py_compiler,
      std::optional<size_t> hook_id,
      size_t hook_input_id) const override {
    py::handle handle(py_compiler);
    py::object proxy = handle.attr("unpack_hook")(hook_id, hook_input_id);
    auto tmp = py::cast<std::optional<at::Tensor>>(std::move(proxy));
    TORCH_INTERNAL_ASSERT(tmp.has_value());
    return tmp.value();
  }
  void call_accumulate_grad(
      PyObject* py_compiler,
      const at::Tensor& variable,
      const at::Tensor& grad) const override {
    py::handle handle(py_compiler);
    py::object stuff = handle.attr("accumulate_grad")(variable, grad);
    TORCH_INTERNAL_ASSERT(stuff.is_none());
  }
};

static PyObject* wrap_int_list(const std::vector<int64_t>& inputs) {
  PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(inputs.size()));
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, PyLong_FromSsize_t(inputs[i]));
  }
  return pyinput;
}

static PyObject* convert_pyobj_list(std::vector<c10::SafePyObject>& inputs) {
  // inplace, consumes the input hooks
  PyObject* pyinput = PyTuple_New(static_cast<Py_ssize_t>(inputs.size()));
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, inputs[i].release());
  }
  return pyinput;
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

static variable_list validate_outputs(
    const variable_list& outputs,
    const ivalue_list& args) {
  auto r = PackedArgs(args);
  auto value = r.unpack<std::vector<std::optional<InputMetadata>>>();
  auto new_outputs = outputs;

  torch::autograd::validate_outputs(
      value, new_outputs, [&](const std::string& msg) {
        std::ostringstream ss;
        ss << "[Compiled Autograd Tracing:]" << msg;
        return ss.str();
      });
  return new_outputs;
}

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
    THPObjectPtr pymethod(PyUnicode_FromString(levelNames_[level]));
    TORCH_INTERNAL_ASSERT(pymethod != nullptr);
    THPObjectPtr pyfunc(PyObject_GetAttr(logger_, pymethod.get()));
    if (pyfunc == nullptr) {
      throw_python_error();
    }
    PyObject* result =
        PyObject_CallFunction(pyfunc.get(), "s", std::string(msg).c_str());
    if (result == nullptr) {
      throw_python_error();
    }
  }

 private:
  static constexpr std::array<const char*, COUNT> levelNames_ = {
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

  std::string log_node_check(
      const Node& fn,
      size_t size_inputs_num,
      const std::unordered_set<CacheKey>& cached_keys,
      const CacheKey& key,
      size_t node_idx) {
    std::string node_name =
        fn.name() + " (NodeCall " + std::to_string(node_idx) + ")";
    return _log_node_miss(typeid(fn), cached_keys, key, node_name);
  }

  std::string _log_node_miss(
      const std::type_info& node_type,
      const std::unordered_set<CacheKey>& cached_keys,
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
    std::string compile_reason = oss.str();
    log(PythonLogger::DEBUG, compile_reason);
    return compile_reason;
  }

  std::string log_dynamic_shapes_miss(
      const std::vector<size_t>& new_dyn_sizes_idx,
      size_t all_dyn_sizes_len) const {
    std::ostringstream oss;
    oss << "Cache miss due to " << new_dyn_sizes_idx.size()
        << " changed tensor shapes (total of " << all_dyn_sizes_len << "): ";
    for (const auto i : c10::irange(new_dyn_sizes_idx.size() - 1)) {
      oss << "sizes[" << std::to_string(new_dyn_sizes_idx[i]) << "], ";
    }
    oss << "sizes["
        << std::to_string(new_dyn_sizes_idx[new_dyn_sizes_idx.size() - 1])
        << "]";
    std::string recompile_reason = oss.str();
    log(PythonLogger::DEBUG, recompile_reason);
    return recompile_reason;
  }
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
    compile_reasons.clear();
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
      std::optional<std::string>& compile_reason,
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
    std::vector<size_t> newly_dynamic;
    for (const auto i : c10::irange(len)) {
      auto& expected = expected_sizes[i];
      bool was_dynamic = expected.dyn_type == SizeInput::DYNAMIC;
      bool changed_value = expected.value != data[i].value;
      if (changed_value) {
        if (!was_dynamic) {
          cache_hit = false;
          if (vlogger.has_value()) {
            newly_dynamic.emplace_back(call.dyn_size_inputs.size());
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
      if (vlogger.has_value() && !newly_dynamic.empty()) {
        // some shapes became dynamic, recompile
        TORCH_INTERNAL_ASSERT(!compile_reason.has_value());
        compile_reason = vlogger->log_dynamic_shapes_miss(
            newly_dynamic, call.dyn_size_inputs.size());
      }
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
  std::vector<std::string> compile_reasons;

  THPObjectPtr runtime_wrapper;
  THPObjectPtr compiled_fn;
};

struct InputBuffers : public std::unordered_map<Node*, InputBuffer> {
  InputBuffer& lookup(Node* function) {
    auto it = emplace(function, InputBuffer(function->num_inputs())).first;
    return it->second;
  }
};

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
    {"call_cpp_tensor_pre_hooks",
     call_cpp_tensor_pre_hooks,
     METH_VARARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.autograd_compiler",
    "Hooks for compiling autograd",
    -1,
    _methods};

static PyObject* wrap_lifted_ivalue_args(
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

static PyObject* wrap_node_origins(
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

static PyObject* wrap_string_list(const std::vector<std::string>& strs) {
  PyObject* pystrs = PyList_New(static_cast<Py_ssize_t>(strs.size()));
  for (const auto i : c10::irange(strs.size())) {
    PyObject* pystr = PyUnicode_FromString(strs[i].c_str());
    PyList_SET_ITEM(pystrs, i, pystr);
  }
  return pystrs;
}

static std::string unwrap_string(PyObject* pystr) {
  TORCH_INTERNAL_ASSERT(PyUnicode_Check(pystr));
  const char* str = PyUnicode_AsUTF8(pystr);
  TORCH_INTERNAL_ASSERT(str != nullptr);
  return std::string(str);
}

static void set_ivalue_proxies(
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

static at::Tensor call_accumulate(
    PyObject* py_compiler,
    const at::Tensor& old_var,
    const at::Tensor& new_var) {
  if (!old_var.defined()) {
    return new_var;
  }
  if (!new_var.defined()) {
    return old_var;
  }
  py::handle handle(py_compiler);
  py::object stuff = handle.attr("accumulate")(old_var, new_var);
  return py::cast<at::Tensor>(std::move(stuff));
}

static TraceState call_begin_capture(
    PyObject* self,
    CacheNode& cache,
    AutogradCompilerCall& compiler_call,
    size_t num_outputs,
    std::optional<std::string>&& maybe_compile_reason,
    bool accumulate_grad,
    bool check_nans) {
  static PyObject* method_name = PyUnicode_InternFromString("begin_capture");
  THPObjectPtr py_input(THPVariable_WrapList(compiler_call.tensor_args.inputs));
  THPObjectPtr py_size_input(cache.wrap_dynamic_inputs());
  THPObjectPtr py_ivalue_args_input(
      wrap_lifted_ivalue_args(compiler_call.lifted_ivalue_args.args));
  THPObjectPtr py_node_origins(
      wrap_node_origins(compiler_call, PyTuple_GET_SIZE(py_size_input.get())));
  THPObjectPtr pyresult(check(PyObject_CallMethodObjArgs(
      self,
      method_name,
      py_input.get(),
      py_size_input.get(),
      py_ivalue_args_input.get(),
      py_node_origins.get(),
      PyBool_FromLong(accumulate_grad),
      PyBool_FromLong(check_nans),
      nullptr)));

  PyObject *compile_id_str{nullptr}, *fake_inputs{nullptr},
      *fake_sizes{nullptr}, *fake_ivalue_args{nullptr};
  check(PyArg_ParseTuple(
      pyresult.get(),
      "OOOO",
      &compile_id_str,
      &fake_inputs,
      &fake_sizes,
      &fake_ivalue_args));

  variable_list proxy_inputs = THPVariable_UnpackList(fake_inputs);
  TORCH_INTERNAL_ASSERT(
      proxy_inputs.size() == compiler_call.tensor_args.inputs.size());
  for (const auto i : c10::irange(proxy_inputs.size())) {
    TensorArg& arg =
        compiler_call.tensor_args.lookup(compiler_call.tensor_args.inputs[i]);
    arg.proxy_tensor = proxy_inputs[i];
  }

  set_ivalue_proxies(fake_ivalue_args, compiler_call.lifted_ivalue_args.args);
  if (auto compile_reason = std::move(maybe_compile_reason);
      compile_reason.has_value()) {
    TORCH_INTERNAL_ASSERT(!Py_IsNone(compile_id_str));
    std::string formatted_compile_reason = unwrap_string(compile_id_str) +
        ": " + std::move(compile_reason.value());
    cache.compile_reasons.emplace_back(formatted_compile_reason);
    THPObjectPtr py_compile_reasons(wrap_string_list(cache.compile_reasons));
    static PyObject* log_compile_reasons =
        PyUnicode_InternFromString("log_compile_reasons");
    check(PyObject_CallMethodObjArgs(
        self, log_compile_reasons, py_compile_reasons.get(), nullptr));
  }
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

static SizeInput::DynType get_default_dyn_type() {
  TORCH_INTERNAL_ASSERT(default_dyn_type_int >= 0 && default_dyn_type_int < 2);
  return default_dyn_type_int == 0 ? SizeInput::STATIC : SizeInput::DYNAMIC;
}

// Only call this function while holding GIL
static CacheNode* _compiled_autograd_impl(
    const std::shared_ptr<Node>& graph_root,
    const GraphTask& graph_task,
    bool accumulate_grad,
    const edge_list& output_edges,
    THPObjectPtr* graph_arg_inputs,
    THPObjectPtr* graph_arg_sizes,
    THPObjectPtr* graph_arg_ivalue_args,
    THPObjectPtr* graph_arg_hooks,
    THPObjectPtr* graph_arg_packed_inputs,
    RuntimeState* rstate) {
  const std::unordered_map<Node*, int>& dependencies = graph_task.dependencies_;
  std::unordered_map<Node*, int> visited_dependencies;
  visited_dependencies.reserve(dependencies.size());

  std::vector<std::shared_ptr<Node>> worklist{graph_root};
  AutogradCompilerCall compiler_call(get_default_dyn_type());

  for (const auto i : c10::irange(output_edges.size())) {
    compiler_call.node_calls
        .lookup(output_edges[i].function)
        // NOLINTNEXTLINE(*-narrowing-conversions)
        .mark_output(output_edges[i].input_nr, i);
  }
  const bool check_exec_info = !graph_task.exec_info_.empty();
  CacheNode* cache = CacheNode::root();
  std::vector<NodeCall*> ordered_calls;
  ordered_calls.reserve(
      check_exec_info ? graph_task.exec_info_.size() : dependencies.size() + 1);

  int i = 0;
  std::optional<VerboseLogger> vlogger = VerboseLogger::maybe_create();
  std::optional<std::string> compile_reason;
  while (!worklist.empty()) {
    std::shared_ptr<Node> fn = std::move(worklist.back());
    worklist.pop_back();
    NodeCall& call = compiler_call.node_calls.lookup(fn);
    ordered_calls.emplace_back(&call);

    { // update cache and gather args into `compiler_call`
      CompiledNodeArgs node_args(compiler_call, call);
      if (vlogger.has_value()) {
        compiler_call.set_active_node_call_idx(i);
      }
      node_args.collect(call);
      if (node_args.cond(call.needed)) {
        fn->compiled_args(node_args);
        node_args.collect(call.node->next_edges());
      }
      CacheKey key = node_args.key();
      if (vlogger.has_value() && !compile_reason.has_value()) {
        std::unordered_set<CacheKey> cached_keys;
        for (const auto& [k, _] : cache->next) {
          cached_keys.emplace(k);
        }
        if (cached_keys.find(key) == cached_keys.end()) {
          // new autograd node found, compile
          compile_reason = vlogger->log_node_check(
              *fn, compiler_call.all_size_inputs.size(), cached_keys, key, i);
        }
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
      int count = ++visited_dependencies[it->first];
      TORCH_INTERNAL_ASSERT(count <= it->second);
      if (count == it->second) {
        worklist.emplace_back(edge.function);
      }
    }
    i++;
  }

  // TODO(jansel): some dynamic sizes seem to be ints not symints
  if (!cache->check_dynamic_sizes(compiler_call, compile_reason, vlogger)) {
    // cache miss, need to capture FX graph
    TORCH_INTERNAL_ASSERT(!vlogger.has_value() || compile_reason.has_value());
    ClosingTHPObjectPtr py_compiler(
        check(PyObject_CallNoArgs((the_autograd_compiler))));
    PyCompilerGuard py_compiler_guard(
        std::make_unique<PyCompilerInterfaceImpl>());

    TraceState state = call_begin_capture(
        py_compiler,
        *cache,
        compiler_call,
        output_edges.size(),
        std::move(compile_reason),
        accumulate_grad,
        AnomalyMode::is_enabled() && AnomalyMode::should_check_nan());
    InputBuffers input_buffers;

    for (size_t i = 0; i < ordered_calls.size(); i++) {
      NodeCall& call = *ordered_calls[i];

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
      if (!call.cpp_tensor_pre_hooks.empty()) {
        // proxy a call to runtimestate
        THPObjectPtr pyinputs(THPVariable_WrapList(inputs));
        for (const auto& [hook_id, idx] : call.cpp_tensor_pre_hooks) {
          pyinputs = check(PyObject_CallMethod(
              py_compiler,
              "cpp_tensor_pre_hook",
              "Oii",
              pyinputs.get(),
              hook_id,
              idx));
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

      auto input_metadata = get_input_metadata(call.node->next_edges());
      TORCH_INTERNAL_ASSERT(input_metadata.size() == outputs.size());

      // Lazily bind the `validate_outputs` function to Python.
      static bool flag [[maybe_unused]] = [&]() {
        auto schema = std::vector<at::TypePtr>{IValuePacker<
            std::vector<std::optional<InputMetadata>>>::packed_type()};
        bind_function(
            py_compiler.get(),
            "validate_outputs",
            validate_outputs,
            schema,
            /*is_custom_function=*/false,
            /*is_traceable=*/true);
        return true;
      }();

      // Don't emit validate_outputs nodes that follow a CompiledBackward node.
      // These nodes would otherwise prevent reordering of accumulate_grad
      // nodes.
      //
      // Note that this will not cause correctness issues, because
      // 1) AOTAutograd already coerces gradients to have the same metadata as
      // the inputs. 2) the AOTAutograd graph already has the necessary
      // aten::sum_to nodes in it (so it doesn't need to rely on
      // validate_outputs to handle that).
      //
      // However, we may be dropping some (edge case) safety checks compared to
      // eager: a backward that would have errored out in eager may not error
      // out in compiled autograd (for example, if the user provided an
      // incorrect number of gradients).
      if (!call.node->is_aot_backward()) {
        PackedArgs args;
        args.pack(input_metadata);
        ivalue_list input_metadata_state = std::move(args).vec();
        outputs = call_function(
            py_compiler,
            "validate_outputs",
            "validate_outputs",
            outputs,
            input_metadata_state,
            input_metadata_state[0]);
      }

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
          auto& buffer = input_buffers.lookup(next.function.get());
          buffer.buffer[next.input_nr] = call_accumulate(
              py_compiler, buffer.buffer[next.input_nr], output);
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
    for (auto& call : ordered_calls) {
      // Once we release variables, we can no longer fallback to eager autograd
      call->node->release_variables();
    }
  }

  *graph_arg_inputs = THPVariable_WrapList(compiler_call.tensor_args.inputs);
  *graph_arg_sizes = wrap_int_list(compiler_call.dyn_size_inputs);
  *graph_arg_ivalue_args =
      wrap_lifted_ivalue_args(compiler_call.lifted_ivalue_args.args);
  *graph_arg_hooks = convert_pyobj_list(compiler_call.hooks);
  *graph_arg_packed_inputs = convert_pyobj_list(compiler_call.packed_inputs);
  rstate->cpp_tensor_pre_hooks = std::move(compiler_call.cpp_tensor_pre_hooks);
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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::mutex& mtx_;
};

static variable_list compiled_autograd(
    const std::shared_ptr<Node>& graph_root,
    const GraphTask& graph_task,
    bool accumulate_grad,
    const edge_list& output_edges) {
  TORCH_CHECK(
      c10::impl::TorchDispatchModeTLS::stack_len() == 0,
      "TorchDispatchMode not yet implemented for compiled autograd")
  static std::mutex mtx;
  LockGuardWithErrorLogs lock_guard(mtx);
  pybind11::gil_scoped_acquire gil;
  at::ThreadLocalStateGuard tls_guard(graph_task.thread_locals_);
  RuntimeStateGuard rstate_guard;

  THPObjectPtr inputs;
  THPObjectPtr sizes;
  THPObjectPtr ivalue_args;
  THPObjectPtr hooks;
  THPObjectPtr packed_inputs;
  CacheNode* cache = _compiled_autograd_impl(
      graph_root,
      graph_task,
      accumulate_grad,
      output_edges,
      &inputs,
      &sizes,
      &ivalue_args,
      &hooks,
      &packed_inputs,
      active_rstate);

  THPObjectPtr pyresult(check(PyObject_CallFunctionObjArgs(
      cache->runtime_wrapper.get(),
      cache->compiled_fn.get(),
      inputs.get(),
      sizes.get(),
      ivalue_args.get(),
      hooks.get(),
      packed_inputs.get(),
      NULL)));
  variable_list outputs = THPVariable_UnpackList(pyresult);
  TORCH_INTERNAL_ASSERT(outputs.size() == output_edges.size());
  return outputs;
}

static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args) {
  HANDLE_TH_ERRORS;
  PyObject* obj = nullptr;
  int b = 0;
  if (!PyArg_ParseTuple(args, "Op", &obj, &b)) {
    return nullptr;
  }

  TORCH_INTERNAL_ASSERT(b >= 0 && b < 2);
  PyObject* prior_compiler = the_autograd_compiler;
  PyObject* prior_dynamic = default_dyn_type_int == 0 ? Py_False : Py_True;
  default_dyn_type_int = b;
  if (obj == Py_None) { // disable
    the_autograd_compiler = nullptr; // decref not needed due to `prior`
    Engine::set_compiled_autograd(nullptr);
  } else { // enable
    Py_INCREF(obj);
    the_autograd_compiler = obj;
    Engine::set_compiled_autograd(&compiled_autograd);
  }

  if (prior_compiler == nullptr) {
    Py_INCREF(Py_None);
    prior_compiler = Py_None;
  }
  PyObject* prior = PyTuple_New(2);
  Py_INCREF(prior_dynamic);
  PyTuple_SET_ITEM(prior, 0, prior_compiler);
  PyTuple_SET_ITEM(prior, 1, prior_dynamic);
  return prior;
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
