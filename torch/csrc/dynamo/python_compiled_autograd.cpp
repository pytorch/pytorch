#include <torch/csrc/dynamo/python_compiled_autograd.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/dynamo/compiled_autograd.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <iostream>
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

namespace torch {
namespace dynamo {
using namespace torch::autograd;
using c10::SymInt;

static PyObject* wrap_int_list(const std::vector<int64_t>& inputs) {
  PyObject* pyinput = PyTuple_New(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, PyLong_FromSsize_t(inputs[i]));
  }
  return pyinput;
}

static PyObject* convert_hook_list(std::vector<c10::SafePyObject>& inputs) {
  // inplace, consumes the input hooks
  PyObject* pyinput = PyTuple_New(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    PyTuple_SET_ITEM(pyinput, i, inputs[i].release());
  }
  return pyinput;
}

static PyObject* check(PyObject* pyresult) {
  if (C10_UNLIKELY(pyresult == nullptr)) {
    // see https://github.com/pytorch/pytorch/pull/34845
    python_error err;
    err.persist();
    throw err;
  }
  return pyresult;
}

static void check(bool result) {
  if (C10_UNLIKELY(!result))
    check(nullptr);
}

struct CacheNode {
  // A node in the shadow graph, we follow next edges until we reach the end of
  // the graph
  static CacheNode* root() {
    static CacheNode _root;
    return &_root;
  }

  CacheNode* lookup(const CacheKey& key) {
    auto it = next.find(key);
    if (it == next.end()) {
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
    compiled_fn = nullptr;
  }

  bool is_empty() const {
    return next.size() == 0 && !compiled_fn;
  }

  CacheNode() : compiled_fn(nullptr) {}
  ~CacheNode() {
    if (!Py_IsInitialized()) {
      compiled_fn.release(); // leak on shutdown
    }
  }
  CacheNode(CacheNode&&) = delete;
  CacheNode(const CacheNode&) = delete;
  CacheNode& operator=(const CacheNode&) = delete;
  CacheNode& operator=(CacheNode&&) = delete;

  bool check_dynamic_sizes(AutogradCompilerCall& call) {
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
    for (const auto i : c10::irange(len)) {
      auto& expected = expected_sizes[i];
      if (expected.dyn_type == SizeInput::DYNAMIC ||
          expected.value != data[i].value) {
        cache_hit = cache_hit && expected.dyn_type == SizeInput::DYNAMIC;
        if (expected.value != data[i].value) {
          expected = SizeInput(SizeInput::DYNAMIC, data[i].value);
        }
        if (call.dyn_size_inputs.empty()) {
          call.dyn_size_inputs.reserve(len);
        }
        call.dyn_size_inputs.emplace_back(data[i].value);
      }
    }

    if (!cache_hit) {
      // we missed cache because static size inputs didn't match; force
      // recompilation with the varying size input as dynamic
      compiled_fn = nullptr;
    }
    return cache_hit;
  }

  PyObject* wrap_dynamic_inputs() {
    size_t dynamic_count = 0;
    size_t idx = 0;
    for (const auto& i : expected_sizes) {
      if (i.dyn_type == SizeInput::DYNAMIC) {
        ++dynamic_count;
      }
    }
    PyObject* pyinput = PyTuple_New(dynamic_count);
    for (const auto& i : expected_sizes) {
      if (i.dyn_type == SizeInput::DYNAMIC) {
        PyTuple_SET_ITEM(pyinput, idx++, PyLong_FromSsize_t(i.value));
      }
    }
    TORCH_INTERNAL_ASSERT(idx == dynamic_count);
    return pyinput;
  }

  std::vector<c10::optional<SymInt>> unwrap_dynamic_inputs(PyObject* pyresult) {
    TORCH_INTERNAL_ASSERT(PyList_CheckExact(pyresult));
    size_t idx = 0;
    size_t result_len = PyList_GET_SIZE(pyresult);
    std::vector<c10::optional<SymInt>> result;
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

  // TODO(jansel): benchmark map vs unordered_map
  std::unordered_map<CacheKey, std::unique_ptr<CacheNode>> next;
  std::vector<CacheKeyBuffer> key_storage;
  std::vector<SizeInput> expected_sizes;
  THPObjectPtr compiled_fn;
};

struct InputBuffers : public std::unordered_map<Node*, InputBuffer> {
  InputBuffer& lookup(Node* function) {
    auto it = find(function);
    if (it == end()) {
      it = emplace(function, InputBuffer(function->num_inputs())).first;
    }
    return it->second;
  }
};

static PyObject* the_autograd_compiler = nullptr;
static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args);

static PyObject* clear_cache(PyObject* dummy, PyObject* args) {
  CacheNode::root()->clear();
  Py_RETURN_NONE;
}

static PyObject* is_cache_empty(PyObject* dummy, PyObject* args) {
  if (CacheNode::root()->is_empty()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyMethodDef _methods[] = {
    {"set_autograd_compiler", set_autograd_compiler, METH_VARARGS, NULL},
    {"clear_cache", clear_cache, METH_NOARGS, NULL},
    {"is_cache_empty", is_cache_empty, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.autograd_compiler",
    "Hooks for compiling autograd",
    -1,
    _methods};

static TraceState call_begin_capture(
    PyObject* self,
    CacheNode& cache,
    AutogradCompilerCall& compiler_call,
    size_t num_outputs) {
  static PyObject* method_name = PyUnicode_InternFromString("begin_capture");
  THPObjectPtr pyinput(THPVariable_WrapList(compiler_call.inputs));
  THPObjectPtr pysizeinput(cache.wrap_dynamic_inputs());
  THPObjectPtr pyresult(check(PyObject_CallMethodObjArgs(
      self, method_name, pyinput.get(), pysizeinput.get(), NULL)));

  PyObject *fake_inputs, *fake_sizes;
  check(PyArg_ParseTuple(pyresult.get(), "OO", &fake_inputs, &fake_sizes));
  return TraceState(
      THPVariable_UnpackList(fake_inputs),
      cache.unwrap_dynamic_inputs(fake_sizes),
      compiler_call.accumulate_grad,
      num_outputs);
}

static PyObject* call_end_capture(PyObject* self, const variable_list& inputs) {
  static PyObject* method_name = PyUnicode_InternFromString("end_capture");
  THPObjectPtr pyinput(THPVariable_WrapList(inputs));
  return check(PyObject_CallMethodOneArg(self, method_name, pyinput.get()));
}

struct ClosingTHPObjectPtr : public THPObjectPtr {
  ClosingTHPObjectPtr(PyObject* o) : THPObjectPtr(o) {}
  ~ClosingTHPObjectPtr() {
    static PyObject* method_name = PyUnicode_InternFromString("close");
    check(PyObject_CallMethodNoArgs(get(), method_name));
  }
};

variable_list compiled_autograd(
    const std::shared_ptr<Node>& graph_root,
    GraphTask& graph_task,
    bool accumulate_grad,
    const edge_list& output_edges) {
  TORCH_CHECK(
      output_edges.empty() || !accumulate_grad,
      "specifying inputs= with .backward() not yet implemented for compiled autograd")
  static std::mutex lock;
  std::lock_guard<std::mutex> lock_guard(lock);
  pybind11::gil_scoped_acquire gil;
  std::unordered_map<Node*, int>& dependencies = graph_task.dependencies_;
  std::vector<std::shared_ptr<Node>> worklist{graph_root};
  AutogradCompilerCall compiler_call(accumulate_grad);
  for (const auto i : c10::irange(output_edges.size())) {
    compiler_call.node_calls.lookup(output_edges[i].function)
        .mark_output(output_edges[i].input_nr, i);
  }
  const bool check_exec_info = !graph_task.exec_info_.empty();
  CacheNode* cache = CacheNode::root();
  std::vector<NodeCall*> calls;
  calls.reserve(
      check_exec_info ? graph_task.exec_info_.size() : dependencies.size() + 1);

  while (!worklist.empty()) {
    std::shared_ptr<Node> fn = std::move(worklist.back());
    worklist.pop_back();
    NodeCall& call = compiler_call.node_calls.lookup(fn);
    calls.emplace_back(&call);

    { // update cache and gather args into `compiler_call`
      CompiledNodeArgs node_args(compiler_call, call);
      node_args.collect(call);
      if (node_args.cond(call.needed)) {
        fn->compiled_args(node_args);
        node_args.collect(call.node->next_edges());
      }
      cache = cache->lookup(node_args.key());
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
  }

  // TODO(jansel): some dynamic sizes seem to be ints not symints
  if (!cache->check_dynamic_sizes(compiler_call)) {
    // cache miss, need to capture FX graph
    ClosingTHPObjectPtr py_compiler(
        check(PyObject_CallNoArgs((the_autograd_compiler))));
    TraceState state = call_begin_capture(
        py_compiler, *cache, compiler_call, output_edges.size());
    InputBuffers input_buffers;

    for (NodeCall* call_ptr : calls) {
      NodeCall& call = *call_ptr;
      // TODO(jansel): consider adding some of this stuff:
      // guard(local_graph_task); NodeGuard ndguard(task.fn_); const auto
      // opt_parent_stream = (*func).stream(c10::DeviceType::CUDA);
      // c10::OptionalStreamGuard parent_stream_guard{opt_parent_stream};
      // CheckpointValidGuard cpvguard(graph_task);
      // at::getStepCallbacksUnlessEmpty(at::RecordScope::BACKWARD_FUNCTION);
      // if (C10_UNLIKELY(step_callbacks.has_value())) { ... }

      variable_list inputs = input_buffers.lookup(call.node.get()).buffer;

      if (call.tensor_pre_hooks.size() > 0) {
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
      if (call.pre_hooks.size() > 0) {
        THPObjectPtr pyinputs(THPVariable_WrapList(inputs));
        for (const auto hook : call.pre_hooks) {
          pyinputs = check(PyObject_CallMethod(
              py_compiler.get(), "pre_hook", "Oi", pyinputs.get(), hook));
        }
        inputs = THPVariable_UnpackList(pyinputs);
      }

      SwapSavedVariables saved(compiler_call, state, call.node);
      variable_list outputs = call.node->apply_with_saved(inputs, saved);
      saved.before(call.node->next_edges());
      validate_outputs(
          call.node->next_edges(), outputs, [&](const std::string& msg) {
            std::ostringstream ss;
            ss << "[Compiled Autograd Tracing: " << call.node->name() << "] "
               << msg;
            return ss.str();
          });
      saved.after(call.node->next_edges());

      if (call.post_hooks.size() > 0) {
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
                  next.input_nr, std::move(output), c10::nullopt, c10::nullopt);
        }
      }
    }

    cache->compiled_fn = check(call_end_capture(py_compiler, state.outputs));
    state.debug_asserts();
  }

  // TODO(jansel): we should release all the variables and then use a
  //               boxed calling convention so activation memory can be freed
  // TODO(jansel): clear grads we will overwrite below
  for (auto& call : calls) {
    call->node->release_variables();
  }

  THPObjectPtr inputs(THPVariable_WrapList(compiler_call.inputs));
  THPObjectPtr sizes(wrap_int_list(compiler_call.dyn_size_inputs));
  THPObjectPtr hooks(convert_hook_list(compiler_call.hooks));
  THPObjectPtr pyresult(check(PyObject_CallFunctionObjArgs(
      cache->compiled_fn.get(), inputs.get(), sizes.get(), hooks.get(), NULL)));
  variable_list outputs = THPVariable_UnpackList(pyresult);
  if (accumulate_grad) {
    TORCH_INTERNAL_ASSERT(
        outputs.size() == compiler_call.set_grad_targets.size());
    for (const auto i : c10::irange(outputs.size())) {
      // TODO(jansel): does this one need to be an inplace copy?  if so it
      // should go in the graph
      at::Tensor& grad = compiler_call.set_grad_targets[i].mutable_grad();
      grad = outputs[i];
    }
    return variable_list();
  } else {
    TORCH_INTERNAL_ASSERT(outputs.size() == output_edges.size());
    return outputs;
  }
}

static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args) {
  PyObject* obj;
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
}

PyObject* torch_c_dynamo_compiled_autograd_init() {
  return PyModule_Create(&_module);
}

} // namespace dynamo
} // namespace torch
