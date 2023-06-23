#include <torch/csrc/dynamo/compiled_autograd_py.h>

#include <torch/csrc/autograd/compiled_autograd.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <iostream>
#include <vector>

namespace torch {
namespace dynamo {
using namespace torch::autograd;
using c10::SymInt;
enum IsDynamic : uint8_t { STATIC = 0, DYNAMIC = 1 };

static PyObject* wrap_variable_list(const variable_list& inputs) {
  PyObject* pyinput = PyList_New(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    PyList_SET_ITEM(pyinput, i, THPVariable_Wrap(inputs[i]));
  }
  return pyinput;
}

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

static variable_list unwrap_variable_list(PyObject* pyresult) {
  TORCH_CHECK(PyList_CheckExact(pyresult));
  auto result_len = PyList_GET_SIZE(pyresult);
  variable_list result;
  result.reserve(result_len);
  for (const auto i : c10::irange(result_len)) {
    result.emplace_back(THPVariable_Unpack(PyList_GET_ITEM(pyresult, i)));
  }
  return result;
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
  static CacheNode* root() {
    static CacheNode _root;
    return &_root;
  }

  CacheNode* lookup(const CacheKey& key) {
    auto it = next.find(key);
    if (it == next.end()) {
      // caller's key is in temporary memory, must copy it
      CacheKeyBuffer buffer(key.key, key.key_size);
      CacheKey key_with_storage(key.node_type, buffer.data, key.key_size);
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
    const int64_t* data = call.all_size_inputs.data();
    if (expected_sizes.empty()) {
      expected_sizes.reserve(len);
      for (const auto i : c10::irange(len)) {
        // static shape until we see it change
        expected_sizes.emplace_back(std::make_pair(STATIC, data[i]));
      }
    } else {
      TORCH_CHECK(expected_sizes.size() == call.all_size_inputs.size());
      for (const auto i : c10::irange(len)) {
        auto& expected = expected_sizes[i];
        if (expected.first == DYNAMIC || expected.second != data[i]) {
          cache_hit = cache_hit && expected.first == DYNAMIC;
          expected = std::make_pair(DYNAMIC, data[i]);
          if (call.dyn_size_inputs.empty())
            call.dyn_size_inputs.reserve(len);
          call.dyn_size_inputs.emplace_back(data[i]);
        }
      }
    }
    if (!cache_hit) {
      compiled_fn = nullptr;
    }
    return cache_hit;
  }

  PyObject* wrap_dynamic_inputs() {
    size_t dynamic_count = 0;
    size_t idx = 0;
    for (const auto& i : expected_sizes) {
      if (i.first == DYNAMIC) {
        ++dynamic_count;
      }
    }
    PyObject* pyinput = PyTuple_New(dynamic_count);
    for (const auto& i : expected_sizes) {
      if (i.first == DYNAMIC) {
        PyTuple_SET_ITEM(pyinput, idx++, PyLong_FromSsize_t(i.second));
      }
    }
    TORCH_CHECK(idx == dynamic_count);
    return pyinput;
  }

  std::vector<c10::optional<SymInt>> unwrap_dynamic_inputs(PyObject* pyresult) {
    TORCH_CHECK(PyList_CheckExact(pyresult));
    size_t idx = 0;
    size_t result_len = PyList_GET_SIZE(pyresult);
    std::vector<c10::optional<SymInt>> result;
    result.reserve(expected_sizes.size());
    for (const auto& i : expected_sizes) {
      if (i.first == DYNAMIC) {
        TORCH_CHECK(idx < result_len);
        result.emplace_back(
            py::cast<c10::SymInt>(PyList_GET_ITEM(pyresult, idx++)));
      } else {
        result.emplace_back();
      }
    }
    TORCH_CHECK(idx == result_len && result.size() == expected_sizes.size());
    return result;
  }

  // TODO(jansel): benchmark map vs unordered_map
  std::unordered_map<CacheKey, std::unique_ptr<CacheNode>> next;
  std::vector<CacheKeyBuffer> key_storage;
  std::vector<std::pair<IsDynamic, int64_t>> expected_sizes;
  THPObjectPtr compiled_fn;
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
  THPObjectPtr pyinput(wrap_variable_list(compiler_call.inputs));
  THPObjectPtr pysizeinput(cache.wrap_dynamic_inputs());
  THPObjectPtr pyresult(check(PyObject_CallMethodObjArgs(
      self, method_name, pyinput.get(), pysizeinput.get(), NULL)));

  PyObject *fake_inputs, *fake_sizes;
  check(PyArg_ParseTuple(pyresult.get(), "OO", &fake_inputs, &fake_sizes));
  return TraceState(
      unwrap_variable_list(fake_inputs),
      cache.unwrap_dynamic_inputs(fake_sizes),
      compiler_call.accumulate_grad,
      num_outputs);
}

static PyObject* call_end_capture(PyObject* self, const variable_list& inputs) {
  static PyObject* method_name = PyUnicode_InternFromString("end_capture");
  THPObjectPtr pyinput(wrap_variable_list(inputs));
  return check(PyObject_CallMethodOneArg(self, method_name, pyinput.get()));
}

variable_list compiled_autograd(
    const std::shared_ptr<Node>& graph_root,
    GraphTask& graph_task,
    bool accumulate_grad,
    const edge_list& output_edges) {
  TORCH_CHECK(
      output_edges.empty() || !accumulate_grad,
      "outputs+accumulate_grad not yet implemented")
  pybind11::gil_scoped_acquire gil;
  NoGradGuard no_grad; // TODO(jansel): double backward

  std::unordered_map<Node*, int> dependencies =
      std::move(graph_task.dependencies_);
  std::vector<std::shared_ptr<Node>> worklist{graph_root};
  worklist.reserve(dependencies.size());
  std::unordered_map<Node*, NodeCall> node_inputs;
  node_inputs.emplace(graph_root.get(), NodeCall(graph_root));
  std::vector<NodeCall> calls;
  calls.reserve(dependencies.size() + 8);
  AutogradCompilerCall compiler_call(accumulate_grad);
  CacheNode* cache = CacheNode::root();

  for (const auto i : c10::irange(output_edges.size())) {
    auto inp = node_inputs.find(output_edges[i].function.get());
    if (inp == node_inputs.end()) {
      inp = node_inputs
                .emplace(
                    output_edges[i].function.get(),
                    NodeCall(output_edges[i].function))
                .first;
    }
    inp->second.mark_output(output_edges[i].input_nr, i);
  }

  while (!worklist.empty()) {
    std::shared_ptr<Node> fn = std::move(worklist.back());
    worklist.pop_back();

    auto node_input_iter = node_inputs.find(fn.get());
    TORCH_CHECK(node_input_iter != node_inputs.end());
    NodeCall& node_call = node_input_iter->second;

    { // update cache and gather args into `compiler_call`
      CompiledNodeArgs node_args(compiler_call, node_call);
      node_args.collect(node_call);
      fn->compiled_args(node_args);
      node_args.collect_hooks_from(fn.get());
      cache = cache->lookup(node_args.key());
    }

    // finalize node construction
    calls.emplace_back(std::move(node_call));
    node_inputs.erase(node_input_iter);
    size_t node_id = calls.size();

    const auto& edges = fn->next_edges();
    for (int output_id = edges.size() - 1; output_id >= 0; --output_id) {
      if (!edges[output_id].is_valid()) {
        continue;
      }
      const std::shared_ptr<Node>& edge_node = edges[output_id].function;
      uint32_t input_nr = edges[output_id].input_nr;

      auto inp = node_inputs.find(edge_node.get());
      if (inp == node_inputs.end()) {
        inp = node_inputs.emplace(edge_node.get(), NodeCall(edge_node)).first;
      }
      NodeCall& input_buffer = inp->second;
      if (!input_buffer[input_nr].is_set()) {
        // normal case
        input_buffer[input_nr] = OutputRef(node_id, output_id);
      } else {
        // create a fake node to add the existing gradient to the new one
        NodeCall implicit_add(std::make_shared<ImplicitAdd>(), 2);
        implicit_add[0] = input_buffer[input_nr];
        implicit_add[1] = OutputRef(node_id, output_id);

        { // update cache
          CompiledNodeArgs node_args(compiler_call, implicit_add);
          node_args.collect(implicit_add.input_refs);
          cache = cache->lookup(node_args.key());
        }

        calls.emplace_back(std::move(implicit_add));
        input_buffer[input_nr] = OutputRef(calls.size(), 0);
      }

      auto it = dependencies.find(edge_node.get());
      TORCH_CHECK(it != dependencies.end());
      if (--it->second == 0) {
        dependencies.erase(it);
        worklist.emplace_back(edge_node);
      }
    }
  }

  // TODO(jansel): many of the dynamic sizes seem to be stored as ints rather
  // than symints
  if (!cache->check_dynamic_sizes(compiler_call)) {
    // cache miss, need to capture FX graph
    THPObjectPtr py_compiler(
        check(PyObject_CallNoArgs((the_autograd_compiler))));

    TraceState state = call_begin_capture(
        py_compiler, *cache, compiler_call, output_edges.size());

    std::vector<variable_list> node_outputs;
    node_outputs.reserve(calls.size() + 1);
    node_outputs.emplace_back(state.proxy_inputs);
    for (auto& call : calls) {
      // TODO(jansel): consider adding some of this stuff:
      // at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);
      // c10::WarningUtils::WarningHandlerGuard
      // warnings_guard(&local_graph_task->warning_handler_); GraphTaskGuard
      // guard(local_graph_task); NodeGuard ndguard(task.fn_); const auto
      // opt_parent_stream = (*func).stream(c10::DeviceType::CUDA);
      // c10::OptionalStreamGuard parent_stream_guard{opt_parent_stream};
      // CheckpointValidGuard cpvguard(graph_task);
      // at::NoNamesGuard no_names_guard;
      // auto step_callbacks =
      //    at::getStepCallbacksUnlessEmpty(at::RecordScope::BACKWARD_FUNCTION);
      // if (C10_UNLIKELY(step_callbacks.has_value())) { ... }

      variable_list inputs;
      inputs.reserve(call.input_refs.size());
      for (auto ref : call.input_refs) {
        if (ref.is_set()) {
          inputs.emplace_back(node_outputs[ref.node_id][ref.index]);
        } else {
          // TODO(jansel): do we need to construct zeros here?
          inputs.emplace_back();
        }
      }

      for (const auto& graph_output : call.graph_output) {
        int input_nr = graph_output.first;
        int output_index = graph_output.second;
        TORCH_CHECK(output_index < static_cast<int>(state.outputs.size()));
        TORCH_CHECK(!state.outputs[output_index].defined());
        state.outputs[output_index] = inputs[input_nr];
      }

      if (call.tensor_pre_hooks.size() + call.pre_hooks.size() > 0) {
        // TODO(jansel): we should lift hooks to be inputs to the graph since we
        // are not specializing on them
        THPObjectPtr pyinputs(wrap_variable_list(inputs));
        for (const auto& hook : call.tensor_pre_hooks) {
          pyinputs = check(PyObject_CallMethod(
              py_compiler,
              "tensor_pre_hook",
              "Oii",
              pyinputs.get(),
              hook.first,
              hook.second));
        }
        for (const auto hook : call.pre_hooks) {
          pyinputs = check(PyObject_CallMethod(
              py_compiler.get(), "pre_hook", "Oi", pyinputs.get(), hook));
        }
        inputs = unwrap_variable_list(pyinputs);
      }

      SwapSavedVariables saved(state, call.node);
      variable_list outputs = call.node->apply_with_saved(inputs, saved);

      if (call.post_hooks.size() > 0) {
        THPObjectPtr pyinputs(wrap_variable_list(inputs));
        THPObjectPtr pyoutputs(wrap_variable_list(outputs));
        for (const auto hook : call.post_hooks) {
          pyoutputs = check(PyObject_CallMethod(
              py_compiler.get(),
              "post_hook",
              "OOi",
              pyoutputs.get(),
              pyinputs.get(),
              hook));
        }
        outputs = unwrap_variable_list(pyoutputs);
      }

      node_outputs.emplace_back(std::move(outputs));
    }

    cache->compiled_fn = check(call_end_capture(py_compiler, state.outputs));
  }

  // TODO(jansel): we should release all the variables and then use a
  //               boxed calling convention so activation memory can be freed
  for (auto& call : calls) {
    call.node->release_variables();
  }

  {
    THPObjectPtr inputs(wrap_variable_list(compiler_call.inputs));
    THPObjectPtr sizes(wrap_int_list(compiler_call.dyn_size_inputs));
    THPObjectPtr hooks(convert_hook_list(compiler_call.hooks));
    THPObjectPtr pyresult(check(PyObject_CallFunctionObjArgs(
        cache->compiled_fn.get(),
        inputs.get(),
        sizes.get(),
        hooks.get(),
        NULL)));
    variable_list outputs = unwrap_variable_list(pyresult);
    if (accumulate_grad) {
      TORCH_CHECK(outputs.size() == compiler_call.set_grad_targets.size());
      for (const auto i : c10::irange(outputs.size())) {
        // TODO(jansel): does this one need to be an inplace copy?  if so it
        // should go in the graph
        at::Tensor& grad = compiler_call.set_grad_targets[i].mutable_grad();
        grad = outputs[i];
      }
      return variable_list();
    } else {
      TORCH_CHECK(outputs.size() == output_edges.size());
      return outputs;
    }
  }
}

static PyObject* set_autograd_compiler(PyObject* dummy, PyObject* args) {
  PyObject* obj;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }

  PyObject* prior = the_autograd_compiler;
  if (obj == Py_None) { // disable
    the_autograd_compiler = nullptr;
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
