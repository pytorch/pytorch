#include <torch/csrc/autograd/python_engine.h>

#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/LegacyVmapMode.h>
#include <c10/util/irange.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_saved_variable_hooks.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>

#ifndef _WIN32
#include <pthread.h>
#endif

#include <memory> // for unique_ptr
#include <utility>

using namespace torch::autograd;

struct THPEngine {
  PyObject_HEAD
};

static bool _reinitialize_engine = false;

namespace torch::autograd::python {

PythonEngine::PythonEngine() = default;

Engine& PythonEngine::get_python_engine() {
  static PythonEngine engine;
  // This is "probably" thread-safe because the flag is set in a fork handler
  // before any threads are created, and this function is only called with the
  // GIL held. However, using fork + threads is playing with fire so this is
  // more of a "best effort" thing. For example, if the fork occurs while the
  // backwards threads hold a lock, we'll probably deadlock in the engine
  // destructor.
  if (_reinitialize_engine) {
    engine.release_workers();
    engine.~PythonEngine();
    new (&engine) torch::autograd::python::PythonEngine();
    _reinitialize_engine = false;
  }
  return engine;
}

PythonEngine::~PythonEngine() {
  Engine::stop();
}

#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 9
#define IS_PYTHON_3_9_PLUS
#endif

void PythonEngine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {
  // Increment thread usage count before acquiring the GIL
  if (should_increment) {
    increment_non_reentrant_thread_count();
  }
  // Create a PyThreadState, but release the GIL. This lets
  // pybind11::gil_scoped_acquire calls inside thread_main acquire the GIL
  // without having to create a new PyThreadState each time.
#if defined(IS_PYTHON_3_9_PLUS)
  auto gil = std::make_unique<pybind11::gil_scoped_acquire>();
#else
  pybind11::gil_scoped_acquire gil;
#endif
  pybind11::gil_scoped_release no_gil;
  Engine::thread_init(device, ready_queue, false);

  if (should_increment) {
    // Decrement the count during shutdown if we incremented earlier.
    decrement_non_reentrant_thread_count();
  }

#if defined(IS_PYTHON_3_9_PLUS)
  // Do not call PyEval_RestoreThread, PyThreadState_[Clear|DeleteCurrent] if
  // runtime is finalizing
  if (!Py_IsInitialized()) {
    no_gil.disarm();
    // TODO: call disarm once PyThreadState_Clear can safely be called from
    // finalize NOTE: deploy.cpp calls `PyInterpreterState_Delete` to destruct
    // PyThreadState, so avoid use-after-free here.
    auto ptr = gil.release();
    operator delete(ptr);
  }
#endif
}

void PythonEngine::thread_on_exception(
    const std::shared_ptr<GraphTask>& graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {
  // See Note [ Persisting PyErr state across autograd engine threads ]
  auto python_err = dynamic_cast<python_error*>(&e);
  if (python_err) {
    python_err->persist();
  }
  Engine::thread_on_exception(graph_task, fn, e);
}

std::unique_ptr<AnomalyMetadata> PythonEngine::make_anomaly_metadata() {
  return std::make_unique<PyAnomalyMetadata>();
}

std::unique_ptr<SavedVariableHooks> PythonEngine::
    get_default_saved_variable_hooks() {
  return PyDefaultSavedVariableHooks::get_hooks();
}

variable_list PythonEngine::execute(
    const edge_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) {
  TORCH_CHECK(
      !PyGILState_Check(),
      "The autograd engine was called while holding the GIL. If you are using the C++ "
      "API, the autograd engine is an expensive operation that does not require the "
      "GIL to be held so you should release it with 'pybind11::gil_scoped_release no_gil;'"
      ". If you are not using the C++ API, please report a bug to the pytorch team.")
  try {
    return Engine::execute(
        roots, inputs, keep_graph, create_graph, accumulate_grad, outputs);
  } catch (python_error& e) {
    e.restore();
    throw;
  }
}

c10::intrusive_ptr<at::ivalue::Future> PythonEngine::execute_with_graph_task(
    const std::shared_ptr<GraphTask>& graph_task,
    std::shared_ptr<Node> graph_root,
    InputBuffer&& input_buffer) {
  try {
    return Engine::execute_with_graph_task(
        graph_task, std::move(graph_root), std::move(input_buffer));
  } catch (python_error& e) {
    pybind11::gil_scoped_acquire gil;
    if (!PyErr_Occurred()) {
      // Set the error indicator only if it is not set already.
      e.restore();
    }
    throw;
  }
}
} // namespace torch::autograd::python

static Edge parseGradientEdge(PyObject* obj, int64_t index) {
  PyObject* grad_fn = PyTuple_GetItem(obj, 0);
  auto output_nr = THPUtils_unpackLong(PyTuple_GetItem(obj, 1));
  std::shared_ptr<torch::autograd::Node> grad_fn_sp;
  if (THPFunction_Check(grad_fn)) {
    grad_fn_sp = ((THPFunction*)grad_fn)->cdata.lock();
  } else if (THPCppFunction_Check(grad_fn)) {
    grad_fn_sp = ((THPCppFunction*)grad_fn)->cdata;
  } else {
    TORCH_CHECK(
        false,
        "GradientEdge's first object must be an autograd.graph.Node "
        "but got ",
        THPUtils_typename(grad_fn));
  }
  return Edge(grad_fn_sp, output_nr);
}

// Implementation of torch._C._EngineBase.run_backward
static PyObject* THPEngine_run_backward(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  PyObject* tensors = nullptr;
  PyObject* grad_tensors = nullptr;
  unsigned char keep_graph = 0;
  unsigned char create_graph = 0;
  PyObject* inputs = nullptr;
  unsigned char allow_unreachable = 0;
  unsigned char accumulate_grad =
      0; // Indicate whether to accumulate grad into leaf Tensors or capture
  constexpr const char* accepted_kwargs[] = {
      "tensors",
      "grad_tensors",
      "keep_graph",
      "create_graph",
      "inputs",
      "allow_unreachable",
      "accumulate_grad",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OObb|Obb",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast,-warnings-as-errors)
          const_cast<char**>(accepted_kwargs),
          &tensors,
          &grad_tensors,
          &keep_graph,
          &create_graph,
          &inputs,
          &allow_unreachable,
          &accumulate_grad))
    return nullptr;
  TORCH_CHECK(
      PyTuple_Check(tensors),
      "tensors argument is expected to "
      "be a tuple, but got ",
      THPUtils_typename(tensors));
  TORCH_CHECK(
      PyTuple_Check(grad_tensors),
      "grad_tensors argument is "
      "expected to be a tuple, but got ",
      THPUtils_typename(grad_tensors));

  Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_tensors);
  TORCH_CHECK(
      num_tensors == num_gradients,
      "got ",
      num_tensors,
      " tensors and ",
      num_gradients,
      " gradients");

  // The user either called autograd.backward(...) or autograd.grad(...) to get
  // here
  bool backward_api_called = accumulate_grad;
  TORCH_CHECK(
      !backward_api_called || at::impl::VmapMode::current_vmap_level() == 0,
      "backward() called inside torch.vmap. This is not supported, "
      "please call backward() outside torch.vmap or instead use "
      "torch.autograd.grad inside torch.vmap");

  edge_list roots;
  roots.reserve(num_tensors);
  variable_list grads;
  grads.reserve(num_tensors);
  for (const auto i : c10::irange(num_tensors)) {
    PyObject* _tensor = PyTuple_GET_ITEM(tensors, i);
    Edge gradient_edge; // Temporary variable to hold the gradient edge
    std::optional<at::Tensor> mb_output;
    if (THPVariable_Check(_tensor)) {
      mb_output = THPVariable_Unpack(_tensor);
      TORCH_CHECK(
          !isBatchedTensor(mb_output.value()),
          "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
          "torch.vmap. We do not support the case where any outputs are ",
          "vmapped tensors (output ",
          i,
          " is being vmapped over). Please "
          "call autograd.grad() outside torch.vmap or file a bug report "
          "with your use case.");
      gradient_edge = torch::autograd::impl::gradient_edge(mb_output.value());
    } else if (PyObject_IsInstance(_tensor, THPGradientEdgeClass)) {
      gradient_edge = parseGradientEdge(_tensor, i);
    } else {
      TORCH_CHECK(
          false,
          "element ",
          i,
          " of tensors tuple is neither a Tensor nor a GradientEdge");
    }
    TORCH_CHECK(
        gradient_edge.function,
        "element ",
        i,
        " of tensors does not require grad and does not have a grad_fn");
    roots.push_back(std::move(gradient_edge));

    PyObject* grad = PyTuple_GET_ITEM(grad_tensors, i);
    if (THPVariable_Check(grad)) {
      const Variable& grad_var = THPVariable_Unpack(grad);
      if (grad_var.has_names()) {
        TORCH_WARN(
            "Autograd was passed a named grad tensor with dims ",
            grad_var.names(),
            ". Autograd does not yet support named tensor semantics, so all names ",
            "will be ignored. In practice all computed gradients will still be correct "
            "according to regular tensor semantics.");
      }
      grads.push_back(grad_var);
    } else {
      TORCH_CHECK(
          grad == Py_None,
          "element ",
          i,
          " of gradients tuple is not a Tensor or None");
      TORCH_CHECK(
          mb_output.has_value(),
          "element ",
          i,
          " of gradients tuple is None, but the corresponding output is a GradientEdge."
          "This is not supported.");
      TORCH_CHECK(
          !mb_output.value().requires_grad(),
          "element ",
          i,
          " of gradients tuple is None, but the corresponding Tensor requires grad");
    }
  }

  std::vector<Edge> output_edges;
  if (inputs != nullptr) {
    TORCH_CHECK(
        PyTuple_CheckExact(inputs), "inputs to run_backward must be a tuple");
    int num_inputs = PyTuple_GET_SIZE(inputs);
    output_edges.reserve(num_inputs);
    for (const auto i : c10::irange(num_inputs)) {
      PyObject* input = PyTuple_GET_ITEM(inputs, i);
      if (THPVariable_Check(input)) {
        const auto& tensor = THPVariable_Unpack(input);
        TORCH_CHECK(
            !isBatchedTensor(tensor),
            "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
            "torch.vmap. We do not support the case where any inputs are ",
            "vmapped tensors (input ",
            i,
            " is being vmapped over). Please "
            "call autograd.grad() outside torch.vmap or file a bug report "
            "with your use case.")
        const auto output_nr = tensor.output_nr();
        auto grad_fn = tensor.grad_fn();
        if (!grad_fn) {
          grad_fn = torch::autograd::impl::try_get_grad_accumulator(tensor);
        }
        if (accumulate_grad) {
          tensor.retain_grad();
        }
        TORCH_CHECK(
            tensor.requires_grad(),
            "One of the differentiated Tensors does not require grad");
        if (!grad_fn) {
          // NOTE [ Autograd Unreachable Input ]
          // Since input has no grad_accumulator, its guaranteed to be
          // unreachable. We initialize an edge pointing to a non-nullptr Node
          // so nodes in the graph (e.g., mul when an operand is scalar) that
          // have edges pointing to nullptr don't get erroneously assigned
          // `needed = True` in exec_info.
          output_edges.emplace_back(std::make_shared<Identity>(), 0);
        } else {
          output_edges.emplace_back(grad_fn, output_nr);
        }
      } else if (PyObject_IsInstance(input, THPGradientEdgeClass)) {
        output_edges.emplace_back(parseGradientEdge(input, i));
      } else {
        TORCH_CHECK(
            false,
            "all inputs have to be Tensors or GradientEdges, but got ",
            THPUtils_typename(input));
      }
    }
  }

  variable_list outputs;
  {
    pybind11::gil_scoped_release no_gil;
    auto& engine = python::PythonEngine::get_python_engine();
    outputs = engine.execute(
        roots, grads, keep_graph, create_graph, accumulate_grad, output_edges);
  }

  if (!backward_api_called && inputs != nullptr) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    THPObjectPtr py_outputs{PyTuple_New(num_inputs)};
    if (!py_outputs)
      return nullptr;
    for (const auto i : c10::irange(num_inputs)) {
      TORCH_CHECK(
          allow_unreachable || outputs[i].defined(),
          "One of the "
          "differentiated Tensors appears to not have been used "
          "in the graph. Set allow_unused=True if this is the "
          "desired behavior.");
      PyTuple_SET_ITEM(py_outputs.get(), i, THPVariable_Wrap(outputs[i]));
    }
    return py_outputs.release();
  } else {
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEngine_queue_callback(PyObject* self, PyObject* _callback) {
  HANDLE_TH_ERRORS
  auto& engine = python::PythonEngine::get_python_engine();
  std::shared_ptr<PyObject> callback(_callback, [](PyObject* obj) {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(obj);
  });
  Py_INCREF(_callback);
  engine.queue_callback([callback]() {
    pybind11::gil_scoped_acquire gil;
    THPObjectPtr result{PyObject_CallFunctionObjArgs(callback.get(), nullptr)};
    if (!result) {
      // Note [ Persisting PyErr state across autograd engine threads ]
      //
      // Since the autograd engine is multi-threaded, and Python error state is
      // local to each thread, it must preserve the python error from the worker
      // thread and rethrow it as-is in the calling thread. This is done via
      // persisting the error in the two places that can encounter Python
      // errors: (1) evaluate function and (2) queued callbacks.
      //
      // TODO: the engine is not actually responsible for persisting the error
      // in the custom autograd Function case today! See the note above
      // `raise_python_error()` function in python_function.cpp and
      // python_hooks.cpp for more details. Persisting an extra time in the
      // engine is fine because doing so is a no-op when the python_error has
      // already been persisted.
      python_error err;
      err.persist();
      throw std::move(err);
    }
  });
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEngine_is_checkpoint_valid(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto& engine = python::PythonEngine::get_python_engine();
  if (engine.is_checkpoint_valid()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPEngine_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  return type->tp_alloc(type, 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMethodDef THPEngine_methods[] = {
    {"run_backward",
     castPyCFunctionWithKeywords(THPEngine_run_backward),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"queue_callback", THPEngine_queue_callback, METH_O, nullptr},
    {"is_checkpoint_valid",
     THPEngine_is_checkpoint_valid,
     METH_NOARGS,
     nullptr},
    {nullptr}};

static PyTypeObject THPEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._EngineBase", /* tp_name */
    sizeof(THPEngine), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPEngine_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPEngine_new /* tp_new */
};

static void child_atfork() {
  _reinitialize_engine = true;
}

bool THPEngine_initModule(PyObject* module) {
#ifndef _WIN32
  if (pthread_atfork(nullptr, nullptr, child_atfork) != 0) {
    throw std::runtime_error("unable to set pthread_atfork handler");
  }
#endif
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject*)&THPEngineType);
  set_default_engine_stub(python::PythonEngine::get_python_engine);
  return true;
}
