#include "torch/csrc/autograd/python_engine.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/python_function.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/PtrWrapper.h"
#include "torch/csrc/utils/auto_gil.h"

#include <unordered_set>

using namespace torch::autograd;

struct THPEngine {
    PyObject_HEAD
};

static torch::autograd::python::PythonEngine engine;

namespace torch { namespace autograd { namespace python {

void PythonEngine::thread_init(int device) {
  // Create a PyThreadState, but release the GIL. This lets AutoGIL calls
  // inside thread_main acquire the GIL without having to create a new
  // PyThreadState each time.
  AutoGIL gil;
  AutoNoGIL no_gil;
  Engine::thread_init(device);
}

void PythonEngine::thread_on_exception(FunctionTask& task, std::exception& e) {
  auto python_err = dynamic_cast<python_error*>(&e);
  if (python_err) {
    python_err->persist();
  }
  Engine::thread_on_exception(task, e);
}

void PythonEngine::execute(
    const function_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    const pre_callback_map& pre_callbacks,
    const post_callback_map& post_callbacks) {
  try {
    Engine::execute(roots, inputs, keep_graph, pre_callbacks, post_callbacks);
  } catch (python_error& e) {
    e.restore();
    throw;
  }
}

PythonEngine& PythonEngine::getDefaultEngine() {
  return engine;
}

}}} // namespace torch::autograd::python

PyObject *THPEngineClass = NULL;

struct CallbackContext {
  std::string error;
  THPObjectPtr outputs;
  // Used to determine which callback arguments should be used to
  // fill outputs.
  // Function -> ([grad_nr, outputs_idx], is_leaf)
  std::unordered_map<
    std::shared_ptr<Function>,
    std::pair<std::vector<std::pair<int, int>>, bool>> output_map;
};

void compute_partial_exec_callbacks(const function_list& roots,
                                    const CallbackContext& ctx,
                                    Engine::pre_callback_map& map,
                                    bool allow_unreachable) {
  // This callback is used to suppress the computation of a node
  // if it is not necessary.
  static Engine::pre_callback_type abort_callback(
      [](Function* fn, variable_list &vars) { return false; });

  std::vector<Function*> queue;
  std::unordered_set<Function*> seen;    // for the initial DFS
  std::unordered_set<Function*> needed;  // functions to compute
  std::unordered_map<Function*, std::vector<Function*>> rev_graph;

  // Reverse the next_fn edges
  queue.reserve(roots.size());
  for (auto& root : roots) {
    auto ptr = root.first.get();
    bool unseen;
    std::tie(std::ignore, unseen) = seen.insert(ptr);
    if (unseen) queue.emplace_back(ptr);
  }
  while (!queue.empty()) {
    auto fn = queue.back(); queue.pop_back();
    for (auto& next_fn_pair : fn->next_functions) {
      auto next_fn = next_fn_pair.first.get();
      if (!next_fn) continue;
      rev_graph[next_fn].push_back(fn);
      if (seen.insert(next_fn).second) {
        queue.push_back(next_fn);
      }
    }
  }
  auto all_functions = std::move(seen); // this is cheap and improves readability

  // Find all functions we need to compute
  queue.clear();
  for (auto input_info: ctx.output_map) {
    auto input = input_info.first.get();
    auto rev_edges_it = rev_graph.find(input);
    if (!allow_unreachable && rev_edges_it == rev_graph.end())
      throw std::runtime_error("differentiated input is unreachable");
    queue.emplace_back(input);
    needed.insert(input);
  }
  while (!queue.empty()) {
    auto fn = queue.back(); queue.pop_back();
    for (auto rev_next_fn : rev_graph[fn]) {
      if (needed.insert(rev_next_fn).second) {
        queue.push_back(rev_next_fn);
      }
    }
  }

  // Prevent expansion for functions in {all_vertices} \ {needed}
  for (auto fn : all_functions) {
    if (needed.count(fn) > 0) continue;
    map.emplace(fn, abort_callback);
  }
}

// Implementation of torch._C._EngineBase.run_backward
PyObject *THPEngine_run_backward(THPEngine *self, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  PyObject *variables = NULL;
  PyObject *grad_variables = NULL;
  unsigned char keep_graph = 0;
  PyObject *inputs = NULL;
  unsigned char only_inputs = 0;
  unsigned char allow_unreachable = 0;
  const char *accepted_kwargs[] = {"variables", "grad_variables",
      "keep_graph", "inputs", "only_inputs", "allow_unreachable", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOb|Obb", (char**)accepted_kwargs,
        &variables, &grad_variables, &keep_graph, &inputs, &only_inputs, &allow_unreachable))
    return NULL;

  THPUtils_assert(PyTuple_Check(variables), "variables argument is expected to "
      "be a tuple, but got %s", THPUtils_typename(variables));
  THPUtils_assert(PyTuple_Check(grad_variables), "variables argument is "
      "expected to be a tuple, but got %s", THPUtils_typename(grad_variables));

  Py_ssize_t num_variables = PyTuple_GET_SIZE(variables);
  Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_variables);
  THPUtils_assert(num_variables == num_gradients, "got %ld variables and %ld "
      "gradients", num_variables, num_gradients);

  function_list roots(num_variables);
  variable_list grads(num_variables);
  for (int i = 0; i < num_variables; i++) {
    PyObject *_variable = PyTuple_GET_ITEM(variables, i);
    THPUtils_assert(THPVariable_Check(_variable), "element %d of variables "
        "tuple is not a Variable", i);
    auto& variable = ((THPVariable*)_variable)->cdata;
    THPUtils_assert(!variable.is_volatile(),
        "element %d of variables tuple is volatile", i);
    // If grad_fn is NULL (as is the case for a leaf node), we instead
    // interpret the gradient function to be a grad accumulator,
    // which will accumulate its inputs into the grad property of the
    // variable. These nodes get suppressed in some situations,
    // see "suppress grad accumulation" below. Note that only variables which
    // have requires_grad=True can have grad accumulators.
    auto grad_fn = variable.grad_fn() ? variable.grad_fn() : variable.grad_accumulator();
    int output_nr = variable.grad_fn() ? variable.output_nr() : 0;
    THPUtils_assert(!variable.is_volatile(),
        "element %d of variables tuple is volatile", i);
    THPUtils_assert(grad_fn,
        "element %d of variables does not require grad and does not have a grad_fn", i);
    roots[i] = std::make_pair<>(std::move(grad_fn), output_nr);

    PyObject *grad = PyTuple_GET_ITEM(grad_variables, i);
    if (THPVariable_Check(grad)) {
      grads[i] = ((THPVariable*)grad)->cdata;
    } else {
      THPUtils_assert(grad == Py_None,
          "element %d of gradients tuple is not a Variable or None", i);
      THPUtils_assert(!variable.requires_grad(),
          "element %d of gradients tuple is None, but the corresponding Variable requires grad");
    }
  }

  Engine::pre_callback_map callbacks;
  CallbackContext ctx;
  if (inputs != NULL) {
    THPUtils_assert(PyTuple_Check(inputs), "inputs argument has to be a tuple");
    int num_inputs = PyTuple_GET_SIZE(inputs);
    ctx.outputs = PyTuple_New(num_inputs);
    if (!ctx.outputs) return NULL;
    // First, find all relevant functions and fill ctx.output_map
    for (int i = 0; i < num_inputs; ++i) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(THPVariable_Check(input),
          "all inputs have to be Variables, but got %s", THPUtils_typename(input));
      THPVariable *input_var = (THPVariable*)input;
      auto grad_fn = input_var->cdata.grad_fn();
      int output_nr = input_var->cdata.output_nr();
      bool is_leaf = !grad_fn;
      if (is_leaf) {
          grad_fn = input_var->cdata.get()->grad_accumulator.lock();
      }
      THPUtils_assert(input_var->cdata.requires_grad(),
          "One of the differentiated Variables does not require grad");
      if (allow_unreachable && !grad_fn) continue;
      THPUtils_assert(grad_fn,
          "One of the differentiated Variables appears to not have been used in the graph");
      THPUtils_assert(grad_fn->is_executable,
          "One of the differentiated Variables has a non-executable grad_fn. Submit a bug report.");
      auto& fn_info = ctx.output_map[grad_fn];
      fn_info.first.emplace_back(output_nr, i);
      fn_info.second = is_leaf;
    }
    // Register callbacks that will gather the outputs
    for (auto& entry : ctx.output_map) {
      auto& fn_info = entry.second;
      callbacks.emplace(entry.first.get(), [&ctx, &fn_info](Function* _unused, variable_list& grads) {
        auto& saved_outputs = fn_info.first;
        bool is_leaf = fn_info.second;
        AutoGIL gil;
        for (auto& saved_out : saved_outputs) {
          PyTuple_SET_ITEM(ctx.outputs.get(), saved_out.second,
            THPVariable_Wrap(grads[saved_out.first]));
        }
        // Suppress grad accumulation.
        // If the variable is a leaf, the next function to execute
        // is a grad_accumulator.  But when inputs != NULL, we should
        // NOT accumulate, so terminate execution.
        return !is_leaf;
      });
    }
    // Disable execution for all unneeded functions
    if (only_inputs) {
      compute_partial_exec_callbacks(roots, ctx, callbacks, allow_unreachable);
    }
  }

  {
    AutoNoGIL no_gil;
    engine.execute(roots, grads, keep_graph, callbacks);
  }

  if (ctx.outputs) {
    for (int i = 0; i < PyTuple_GET_SIZE(inputs); i++) {
      // XXX: initializing tuples with NULL pointers might be a CPython
      // implementation detail
      if (PyTuple_GET_ITEM(ctx.outputs.get(), i)) continue;
      Py_INCREF(Py_None);
      PyTuple_SET_ITEM(ctx.outputs.get(), i, Py_None);
    }
    return ctx.outputs.release();
  } else {
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEngine_queue_callback(PyObject *self, PyObject *_callback) {
  std::shared_ptr<PyObject> callback(_callback, [](PyObject *obj) { AutoGIL gil; Py_DECREF(obj); });
  Py_INCREF(_callback);
  engine.queue_callback([callback]() {
    AutoGIL gil;
    THPObjectPtr result {PyObject_CallFunctionObjArgs(callback.get(), NULL)};
    if (!result) throw python_error();
  });
  Py_RETURN_NONE;
}

PyObject *THPEngine_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  return type->tp_alloc(type, 0);
}

static struct PyMethodDef THPEngine_methods[] = {
  {(char*)"run_backward", (PyCFunction)THPEngine_run_backward, METH_VARARGS | METH_KEYWORDS, NULL},
  {(char*)"queue_callback", (PyCFunction)THPEngine_queue_callback, METH_O, NULL},
  {NULL}
};


PyTypeObject THPEngineType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._EngineBase",                /* tp_name */
  sizeof(THPEngine),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPEngine_methods,                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPEngine_new                          /* tp_new */
};

bool THPEngine_initModule(PyObject *module)
{
  if (PyType_Ready(&THPEngineType) < 0)
    return false;
  Py_INCREF(&THPEngineType);
  PyModule_AddObject(module, "_ImperativeEngine", (PyObject *)&THPEngineType);
  return true;
}
