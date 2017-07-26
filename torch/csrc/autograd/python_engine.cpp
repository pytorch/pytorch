#include "torch/csrc/autograd/python_engine.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/python_function.h"
#include "torch/csrc/autograd/jit_closure.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/PtrWrapper.h"
#include "torch/csrc/utils/auto_gil.h"

#include <unordered_set>

using namespace torch::autograd;

struct THPEngine {
    PyObject_HEAD
};

struct PythonEngine : public Engine {
  virtual void thread_main(std::shared_ptr<ReadyQueue> queue, int device) override {
    // Create a PyThreadState, but release the GIL. This lets AutoGIL calls
    // inside thread_main acquire the GIL without having to create a new
    // PyThreadState each time.
    AutoGIL gil;
    AutoNoGIL no_gil;
    Engine::thread_main(queue, device);
  }

  virtual void thread_on_exception(FunctionTask& task, std::exception& e) override {
    auto python_err = dynamic_cast<python_error*>(&e);
    if (python_err) {
      python_err->persist();
    }
    Engine::thread_on_exception(task, e);
  }
};

static PythonEngine engine;

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
                                    Engine::callback_map& map) {
  // This callback is used to suppress the computation of a node
  // if it is not necessary.
  static Engine::callback_type abort_callback(
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
    auto& rev_edges = rev_graph[input];
    if (rev_edges.size() == 0) throw std::runtime_error("differentiated input is unreachable");
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

PyObject *THPEngine_run_forward(THPEngine *self, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  PyObject *pyclosure = NULL;
  PyObject *inputs = NULL;
  const char *accepted_kwargs[] = {"closure", "inputs", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char**)accepted_kwargs,
        &pyclosure, &inputs))
    return NULL;

  THPUtils_assert(THPWrapper_check(pyclosure), "closure should be a PtrWrapper object");
  THPUtils_assert(PyTuple_Check(inputs), "inputs should be a tuple");

  variable_list var_inputs;
  auto num_inputs = PyTuple_GET_SIZE(inputs);
  var_inputs.reserve(1 + num_inputs);
  var_inputs.emplace_back(nullptr); // For ConstantFactory
  for (int i = 0; i < num_inputs; ++i) {
    PyObject *input = PyTuple_GET_ITEM(inputs, i);
    THPUtils_assert(THPVariable_Check(input), "%d input is not a Variable", i);
    var_inputs.emplace_back(((THPVariable*)input)->cdata);
  }

  AutogradClosure *closure = reinterpret_cast<AutogradClosure*>(THPWrapper_get(pyclosure));

  variable_list outputs;
  Engine::callback_map callbacks;
  callbacks.emplace(closure->output.get(), [&outputs](Function* _unused, variable_list& inputs) -> bool {
    outputs = inputs;
    return false;
  });

  try {
    AutoNoGIL no_gil;
    engine.execute(closure->roots, var_inputs, true, callbacks);
  } catch (python_error &e) {
    e.restore();
    return nullptr;
  }

  int num_outputs = outputs.size();
  THPObjectPtr pyoutputs { PyTuple_New(num_outputs) };
  for (int i = 0; i < num_outputs; ++i) {
    PyTuple_SET_ITEM(pyoutputs.get(), i, THPVariable_Wrap(outputs[i]));
  }
  return pyoutputs.release();
  END_HANDLE_TH_ERRORS
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
  const char *accepted_kwargs[] = {"variables", "grad_variables",
      "keep_graph", "inputs", "only_inputs", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOb|Ob", (char**)accepted_kwargs,
        &variables, &grad_variables, &keep_graph, &inputs, &only_inputs))
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
    THPUtils_assert(!variable->is_volatile,
        "element %d of variables tuple is volatile", i);
    // If grad_fn is NULL (as is the case for a leaf node), we instead
    // interpret the gradient function to be a grad accumulator,
    // which will accumulate its inputs into the grad property of the
    // variable.  These nodes get suppressed in some situations,
    // see "suppress grad accumulation" below.
    auto grad_fn = variable->grad_fn ? variable->grad_fn : variable->get_grad_accumulator();
    THPUtils_assert(grad_fn, "element %d of variables tuple does not require grad", i);
    int output_nr = variable->grad_fn ? variable->output_nr : 0;
    roots[i] = std::make_pair<>(std::move(grad_fn), output_nr);

    PyObject *grad = PyTuple_GET_ITEM(grad_variables, i);
    if (THPVariable_Check(grad)) {
      grads[i] = ((THPVariable*)grad)->cdata;
    } else {
      THPUtils_assert(grad == Py_None,
          "element %d of gradients tuple is not a Variable or None", i);
      THPUtils_assert(!variable->requires_grad,
          "element %d of gradients tuple is None, but the corresponding Variable requires grad");
    }
  }

  Engine::callback_map callbacks;
  CallbackContext ctx;
  if (inputs != NULL) {
    THPUtils_assert(PyTuple_Check(inputs), "inputs argument has to be a tuple");
    int num_inputs = PyTuple_GET_SIZE(inputs);
    ctx.outputs = PyTuple_New(num_inputs);
    // First, find all relevant functions and fill ctx.output_map
    for (int i = 0; i < num_inputs; ++i) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(THPVariable_Check(input),
          "all inputs have to be Variables, but got %s", THPUtils_typename(input));
      THPVariable *input_var = (THPVariable*)input;
      auto grad_fn = input_var->cdata->grad_fn;
      int output_nr = input_var->cdata->output_nr;
      bool is_leaf = !grad_fn;
      if (is_leaf) {
          grad_fn = input_var->cdata->grad_accumulator.lock();
      }
      THPUtils_assert(grad_fn, "One of the differentiated Variables appears to not have "
          "been used in the graph");
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
      compute_partial_exec_callbacks(roots, ctx, callbacks);
    }
  }

  try {
    AutoNoGIL no_gil;
    engine.execute(roots, grads, keep_graph, callbacks);
  } catch (python_error &e) {
    e.restore();
    return nullptr;
  }

  if (ctx.outputs) {
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
  {(char*)"run_forward", (PyCFunction)THPEngine_run_forward, METH_VARARGS | METH_KEYWORDS, NULL},
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
