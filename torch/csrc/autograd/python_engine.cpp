#include "torch/csrc/autograd/python_engine.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/utils/auto_gil.h"

#include <unordered_set>

using namespace torch::autograd;

struct THPEngine {
    PyObject_HEAD
};

struct PythonEngine : public Engine {
  virtual void thread_main(ReadyQueue& queue) override {
    // Create a PyThreadState, but release the GIL. This lets AutoGIL calls
    // inside thread_main acquire the GIL without having to create a new
    // PyThreadState each time.
    AutoGIL gil;
    AutoNoGIL no_gil;
    Engine::thread_main(queue);
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
  std::mutex mutex;
  std::string error;
  THPObjectPtr outputs;
};

void compute_partial_exec_callbacks(const function_list& roots,
                                    const std::vector<Function*> inputs,
                                    Engine::callback_map& map) {
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
    queue.emplace_back(ptr);
    seen.insert(ptr);
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
  for (auto input: inputs) {
    auto& rev_edges = rev_graph[input];
    if (rev_edges.size() == 0) throw std::runtime_error("unreachable input");
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

  // Prevent expantion for functions in {all_vertices} \ {needed}
  for (auto fn : all_functions) {
    if (needed.count(fn) > 0) continue;
    map.emplace(fn, abort_callback);
  }
}

PyObject *THPEngine_run_backward(THPEngine *self, PyObject *args, PyObject *kwargs)
{
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
    auto grad_fn = variable->grad_fn ? variable->grad_fn : variable->get_grad_accumulator();
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
    THPUtils_assert(PyTuple_Check(inputs), "outputs argument has to be a tuple");
    int num_inputs = PyTuple_GET_SIZE(inputs);
    ctx.outputs = PyTuple_New(num_inputs);
    std::vector<Function*> grad_accumulators;
    for (int i = 0; i < num_inputs; ++i) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(THPVariable_Check(input),
          "all inputs have to be Variables, but got %s", THPUtils_typename(input));
      THPVariable *input_var = (THPVariable*)input;
      auto grad_acc = input_var->cdata->grad_accumulator.lock();
      // TODO: maybe just return a zero tensor?
      THPUtils_assert(grad_acc, "One of the differentiated Variables appears to not have "
          "been used in any computation");
      grad_accumulators.push_back(grad_acc.get());
      callbacks.emplace(grad_acc.get(), [&ctx, i](Function* _unused, variable_list& grads) {
        std::lock_guard<std::mutex> guard(ctx.mutex);
        if (grads.size() != 1) {
          ctx.error = "expected to get a single gradient, but got ";
          ctx.error += std::to_string(grads.size());
        }
        PyTuple_SET_ITEM(ctx.outputs.get(), i, THPVariable_Wrap(grads[0]));
        return false;
      });
    }
    // Disable execution for all unneeded functions
    if (only_inputs) {
      compute_partial_exec_callbacks(roots, grad_accumulators, callbacks);
    }
  }

  try {
    AutoNoGIL no_gil;
    engine.execute(roots, grads, keep_graph, callbacks);
  } catch (python_error &e) {
    e.restore();
    return nullptr;
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }

  if (ctx.outputs) {
    return ctx.outputs.release();
  } else {
    Py_RETURN_NONE;
  }
}

PyObject *THPEngine_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  return type->tp_alloc(type, 0);
}

static struct PyMethodDef THPEngine_methods[] = {
  {(char*)"run_backward", (PyCFunction)THPEngine_run_backward, METH_VARARGS | METH_KEYWORDS, NULL},
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
