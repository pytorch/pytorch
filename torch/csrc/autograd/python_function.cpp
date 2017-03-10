#include "torch/csrc/autograd/python_function.h"

#include <Python.h>
#include <structmember.h>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <THPP/THPP.h>

#include "THP.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/autograd/python_hook.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/Exceptions.h"

#ifdef WITH_CUDA
#include "cuda/AutoGPU.h"
#endif

using namespace torch;
using namespace torch::autograd;
using thpp::Tensor;

PyObject *THPFunctionClass = NULL;
PyObject *THPStochasticFunctionClass = NULL;


#define THPFunction_assert(condition, ...)                                     \
  if (!(condition)) { THPUtils_setError(__VA_ARGS__); throw python_error(); }


namespace torch { namespace autograd {

auto PyFunction::apply(const variable_list& gradOutputs) -> variable_list {
  AutoGIL gil;

  THPObjectPtr pyGradOutputs = PyTuple_New(gradOutputs.size());
  if (!pyGradOutputs) throw python_error();

  for (size_t i = 0; i != gradOutputs.size(); ++i) {
    PyObject* gradOutput;
    if (gradOutputs[i]) {
      gradOutput = createPyObject(*gradOutputs[i]->data);
      if (!gradOutput) throw python_error();
    } else {
      gradOutput = Py_None;
      Py_INCREF(gradOutput);
    }
    PyTuple_SET_ITEM(pyGradOutputs.get(), i, gradOutput);
  }

  THPObjectPtr r = PyObject_CallMethod(
      obj, "_do_backward", "OO", pyGradOutputs.get(), Py_True);
  if (!r) throw python_error();

  auto num_outputs = PyTuple_GET_SIZE(r.get());
  variable_list results(num_outputs);
  for (int i = 0; i != num_outputs; ++i) {
    PyObject* obj = PyTuple_GET_ITEM(r.get(), i);
    if (obj != Py_None) {
      if (!THPModule_isTensor(obj)) {
        std::string msg("expected Tensor (got '");
        msg += THPUtils_typename(obj);
        msg += "')'";
        throw std::runtime_error(msg);
      }
      results[i] = std::make_shared<Variable>(createTensor(obj), false, true);
    }
  }

  return results;
}

auto PyFunction::releaseVariables() -> void {
  AutoGIL gil;
  auto f = (THPFunction*) obj;
  delete f->saved_variables;
  f->saved_variables = nullptr;
  f->has_freed_buffers = 1;
}

auto PyFunction::name() -> std::string {
  AutoGIL gil;
  auto f = (THPFunction*) obj;
  return std::string(Py_TYPE(f)->tp_name);
}

}} // namespace torch::autograd

// Traverse and clear are required for supporting Python's GC cycle handling.
static int THPFunction_traverse(THPFunction *self, visitproc visit, void *arg)
{
  Py_VISIT(self->needs_input_grad);
  if (self->saved_variables) {
    for (unsigned int i = 0; i < self->saved_variables->size(); i++)
      Py_VISIT(std::get<0>(self->saved_variables->at(i)));
  }
  for (auto& hook : self->cdata.pre_hooks) {
    if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
      Py_VISIT(pyhook->dict);
    }
  }
  for (auto& hook : self->cdata.post_hooks) {
    if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
      Py_VISIT(pyhook->dict);
    }
  }
  Py_VISIT(self->to_save);
  Py_VISIT(self->shared_pairs);
  Py_VISIT(self->non_differentiable);
  Py_VISIT(self->dirty_tensors);
  return 0;
}

static int THPFunction_clear(THPFunction *self)
{
  self->num_inputs = 0;
  self->cdata.num_outputs = 0;

  Py_CLEAR(self->needs_input_grad);

  Py_CLEAR(self->to_save);
  Py_CLEAR(self->shared_pairs);
  Py_CLEAR(self->non_differentiable);
  Py_CLEAR(self->dirty_tensors);

  auto saved_variables = self->saved_variables;
  self->saved_variables = NULL;
  delete saved_variables;

  auto output_info = self->output_info;
  self->output_info = NULL;
  delete output_info;

  // clear pre and post hooks
  auto pre_hooks = std::move(self->cdata.pre_hooks);
  auto post_hooks = std::move(self->cdata.post_hooks);

  return 0;
}

static void THPFunction_dealloc(THPFunction* self)
{
  PyObject_GC_UnTrack(self);
  THPFunction_clear(self);
  self->cdata.~PyFunction();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *THPFunction_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (!obj)
    return NULL;
  // Python zero-initializes the object memory, so there's no need to initialize
  // most fields
  THPFunction* self = (THPFunction*)obj;
  new (&self->cdata) torch::autograd::PyFunction(obj);
  self->cdata.num_outputs = -1;
  self->cdata.is_stochastic = PyObject_IsInstance(obj, THPStochasticFunctionClass);
  return obj;
}

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

using t2var_type = std::unordered_map<PyObject *, THPVariable *>;

static void _mark_dirty(THPFunction *self, t2var_type &t2var,
        std::unordered_set<PyObject *> &dirty_inputs)
{
  // Increase versions of modified tensors
  if (!self->dirty_tensors) return;

  THPFunction_assert(PyTuple_Check(self->dirty_tensors), "autograd "
      "internal error: dirty_tensors attribute is expected to be a tuple "
      "but is %s", THPUtils_typename(self->dirty_tensors));
  Py_ssize_t num_dirty = PyTuple_GET_SIZE(self->dirty_tensors);
  for (int i = 0; i < num_dirty; i++) {
    PyObject *tensor = PyTuple_GET_ITEM(self->dirty_tensors, i);
    dirty_inputs.insert(tensor);
    THPVariable *variable;
    try {
      variable = t2var.at(tensor);
    } catch (std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(tensor), "mark_dirty can "
          "only accept tensors, but argument %d is of type %s", i,
          THPUtils_typename(tensor));
      THPFunction_assert(false, "mark_dirty only accepts input tensors, but "
          "argument %d isn't one", i);
    }
    auto &v_counter = *variable->cdata->version_counter;
    THPFunction_assert(v_counter.var_refcnt() == 1, "in-place operations can be "
        "only used on variables that don't share storage with any other "
        "variables, but detected that there are %d objects sharing it",
        v_counter.var_refcnt());
    v_counter++;
  }
  // We're not going to ever need this so let's remove references now
  Py_DECREF(self->dirty_tensors);
  self->dirty_tensors = NULL;
}

static void _wrap_outputs(THPFunction *self, t2var_type &t2var,
    std::unordered_set<PyObject *> &dirty_inputs, PyObject *raw_output,
    PyObject *outputs)
{
  // Wrap outputs in Variables
  Py_ssize_t num_outputs = PyTuple_GET_SIZE(raw_output);
  self->output_info = new std::vector<output_info_type>(num_outputs);
  auto &output_info = *self->output_info;
  for (int i = 0; i < num_outputs; i++) {
    PyObject *output = PyTuple_GET_ITEM(raw_output, i);
    THPVariable *output_var;
    auto it = t2var.find(output);
    if (it == t2var.end()) {
      // A completely new tensor - just wrap it and continue
      output_var = (THPVariable*)THPVariable_New(output, (PyObject*)self, self->cdata.requires_grad);
    } else {
      // If one of the outputs was also an input tensor it's a bit more complicated.
      THPVariable *input_var = it->second;
      auto& input_var_ = *input_var->cdata;
      if (input_var_.creator) {
        // If it's not a leaf we want to move it in the graph so backprop
        // will be computed correctly:
        // creator <- variable <- self  ==>  creator <- self <- variable
        Py_INCREF(input_var);
        output_var = input_var;
        input_var_.creator = THPFunction_asFunction(self);
        input_var_.requires_grad = self->cdata.requires_grad;
      } else {
        // If the leaf Variable has been returned, we have to move it after the
        // current function to ensure the gradient is computed correctly.
        // There are two cases now:
        // 1. It has been modified in-place. If it didn't require_grad it's ok,
        // but if it does, then it's a clear error.
        // 2. It hasn't been modified. This means that it must have been
        // returned unchanged, and we can simply return a new Variable
        // referencing the same storage.
        if (dirty_inputs.count(output) > 0) {
          Py_INCREF(input_var);
          output_var = input_var;
          auto& output_var_ = *output_var->cdata;
          output_var_.creator = THPFunction_asFunction(self);
          if (!output_var_.requires_grad) {
            // Now, there's another subtlety. We move the input in the graph
            // and possibly change its requires_grad to True. However, remember
            // that we're still holding a reference to is as a previous
            // function. Backward engine will think that it was really a
            // leaf that initialy did require grad and call its _do_backward
            // and that will throw. Because of this, we need to allocate
            // a dummy leaf that doesn't require grad and put it as our
            // previous function.
            // Even if the function doesn't require grad, creating a dummy leaf
            // prevents the creation of reference cycles.
            output_var_.requires_grad = self->cdata.requires_grad;
            auto dummy_prev_fn = std::make_shared<Variable>(
                std::unique_ptr<Tensor>(output_var_.data->clone_shallow()), false, false);
            // Replace all references to the variable
            auto& previous_functions = self->cdata.previous_functions;
            for (int inp = 0; inp < self->num_inputs; inp++) {
              if (previous_functions[inp].first.get() == &output_var_) {
                previous_functions[inp] = std::make_pair<>(dummy_prev_fn, 0);
              }
            }
          } else { // output_var_.requires_grad
            throw std::runtime_error("a leaf Variable that requires grad has been used in an in-place operation.");
          }
        } else {
          // An input has been returned, but it wasn't modified. It's better
          // not to move the Variable, because there are some legitimate cases
          // where making it non-leaf would break stuff (e.g. broadcast). Also,
          // returning the input Variable is not a good option either,
          // because if someone registers hooks on it, they will fire with grads
          // from all usages, not only from usages of this output. This is why
          // we'll return a copy and join their version counters. This has
          // a side-effect of making in-place ops on any of these Variables an
          // immediate error, but it would be raised anyway once someone
          // calls backward.
          output_var = (THPVariable*)THPVariable_New(output, (PyObject*)self,
                  self->cdata.requires_grad);
          if (!output_var) throw python_error();
          output_var->cdata->version_counter->join_with(*input_var->cdata->version_counter);
        }
      }
    }
    if (!output_var) throw python_error();

    auto& output_tensor = *output_var->cdata->data;
    output_info[i] = std::make_tuple(
      (PyObject *)getPyTypeObject(output_tensor),
      output_tensor.getDevice(),
      output_tensor.sizes()
    );
    t2var[output] = output_var;
    output_var->cdata->output_nr = i;
    PyTuple_SET_ITEM(outputs, i, (PyObject*)output_var);
  }
}

static void _save_variables(THPFunction*self, t2var_type &t2var)
{
  if (!self->to_save) return;

  THPFunction_assert(PyTuple_Check(self->to_save), "autograd internal "
      "error: to_save attribute is expected to be a tuple but is %s",
      THPUtils_typename(self->to_save));
  Py_ssize_t num_saved = PyTuple_GET_SIZE(self->to_save);
  self->saved_variables = new std::vector<saved_var_info_type>();
  self->saved_variables->reserve(num_saved);
  for (int i = 0; i < num_saved; i++) {
    PyObject *tensor = PyTuple_GET_ITEM(self->to_save, i);
    if (tensor == Py_None) {
      Py_INCREF(tensor);
      self->saved_variables->emplace_back(tensor, 0, nullptr);
      continue;
    }

    THPVariable *variable;
    try {
      variable = t2var.at(tensor);
    } catch(std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(tensor),
          "save_for_backward can only save tensors, but argument %d is of "
          "type %s", i, THPUtils_typename(tensor));
      THPFunction_assert(false, "save_for_backward can only save input or output "
          "tensors, but argument %d doesn't satisfy this condition", i);
    }

    Py_INCREF(tensor);
    self->saved_variables->emplace_back(
      tensor,
      **variable->cdata->version_counter,
      std::unique_ptr<VariableVersion>(variable->cdata->version_counter->new_saved_ref())
    );
  }
  // Free .to_save
  Py_DECREF(self->to_save);
  self->to_save = NULL;
}

static void _join_version_counters(THPFunction *self, t2var_type &t2var)
{
  if (!self->shared_pairs) return;
  THPFunction_assert(PyTuple_Check(self->shared_pairs), "autograd internal "
      "error: shared_pairs attribute is expected to be a tuple but is %s",
      THPUtils_typename(self->shared_pairs));
  Py_ssize_t num_shared = PyTuple_GET_SIZE(self->shared_pairs);
  for (int i = 0; i < num_shared; i++) {
    PyObject *shared_tuple = PyTuple_GET_ITEM(self->shared_pairs, i);
    THPFunction_assert(PyTuple_Check(shared_tuple), "mark_shared_storages "
        "accepts a number of pairs, but one of the arguments is of type %s",
        THPUtils_typename(shared_tuple));
    THPFunction_assert(PyTuple_GET_SIZE(shared_tuple) == 2,
        "mark_shared_storages accepts pairs, but argument %d is a tuple of "
        "%d elements", i, PyTuple_GET_SIZE(shared_tuple));

    // Now we're sure it's really a pair!
    THPVariable *v1, *v2;
    try {
      v1 = t2var.at(PyTuple_GET_ITEM(shared_tuple, 0));
      v2 = t2var.at(PyTuple_GET_ITEM(shared_tuple, 1));
    } catch(std::out_of_range &e) {
      // One tuple items wasn't present in t2var, so there are two cases:
      // 1. it's not a tensor
      // 2. it's not an input nor an output
      PyObject *t1 = PyTuple_GET_ITEM(shared_tuple, 0);
      PyObject *t2 = PyTuple_GET_ITEM(shared_tuple, 1);
      THPFunction_assert(THPModule_isTensor(t1) && THPModule_isTensor(t2),
        "mark_shared_storages accepts pairs of tensors, but one of them "
        "contains %s and %s", THPUtils_typename(t1), THPUtils_typename(t2));
      THPFunction_assert(false, "mark_shared_storages only accepts pairs of input "
          "and output tensors, but argument %d doesn't satify this "
          "condition", i);
    }
    v2->cdata->version_counter->join_with(*v1->cdata->version_counter);
  }
  // Free .shared_pairs
  Py_DECREF(self->shared_pairs);
  self->shared_pairs = NULL;
}

static void _mark_non_differentiable(THPFunction *self, t2var_type &t2var)
{
  if (!self->non_differentiable) return;

  THPFunction_assert(PyTuple_Check(self->non_differentiable), "autograd "
      "internal error: non_differentiable attribute is expected to be a "
      "tuple but is %s", THPUtils_typename(self->non_differentiable));
  Py_ssize_t num_nondiff = PyTuple_GET_SIZE(self->non_differentiable);
  for (int i = 0; i < num_nondiff; i++) {
    PyObject *t = PyTuple_GET_ITEM(self->non_differentiable, i);
    THPVariable *var;
    try {
      var = t2var.at(t);
      auto tmp = &self->cdata;
      THPFunction_assert(var->cdata->creator.get() == tmp,
          "mark_non_differentiable only accepts output tensors, but "
          "argument %d isn't an output", i);
    } catch (std::out_of_range &e) {
      THPFunction_assert(THPModule_isTensor(t), "mark_non_differentiable "
          "only accepts tensor arguments, but got %s", THPUtils_typename(t));
      THPFunction_assert(false, "mark_non_differentiable only accepts function "
          "outputs");
    }
    var->cdata->requires_grad = 0;
  }
  Py_DECREF(self->non_differentiable);
  self->non_differentiable = NULL;
}

static bool _ensure_tuple(THPObjectPtr& obj)
{
  if (PyTuple_Check(obj.get()))
    return false;

  PyObject *tuple = PyTuple_New(1);
  if (!tuple) throw python_error();
  PyTuple_SET_ITEM(tuple, 0, obj.release());
  obj = tuple;
  return true;
}

PyObject *THPFunction_do_forward(THPFunction *self, PyObject *inputs)
{
  try {
    Py_ssize_t num_inputs = inputs ? PyTuple_GET_SIZE(inputs) : 0;

    // Unpack inputs and check if they require gradients or are volatile
    THPObjectPtr unpacked_inputs = PyTuple_New(num_inputs);
    self->needs_input_grad = PyTuple_New(num_inputs);
    self->cdata.requires_grad = false;
    bool is_volatile = false;
    for (int i = 0; i < num_inputs; i++) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(THPVariable_Check(input), "expected a Variable argument, "
          "but got %s", THPUtils_typename(input));
      THPVariable *variable = (THPVariable*)input;

      // Unpack the variable
      PyTuple_SET_ITEM(unpacked_inputs.get(), i, THPVariable_get_data(variable));

      // We can't move this to C, because it's going to be accessed from user code.
      PyTuple_SET_ITEM(self->needs_input_grad, i, PyBool_FromLong(variable->cdata->requires_grad));

      is_volatile = is_volatile || variable->cdata->is_volatile;
      self->cdata.requires_grad = self->cdata.requires_grad || variable->cdata->requires_grad;
    }

    // Now we're ready to call a forward (implemented in Python)
    THPObjectPtr forward_fn = PyObject_GetAttrString((PyObject*)self, "forward");
    THPUtils_assert(forward_fn.get(), "function %s doesn't implement a required "
        "'forward' method", THPUtils_typename((PyObject*)self));
    THPObjectPtr raw_output = PyObject_CallObject(forward_fn, unpacked_inputs);
    if (!raw_output) return NULL;
    // Wrap output in a tuple, if it's not one already
    bool unpack_output = _ensure_tuple(raw_output);
    int num_outputs = PyTuple_GET_SIZE(raw_output.get());


    THPObjectPtr outputs = PyTuple_New(num_outputs);
    if (!outputs) return NULL;
    if (is_volatile) {
      // If one of the inputs is volatile let's take a fast path - we want
      // minimize the overhead of inference
      for (int i = 0; i < num_outputs; i++) {
        PyObject *output = PyTuple_GET_ITEM(raw_output.get(), i);
        THPVariable *output_var = (THPVariable*)THPVariable_NewVolatile(output);
        if (!output_var) return NULL;
        output_var->cdata->output_nr = i;
        PyTuple_SET_ITEM(outputs.get(), i, (PyObject*)output_var);
      }
    } else {
      // We're not volatile, so there's a lot of bookkeeping to do...
      self->num_inputs = num_inputs;
      self->cdata.num_outputs = num_outputs;
      t2var_type t2var;

      // Save previous functions and initialize t2var map
      self->cdata.previous_functions.resize(num_inputs);
      for (int i = 0; i < num_inputs; i++) {
        THPVariable *input_var = (THPVariable*)PyTuple_GET_ITEM(inputs, i);
        PyObject *input_tensor = PyTuple_GET_ITEM(unpacked_inputs.get(), i);
        t2var.emplace(input_tensor, input_var);

        // Save previous function
        std::shared_ptr<Function> prev_fn;
        if (input_var->cdata->creator) {
          prev_fn = input_var->cdata->creator;
        } else {
          prev_fn = input_var->cdata;
        }
        self->cdata.previous_functions[i] = std::make_pair<>(prev_fn, input_var->cdata->output_nr);
      }

      std::unordered_set<PyObject *> dirty_inputs;
      _mark_dirty(self, t2var, dirty_inputs);
      _wrap_outputs(self, t2var, dirty_inputs, raw_output, outputs);
      _join_version_counters(self, t2var);
      if (self->cdata.requires_grad || self->cdata.is_stochastic) {
        _save_variables(self, t2var);
        _mark_non_differentiable(self, t2var);
      } else {
        // Remove unnecessary attributes
        Py_XDECREF(self->to_save);
        self->to_save = NULL;
        Py_XDECREF(self->non_differentiable);
        self->non_differentiable = NULL;
      }
    }

    // Unpack the output, unless .forward() returned a tuple
    if (unpack_output) {
      PyObject *output = PyTuple_GET_ITEM(outputs.get(), 0);
      Py_INCREF(output);
      return output;
    }

    return outputs.release();

  } catch (python_error& e) {
    return NULL;
  } catch (std::exception& e) {
    THPUtils_setError(e.what());
    return NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Backward
////////////////////////////////////////////////////////////////////////////////

static void _prepare_grad_output(THPFunction *self, THPObjectPtr& raw_grad_output)
{
#ifdef WITH_CUDA
  THCPAutoGPU gpu_guard(-1);
#endif
  int num_grad_output = PyTuple_GET_SIZE(raw_grad_output.get());
  // First, check if any of grad_outputs is None. If not, there's nothing to do
  bool has_none = false;
  for (int i = 0; i < num_grad_output; i++) {
    if (PyTuple_GET_ITEM(raw_grad_output.get(), i) == Py_None) {
      has_none = true;
      break;
    }
  }
  if (!has_none)
      return;

  THPObjectPtr grad_output;
  grad_output = PyTuple_New(num_grad_output);
  if (!grad_output) throw python_error();

  // Look for Nones and replace them with new buffers
  for (int i = 0; i < num_grad_output; i++) {
    PyObject *grad = PyTuple_GET_ITEM(raw_grad_output.get(), i);
    if (grad == Py_None) {
      auto &info = (*self->output_info)[i];
      PyObject *tensor_cls = std::get<0>(info);
#ifdef WITH_CUDA
      gpu_guard.setDevice(std::get<1>(info));
#endif
      std::vector<long> &sizes = std::get<2>(info);
      THPObjectPtr grad_size = THPSize_New(sizes.size(), sizes.data());
      THPObjectPtr new_grad = PyObject_CallFunctionObjArgs(tensor_cls, grad_size.get(), NULL);
      if (!new_grad) throw python_error();
      THPObjectPtr result = PyObject_CallMethod(new_grad.get(), "zero_", "");
      if (!result) throw python_error();
      grad = new_grad.release();
    } else {
      Py_INCREF(grad);
    }
    PyTuple_SET_ITEM(grad_output.get(), i, grad);
  }
  raw_grad_output = grad_output.release();
}

static void _trim_grad_input(THPFunction *self, THPObjectPtr& grad_input)
{
  int num_grads = PyTuple_GET_SIZE(grad_input.get());
  int num_prev_fns = self->num_inputs;
  if (num_grads > num_prev_fns) {
    // Check that all extra grads are none
    bool all_none = true;
    for (int i = num_prev_fns; i < num_grads; i++) {
      all_none = (PyTuple_GET_ITEM(grad_input.get(), i) == Py_None);
      if (!all_none) break;
    }
    // If yes, slice the tuple
    if (all_none) {
      num_grads = num_prev_fns;
      grad_input = PyTuple_GetSlice(grad_input.get(), 0, num_grads);
      if (!grad_input) throw python_error();
    }
  }
}

PyObject * THPFunction_do_backward(THPFunction *self, PyObject *args)
{
  try {
    Py_ssize_t num_args = args ? PyTuple_GET_SIZE(args) : 0;
    THPUtils_assert(num_args == 2, "_do_backward expects exactly two arguments");
    PyObject *raw_grad_output = PyTuple_GET_ITEM(args, 0);
    PyObject *retain_variables = PyTuple_GET_ITEM(args, 1);
    if (!PyTuple_Check(raw_grad_output) || !PyBool_Check(retain_variables)) {
      THPUtils_invalidArguments(args, NULL, "_do_backward", 1, "(tuple, bool)");
      return NULL;
    }

    // Some of the output might have been unused, so we have to allocate
    // zero-filled buffers instead
    Py_INCREF(raw_grad_output);
    THPObjectPtr grad_output = raw_grad_output;
    _prepare_grad_output(self, grad_output);

    // self.backward(*grad_output)
    THPObjectPtr backward_fn = PyObject_GetAttrString((PyObject*)self, "backward");
    THPUtils_assert(backward_fn.get(), "function %s doesn't implement a required "
        "'backward' method", THPUtils_typename((PyObject*)self));
    THPObjectPtr grad_input = PyObject_CallObject(backward_fn, grad_output.get());
    if (!grad_input) return NULL;
    _ensure_tuple(grad_input);

    // We allow functions to return more gradients, than there were outputs,
    // if and only if the additional ones are all None
    _trim_grad_input(self, grad_input);
    int num_grads = PyTuple_GET_SIZE(grad_input.get());
    int num_prev_fns = self->num_inputs;
    THPUtils_assert(num_grads == num_prev_fns, "%s returned an invalid number of "
        "gradient tensors (expected %d, but got %d)", THPUtils_typename(self),
        num_prev_fns, num_grads);

    return grad_input.release();

  } catch (python_error& e) {
    return NULL;
  } catch (std::exception& e) {
    THPUtils_setError(e.what());
    return NULL;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Other methods / attributes
////////////////////////////////////////////////////////////////////////////////

PyObject* THPFunction__register_hook_dict(THPFunction *self, PyObject *_var)
{
  THPUtils_assert(THPVariable_Check(_var), "_register_hook_dict expected a variable");
  THPVariable *var = (THPVariable*)_var;
  self->cdata.pre_hooks.emplace_back(new PyFunctionPreHook(var->backward_hooks, var->cdata->output_nr));
  Py_RETURN_NONE;
}

PyObject* THPFunction_register_hook(THPFunction *self, PyObject *hook)
{
  return torch::autograd::registerFunctionHook(self->cdata, hook);
}

PyObject *THPFunction_saved_tensors(THPFunction *self, void *_unused)
{
  THPUtils_assert(!self->has_freed_buffers, "Trying to backward through the "
      "graph second time, but the buffers have already been freed. Please "
      "specify retain_variables=True when calling backward for the first time.");
  if (!self->saved_variables)
    return PyTuple_New(0);

  int num_saved = self->saved_variables->size();
  THPObjectPtr saved_tensors = PyTuple_New(num_saved);
  if (!saved_tensors)
    return NULL;
  for (int i = 0; i < num_saved; i++) {
    saved_var_info_type &tuple = (*self->saved_variables)[i];
    PyObject *tensor = std::get<0>(tuple);
    if (tensor != Py_None) {
      int expected_version = std::get<1>(tuple);
      int current_version = **(std::get<2>(tuple));
      THPUtils_assert(expected_version == current_version, "one of the variables "
          "needed for gradient computation has been modified by an "
          "inplace operation");
    }
    Py_INCREF(tensor);
    PyTuple_SET_ITEM(saved_tensors.get(), i, tensor);
  }
  return saved_tensors.release();
}

PyObject *THPFunction_previous_functions(THPFunction *self, void *_unused)
{
  auto& prev_fns = self->cdata.previous_functions;
  int size = prev_fns.size();
  THPObjectPtr result = PyTuple_New(size);
  if (!result)
    return NULL;
  for (int i = 0; i < size; i++) {
    THPObjectPtr fn_tuple = PyTuple_New(2);
    if (!fn_tuple) return NULL;
    PyObject* fn = functionToPyObject(prev_fns[i].first);
    if (!fn) return NULL;
    PyTuple_SET_ITEM(fn_tuple.get(), 0, fn);
    PyTuple_SET_ITEM(fn_tuple.get(), 1, PyInt_FromLong(prev_fns[i].second));
    PyTuple_SET_ITEM(result.get(), i, fn_tuple.release());
  }
  return result.release();
}


typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

namespace {

template<PyObject* THPFunction::*ptr>
PyObject* getObject(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  PyObject* value = self->*ptr;
  if (!value) {
    Py_RETURN_NONE;
  }
  Py_INCREF(value);
  return value;
}

template<PyObject* THPFunction::*ptr>
int setObject(PyObject* obj, PyObject* value, void* _unused) {
  auto self = (THPFunction*)obj;
  if (value == Py_None) {
    value = nullptr;
  }
  Py_XDECREF(self->*ptr);
  Py_XINCREF(value);
  self->*ptr = value;
  return 0;
}

template<typename M, M THPFunction::*ptr, PyObject* (*Convert)(long)>
PyObject* getMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->*ptr);
}

template<typename M, M Function::*ptr, PyObject* (*Convert)(long)>
PyObject* getImplMember(PyObject* obj, void* _unused) {
  auto self = (THPFunction*)obj;
  return Convert(self->cdata.*ptr);
}

int setRequiresGrad(PyObject* obj, PyObject* value, void* _unused) {
  auto self = (THPFunction*)obj;
  if (!PyBool_Check(value)) {
    PyErr_Format(PyExc_TypeError, "'requires_grad' must be a bool");
    return -1;
  }
  self->cdata.requires_grad = (value == Py_True);
  return 0;
}

}

static struct PyGetSetDef THPFunction_properties[] = {
  {"saved_tensors", (getter)THPFunction_saved_tensors, NULL, NULL, NULL},
  {"previous_functions", (getter)THPFunction_previous_functions, NULL, NULL, NULL},
  {"to_save", &getObject<&THPFunction::to_save>, &setObject<&THPFunction::to_save>, NULL, NULL},
  {"shared_pairs", &getObject<&THPFunction::shared_pairs>, &setObject<&THPFunction::shared_pairs>, NULL, NULL},
  {"non_differentiable", &getObject<&THPFunction::non_differentiable>, &setObject<&THPFunction::non_differentiable>, NULL, NULL},
  {"dirty_tensors", &getObject<&THPFunction::dirty_tensors>, &setObject<&THPFunction::dirty_tensors>, NULL, NULL},
  {"needs_input_grad", &getObject<&THPFunction::needs_input_grad>, &setObject<&THPFunction::needs_input_grad>, NULL, NULL},
  {"requires_grad", &getImplMember<bool, &Function::requires_grad, PyBool_FromLong>, &setRequiresGrad, NULL, NULL},
  {"num_inputs", &getMember<int, &THPFunction::num_inputs, PyInt_FromLong>, NULL, NULL, NULL},
  {"num_outputs", &getImplMember<int, &Function::num_outputs, PyInt_FromLong>, NULL, NULL, NULL},
  {NULL}
};

static struct PyMethodDef THPFunction_methods[] = {
  {(char*)"_do_forward", (PyCFunction)THPFunction_do_forward, METH_VARARGS, NULL},
  {(char*)"_do_backward", (PyCFunction)THPFunction_do_backward, METH_VARARGS, NULL},
  {(char*)"_register_hook_dict", (PyCFunction)THPFunction__register_hook_dict, METH_O, NULL},
  {(char*)"register_hook", (PyCFunction)THPFunction_register_hook, METH_O, NULL},
  {NULL}
};

PyTypeObject THPFunctionType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._FunctionBase",              /* tp_name */
  sizeof(THPFunction),                   /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPFunction_dealloc,       /* tp_dealloc */
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
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPFunction_traverse,    /* tp_traverse */
  (inquiry)THPFunction_clear,            /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPFunction_methods,                   /* tp_methods */
  0,                                     /* tp_members */
  THPFunction_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPFunction_new                        /* tp_new */
};

bool THPFunction_initModule(PyObject *module)
{
  if (PyType_Ready(&THPFunctionType) < 0)
    return false;
  Py_INCREF(&THPFunctionType);
  PyModule_AddObject(module, "_FunctionBase", (PyObject *)&THPFunctionType);
  return true;
}

struct Decref {
  void operator()(PyFunction* p) const {
    AutoGIL gil;
    Py_DECREF(p->obj);
  }
};

std::shared_ptr<PyFunction> THPFunction_asFunction(THPFunction* self)
{
  if (!self) {
    return std::shared_ptr<PyFunction>();
  }
  Py_INCREF((PyObject*)self);
  return std::shared_ptr<PyFunction>(&self->cdata, Decref());
}
