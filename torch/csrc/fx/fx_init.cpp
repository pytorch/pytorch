#include <torch/csrc/fx/fx_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace fx {

struct ToRestore {
  PyObject* m_self;
  PyMethodDef* m_ml;
#if PY_VERSION_HEX >= 0x03080000
  vectorcallfunc vectorcall;
#endif
  PyObject* original_fn; // The original method we are trying to patch
  PyObject* patch_fn; // The function we're patching in place of original_fn
};

class DecRefGuard {
 public:
  DecRefGuard(PyObject* obj) : obj(obj) {}
  ~DecRefGuard() {
    Py_DECREF(obj);
  }

 private:
  PyObject* obj;
};

PyObject* replacement_method(PyObject* self, PyObject* args, PyObject* kwargs) {
  DecRefGuard self_guard(self);
  // restore the implementation immediately so that patch_fn lives for as little
  // as possible
  ToRestore* to_restore = (ToRestore*)PyBytes_AsString(self);
  PyCFunctionObject* patch_method_c =
      ((PyCFunctionObject*)to_restore->original_fn);
  patch_method_c->m_self = to_restore->m_self;
  patch_method_c->m_ml = to_restore->m_ml;
#if PY_VERSION_HEX >= 0x03080000
  patch_method_c->vectorcall = to_restore->vectorcall;
#endif

  if (kwargs) {
    Py_INCREF(kwargs);
  } else {
    kwargs = PyDict_New();
  }
  DecRefGuard kwargs_guard(kwargs);

  PyObject* result = nullptr;
  // Creates a tuple of 3 python objects
  PyObject* args_ =
      Py_BuildValue("(OOO)", to_restore->original_fn, args, kwargs);
  if (!args_) {
    return nullptr;
  }
  DecRefGuard args_guard(args_);
  // Calls the patched function with arguments of (original function, args,
  // kwargs)
  result = PyEval_CallObject(to_restore->patch_fn, args_);
  return result;
}
// The general idea is that we're patching a PyCFunctionObject, which has a
// couple relevant parts: m_ml: A PyMethodDef (the actual function to call)
// m_self: The self arg.
// vectorcall: An alternate calling convention (Python 3.8+)
// Usually we call obj.m_ml(obj.m_self, args, kwargs). However, we want to patch
// m_ml with ReplacementMethod (which calls our user-provided `patch_fn`). Thus,
// we also replace `m_self` with `ToRestore`, which contains all the information
// needed to restore the original function.
//
// `patch_function` parses the necessary information from the original
// PyCFunction and then patches it. When that function is called, it calls
// `replacement_method`, which then restores back the original `m_ml` and
// `m_self` values, as well as calling the user-defined `patch_fn`.

static PyObject* patch_function(PyObject* self, PyObject* args) {
  static PyMethodDef ReplacementMethod = {
      "replace",
      (PyCFunction)(void (*)())replacement_method,
      METH_VARARGS | METH_KEYWORDS,
      "Replaced method implementation."};

  ToRestore to_restore = {};
  if (!PyArg_ParseTuple(
          args, "OO", &to_restore.original_fn, &to_restore.patch_fn)) {
    return nullptr;
  }
  if (!PyCFunction_Check(to_restore.original_fn)) {
    std::stringstream err;
    err << "Patched object ";
    PyObject* obj_repr = PyObject_Repr(to_restore.original_fn);
    if (PyUnicode_Check(obj_repr)) {
      err << PyUnicode_AS_DATA(obj_repr) << " ";
    }
    err << " is not a CFunction. Please report a bug to PyTorch!";
    PyErr_SetString(PyExc_RuntimeError, err.str().c_str());
    return nullptr;
  }
  DecRefGuard patch_fn_guard(to_restore.patch_fn);
  Py_INCREF(to_restore.patch_fn);
  DecRefGuard patched_method_guard(to_restore.original_fn);
  Py_INCREF(to_restore.original_fn);
  PyCFunctionObject* patch_method_c =
      ((PyCFunctionObject*)to_restore.original_fn);

  to_restore.m_self = patch_method_c->m_self;
  to_restore.m_ml = patch_method_c->m_ml;
#if PY_VERSION_HEX >= 0x03080000
  to_restore.vectorcall = patch_method_c->vectorcall;
#endif

  patch_method_c->m_self =
      PyBytes_FromStringAndSize((const char*)&to_restore, sizeof(ToRestore));
  patch_method_c->m_ml = &ReplacementMethod;
#if PY_VERSION_HEX >= 0x03080000
  patch_method_c->vectorcall = nullptr;
#endif
  return Py_None;
}

bool isPythonTensor(at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(
      c10::DispatchKey::PythonKey);
}
PythonTensorImpl* getPythonImpl(at::Tensor tensor) {
  return static_cast<PythonTensorImpl*>(tensor.unsafeGetTensorImpl());
}

at::Tensor addKey(const py::object& tensor) {
  return at::detail::make_tensor<PythonTensorImpl>(tensor);
}

py::object removeKey(at::Tensor tensor) {
  assert(isPythonTensor(tensor));
  return getPythonImpl(tensor)->value_;
}

void initFx(PyObject* module) {
  static std::array<PyMethodDef, 2> PatchMethods = {{
      {"patch_function", patch_function, METH_VARARGS, "Save"},
      {nullptr},
  }};

  static struct PyModuleDef path = {
      PyModuleDef_HEAD_INIT,
      "patch", /* name of module */
      "", /* module documentation, may be NULL */
      -1, /* size of per-interpreter state of the module, or -1 if the module
            keeps state in global variables. */
      PatchMethods.data()};
  PyObject* patch = PyModule_Create(&path);
  if (!patch) {
    throw python_error();
  }
  if (PyModule_AddObject(module, "_fx", patch) != 0) {
    throw python_error();
  }

  auto m = py::handle(module).cast<py::module>();
  auto key = m.def_submodule("key");
  key.def("addKey", &addKey, py::return_value_policy::copy);
  key.def("removeKey", &removeKey);
}



PyObject* pyIdentity(py::object x) {
  return x.ptr();
}
template<class T>
py::tuple vectorToPyTuple(const std::vector<T> &data, std::function<PyObject*(T)> converter) {
	PyObject* tuple = PyTuple_New( data.size() );
	if (!tuple) throw std::runtime_error("Unable to allocate memory for Python tuple");
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *num = converter(data[i]);
		if (!num) {
			Py_DECREF(tuple);
			throw std::runtime_error("Unable to allocate memory for Python tuple");
		}
		PyTuple_SET_ITEM(tuple, i, num);
	}
        return py::cast<py::tuple>(tuple);
}
void pythonFallBack(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "python fallback" << std::endl;
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();

  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);

  py::gil_scoped_acquire g;
  std::vector<py::object> pyArgs;
  std::vector<py::object> pyTensorArgs;
  std::vector<torch::jit::IValue> unwrappedArgs;
  for (int idx = 0; idx < arguments.size(); idx++) {
    const auto ivalue = arguments[idx];
    if (ivalue.isTensor() && isPythonTensor(ivalue.toTensor())) {
      auto pyTensor = getPythonImpl(ivalue.toTensor());
      pyArgs.push_back(pyTensor->value_);
      pyTensorArgs.push_back(pyTensor->value_);
      unwrappedArgs.push_back(getValueFromPyTensor(pyTensor->value_));
    } else {
      pyArgs.push_back(torch::jit::toPyObject(ivalue));
      unwrappedArgs.push_back(ivalue);
    }
  }

  py::object torch_function = PyObject_FastGetAttrString(pyTensorArgs[0].ptr(), "__torch_function__");
  for (auto v : unwrappedArgs) {
    torch::jit::push(stack, v);
  }
  op.callBoxed(stack);
  auto realOut = torch::jit::pop(stack);

  py::tuple py_types = py::cast<py::tuple>(vectorToPyTuple<py::object>(pyArgs, [](py::object x) -> PyObject* { return PyObject_Type(x.ptr()); }));

  py::dict kwargs;
  kwargs["val"] = torch::jit::toPyObject(realOut);

  std::string func_name = op.operator_name().name;
  std::string delimiter = "aten::";
  func_name = func_name.substr(func_name.find(delimiter) + delimiter.size());
  py::object torch_api_function =
      PyObject_FastGetAttrString(THPVariableClass, (char*)func_name.c_str());
  if (torch_api_function == nullptr) {
    torch_api_function = py::str(op.operator_name().name);
  }
  auto pyTupleArgs = vectorToPyTuple<py::object>(pyArgs, pyIdentity);

  auto out = PyObject_CallFunctionObjArgs(torch_function.ptr(), torch_api_function.ptr(), py_types.ptr(), pyTupleArgs.ptr(), kwargs.ptr(), 0);
  if (out == nullptr) {
    throw std::runtime_error("call failed");
  }
  torch::jit::drop(stack, num_arguments);

  auto ret_type = op.schema().returns()[0].type();
  if (ret_type->kind() == c10::TensorType::Kind) {
    torch::jit::push(stack, addKey(py::cast<py::object>(out)));
  } else {
    auto ivalue_out = torch::jit::toIValue(out, ret_type);
    torch::jit::push(stack, ivalue_out);
  }
  return;
}
TORCH_LIBRARY_IMPL(_, PythonKey, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallBack>());
}
} // namespace fx
} // namespace torch
