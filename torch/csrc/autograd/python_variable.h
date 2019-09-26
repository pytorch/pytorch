#pragma once

#include <torch/csrc/python_headers.h>
#include <memory>
#include <ATen/ATen.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/THP_export.h>

// Python object that backs torch.autograd.Variable
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPVariable {
    PyObject_HEAD
    // Payload
    torch::autograd::Variable cdata;
    // Hooks to be run on backwards pass (corresponds to Python attr
    // '_backwards_hooks', set by 'register_hook')
    PyObject* backward_hooks = nullptr;
};

THP_API PyObject *THPVariableClass;

bool THPVariable_initModule(PyObject *module);
THP_API PyObject * THPVariable_Wrap(torch::autograd::Variable var);

static inline bool THPVariable_CheckExact(PyObject *obj) {
  return Py_TYPE(obj) == (PyTypeObject*)THPVariableClass;
}

inline bool THPVariable_Check_Subclass(PyObject *obj){
  int is_subclass = PyObject_IsSubclass((PyObject *)Py_TYPE(obj), (PyObject *)THPVariableClass);
  if (is_subclass == -1) {
    return false;
  }
  return true;
}

inline bool THPVariable_Check(PyObject *obj)
{
  return THPVariableClass && PyObject_IsInstance(obj, THPVariableClass);
}

inline torch::autograd::Variable& THPVariable_Unpack(PyObject* obj) {
  auto var = (THPVariable*)obj;
  return var->cdata;
}

/*
 * Stripped down version of PyObject_GetAttrString,
 * avoids lookups for None, tuple, and List objects,
 * and doesn't create a PyErr since this code ignores it.
 *
 * This can be much faster then PyObject_GetAttrString where
 * exceptions are not used by caller.
 *
 * 'obj' is the object to search for attribute.
 *
 * 'name' is the attribute to search for.
 *
 * Returns attribute value on success, NULL on failure.
 */

static PyObject * PyObject_FastGetAttrString(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, name);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    /* Attribute referenced by (PyObject *)name */
    else if (tp->tp_getattro != NULL) {
        PyObject *w = PyUnicode_InternFromString(name);
        if (w == NULL) {
            return (PyObject *)NULL;
        }
        res = (*tp->tp_getattro)(obj, w);
        Py_DECREF(w);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    return res;
}

// Makes sure that we don't check for __torch_function__ on basic Python types
static bool _is_basic_python_type(PyTypeObject *tp)
{
    return (
        /* Basic number types */
        tp == &PyBool_Type ||

        tp == &PyLong_Type ||
        tp == &PyFloat_Type ||
        tp == &PyComplex_Type ||

        /* Basic sequence types */
        tp == &PyList_Type ||
        tp == &PyTuple_Type ||
        tp == &PyDict_Type ||
        tp == &PySet_Type ||
        tp == &PyFrozenSet_Type ||
        tp == &PyUnicode_Type ||
        tp == &PyBytes_Type ||
/*#if !defined(NPY_PY3K)
        tp == &PyString_Type ||
#endif  DISCUSS*/

        /* other builtins */
        tp == &PySlice_Type ||
        tp == Py_TYPE(Py_None) ||
        tp == Py_TYPE(Py_Ellipsis) ||
        tp == Py_TYPE(Py_NotImplemented) ||

        PyModule_Check(tp) ||
        /* TODO: ndarray, but we can't see PyArray_Type here */

        /* sentinel to swallow trailing || */
        false
    );
}

/*
 * Lookup a special method, following the python approach of looking up
 * on the type object, rather than on the instance itself.
 *
 * Assumes that the special method is a numpy-specific one, so does not look
 * at builtin types, nor does it look at a base ndarray.
 *
 * In future, could be made more like _Py_LookupSpecial
 */

static PyObject* PyTorch_LookupSpecial(PyObject *obj, char *name)
{
  if(PyObject_HasAttrString(obj, name) == 0){
    return NULL;
  }
  PyTypeObject *tp = Py_TYPE(obj);
  if (_is_basic_python_type(tp)) {
    return NULL;
  }
  return PyObject_FastGetAttrString((PyObject *)tp, name);
}

static PyObject* get_torch_function(PyObject* obj)
{
  return PyTorch_LookupSpecial(obj, "__torch_function__");
}
