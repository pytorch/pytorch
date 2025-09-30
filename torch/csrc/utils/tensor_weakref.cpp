#include <ATen/core/TensorBody.h>
#include <Python.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/tensor_weakref.h>

typedef struct {
  PyObject_HEAD
  c10::weak_intrusive_ptr<c10::TensorImpl> weak;
  Py_hash_t hash;
} THPTensorImplWeakRef;

static PyObject* THPTensorImplWeakRef_call(
    THPTensorImplWeakRef* self,
    PyObject*,
    PyObject*) {
  auto strong = self->weak.lock();
  if (!strong) {
    Py_RETURN_NONE;
  }
  at::Tensor t(std::move(strong));
  return THPVariable_Wrap(t);
}

static Py_hash_t THPTensorImplWeakRef_hash(THPTensorImplWeakRef* self) {
  if (self->hash != -1) {
    return self->hash;
  }
  auto strong = self->weak.lock();
  if (!strong) {
    PyErr_SetString(PyExc_TypeError, "weak object has gone away");
    return -1;
  }
  at::Tensor t(strong);
  PyObject* py_tensor = THPVariable_Wrap(t);
  if (!py_tensor) {
    return -1;
  }
  self->hash = PyObject_Hash(py_tensor);
  Py_DECREF(py_tensor);
  return self->hash;
}

static PyObject* THPTensorImplWeakRef_richcmp(
    PyObject* a,
    PyObject* b,
    int op) {
  if (op != Py_EQ && op != Py_NE) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  auto* wa = reinterpret_cast<THPTensorImplWeakRef*>(a);
  auto* wb = reinterpret_cast<THPTensorImplWeakRef*>(b);
  auto aptr = wa->weak.lock();
  auto bptr = wb->weak.lock();
  if (!aptr || !bptr) {
    int res = (a == b);
    if (op == Py_NE)
      res = !res;
    if (res)
      Py_RETURN_TRUE;
    else
      Py_RETURN_FALSE;
  }
  at::Tensor t_a(aptr);
  at::Tensor t_b(bptr);
  PyObject* py_a = THPVariable_Wrap(t_a);
  PyObject* py_b = THPVariable_Wrap(t_b);
  if (!py_a || !py_b) {
    Py_XDECREF(py_a);
    Py_XDECREF(py_b);
    return nullptr;
  }
  PyObject* res = PyObject_RichCompare(py_a, py_b, op);
  Py_DECREF(py_a);
  Py_DECREF(py_b);
  return res;
}

static void THPTensorImplWeakRef_dealloc(THPTensorImplWeakRef* self) {
  self->weak.reset();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

static PyObject* THPTensorImplWeakRef_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  PyObject* py_tensor = nullptr;
  if (!PyArg_ParseTuple(args, "O", &py_tensor)) {
    return nullptr;
  }
  if (!THPVariable_Check(py_tensor)) {
    PyErr_SetString(PyExc_TypeError, "expected a Tensor");
    return nullptr;
  }
  const at::Tensor t = THPVariable_Unpack(py_tensor);
  c10::intrusive_ptr<c10::TensorImpl> strong = t.getIntrusivePtr();
  c10::weak_intrusive_ptr<c10::TensorImpl> weak(strong);
  strong.reset();
  auto* self = reinterpret_cast<THPTensorImplWeakRef*>(type->tp_alloc(type, 0));
  if (!self)
    return nullptr;
  try {
    new (&self->weak) c10::weak_intrusive_ptr<c10::TensorImpl>(std::move(weak));
    self->hash = -1;
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    type->tp_free(self);
    PyErr_SetString(PyExc_RuntimeError, "Failed to initialize weakref");
    return nullptr;
  }
}

static PyObject* THPTensorImplWeakRef_repr(PyObject* self_) {
  THPTensorImplWeakRef* self = reinterpret_cast<THPTensorImplWeakRef*>(self_);
  c10::intrusive_ptr<c10::TensorImpl> strong = self->weak.lock();
  if (strong) {
    return PyUnicode_FromFormat(
        "<weakref at %p; to 'Tensor' at %p>", self_, strong.get());
  } else {
    return PyUnicode_FromFormat("<weakref at %p; dead>", self_);
  }
}

static int THPTensorImplWeakRef_bool(PyObject* self_) {
  THPTensorImplWeakRef* self = reinterpret_cast<THPTensorImplWeakRef*>(self_);
  c10::intrusive_ptr<c10::TensorImpl> strong = self->weak.lock();
  return strong ? 1 : 0;
}

static PyType_Slot THPTensorImplWeakRef_slots[] = {
    {Py_tp_dealloc, (void*)THPTensorImplWeakRef_dealloc},
    {Py_tp_hash, (void*)THPTensorImplWeakRef_hash},
    {Py_tp_call, (void*)THPTensorImplWeakRef_call},
    {Py_tp_richcompare, (void*)THPTensorImplWeakRef_richcmp},
    {Py_tp_repr, (void*)THPTensorImplWeakRef_repr},
    {Py_nb_bool, (void*)THPTensorImplWeakRef_bool},
    {Py_tp_new, (void*)THPTensorImplWeakRef_new},
    {0, 0}};

static PyType_Spec THPTensorImplWeakRef_spec = {
    "torch._C.TensorImplWeakRef",
    sizeof(THPTensorImplWeakRef),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE,
    THPTensorImplWeakRef_slots};

bool TensorImplWeakRef_init(PyObject* module) {
  PyObject* type = PyType_FromSpec(&THPTensorImplWeakRef_spec);
  if (!type) {
    return false;
  }
  if (PyModule_AddObject(module, "TensorImplWeakRef", type) < 0) {
    Py_DECREF(type);
    return false;
  }
  return true;
}
