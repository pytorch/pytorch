#if defined(USE_ROCM)

#include <torch/csrc/inductor/static_launcher/flydsl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_numbers.h>

namespace {

constexpr int kNumPackedArgs = 7;
constexpr int kNumTensorArgs = 3;
constexpr int kStreamArg = 6;

using PackedCifaceFn = void (*)(void*);

struct FlyDSLMMFp16Bf16CWrapperObject {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  PackedCifaceFn func;
  PyObject* owner;
  uint64_t argStorage[kNumPackedArgs];
  void* packedArgs[kNumPackedArgs];
};

uint64_t tensorDataPtr(PyObject* obj) {
  if (C10_LIKELY(THPVariable_CheckExact(obj))) {
    const auto& tensor = THPVariable_Unpack(obj);
    TORCH_CHECK(
        tensor.defined(),
        "_FlyDSLMMFp16Bf16CWrapper: received an undefined tensor");
    return static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(tensor.data_ptr()));
  }
  TORCH_CHECK(
      false,
      "_FlyDSLMMFp16Bf16CWrapper: expected an exact Tensor or Parameter");
  return 0;
}

PyObject* flydsl_c_wrapper_vectorcall(
    PyObject* callable,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      kwnames == nullptr,
      "_FlyDSLMMFp16Bf16CWrapper: keyword arguments are unsupported");
  const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  TORCH_CHECK(
      nargs == 4,
      "_FlyDSLMMFp16Bf16CWrapper: expected 4 arguments, got ",
      nargs);

  auto* self = reinterpret_cast<FlyDSLMMFp16Bf16CWrapperObject*>(callable);
  for (int i = 0; i < kNumTensorArgs; ++i) {
    self->argStorage[i] = tensorDataPtr(args[i]);
  }
  self->argStorage[kStreamArg] = THPUtils_unpackUInt64(args[kNumTensorArgs]);
  self->func(static_cast<void*>(self->packedArgs));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* FlyDSLMMFp16Bf16CWrapper_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  HANDLE_TH_ERRORS
  auto* self =
      reinterpret_cast<FlyDSLMMFp16Bf16CWrapperObject*>(
          type->tp_alloc(type, 0));
  if (!self) {
    return nullptr;
  }
  self->owner = nullptr;

  unsigned long long funcPtr = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  PyObject* owner = nullptr;
  if (!PyArg_ParseTuple(args, "KiiiO", &funcPtr, &m, &n, &k, &owner)) {
    Py_DECREF(self);
    return nullptr;
  }
  if (funcPtr == 0) {
    Py_DECREF(self);
    PyErr_SetString(
        PyExc_ValueError,
        "_FlyDSLMMFp16Bf16CWrapper: function pointer must be non-zero");
    return nullptr;
  }

  self->func = reinterpret_cast<PackedCifaceFn>(funcPtr); // NOLINT
  self->owner = owner;
  Py_INCREF(owner);
  std::memset(self->argStorage, 0, sizeof(self->argStorage));
  for (int i = 0; i < kNumPackedArgs; ++i) {
    self->packedArgs[i] = &self->argStorage[i];
  }

  const int32_t m32 = static_cast<int32_t>(m);
  const int32_t n32 = static_cast<int32_t>(n);
  const int32_t k32 = static_cast<int32_t>(k);
  std::memcpy(&self->argStorage[3], &m32, sizeof(m32));
  std::memcpy(&self->argStorage[4], &n32, sizeof(n32));
  std::memcpy(&self->argStorage[5], &k32, sizeof(k32));
  self->vectorcall = flydsl_c_wrapper_vectorcall;

  return reinterpret_cast<PyObject*>(self);
  END_HANDLE_TH_ERRORS
}

void FlyDSLMMFp16Bf16CWrapper_dealloc(PyObject* obj) {
  auto* self = reinterpret_cast<FlyDSLMMFp16Bf16CWrapperObject*>(obj);
  Py_XDECREF(self->owner);
  Py_TYPE(obj)->tp_free(obj);
}

PyTypeObject FlyDSLMMFp16Bf16CWrapperType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._FlyDSLMMFp16Bf16CWrapper", // tp_name
    sizeof(FlyDSLMMFp16Bf16CWrapperObject), // tp_basicsize
    0, // tp_itemsize
    FlyDSLMMFp16Bf16CWrapper_dealloc, // tp_dealloc
    offsetof(
        FlyDSLMMFp16Bf16CWrapperObject, vectorcall), // tp_vectorcall_offset
    nullptr, // tp_getattr
    nullptr, // tp_setattr
    nullptr, // tp_reserved
    nullptr, // tp_repr
    nullptr, // tp_as_number
    nullptr, // tp_as_sequence
    nullptr, // tp_as_mapping
    nullptr, // tp_hash
    PyVectorcall_Call, // tp_call
    nullptr, // tp_str
    nullptr, // tp_getattro
    nullptr, // tp_setattro
    nullptr, // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    "Native packed C wrapper for FlyDSL FP16/BF16 MM kernels", // tp_doc
    nullptr, // tp_traverse
    nullptr, // tp_clear
    nullptr, // tp_richcompare
    0, // tp_weaklistoffset
    nullptr, // tp_iter
    nullptr, // tp_iternext
    nullptr, // tp_methods
    nullptr, // tp_members
    nullptr, // tp_getset
    nullptr, // tp_base
    nullptr, // tp_dict
    nullptr, // tp_descr_get
    nullptr, // tp_descr_set
    0, // tp_dictoffset
    nullptr, // tp_init
    nullptr, // tp_alloc
    FlyDSLMMFp16Bf16CWrapper_new, // tp_new
};

} // namespace

bool FlyDSLMMFp16Bf16CWrapper_init(PyObject* module) {
  return PyModule_AddType(module, &FlyDSLMMFp16Bf16CWrapperType) == 0;
}

#endif
