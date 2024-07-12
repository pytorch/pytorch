#include <pybind11/pybind11.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/cuda/GdsFile.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include <ATen/cuda/CUDAGdsFile.h>
#include <structmember.h>

PyObject* THCPGdsFileClass = nullptr;

static PyObject* THCPGdsFile_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  // NOLINTNEXTLINE(*-c-arrays*)
  constexpr const char* kwlist[] = {"filename", "mode", nullptr};
  const char* filename = nullptr;
  const char* mode = nullptr;

  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "ss",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &filename,
          &mode)) {
    return nullptr;
  }

  // TODO: Need error checking for filename and mode?

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THCPGdsFile* self = (THCPGdsFile*)ptr.get();

  new (&self->gds_file) at::cuda::GDSFile(filename, mode);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPGdsFile_dealloc(THCPGdsFile* self) {
  self->gds_file.~GDSFile();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THCPGdsFile_load_tensor(PyObject* _self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto self = (THCPGdsFile*)_self;
  PyObject* t_ = PyTuple_GetItem(args, 0);
  PyObject* offset_ = PyTuple_GetItem(args, 1);

  TORCH_CHECK(THPVariable_Check(t_));
  TORCH_CHECK(THPUtils_checkLong(offset_));
  auto& t = THPVariable_Unpack(t_);
  int64_t offset = THPUtils_unpackLong(offset_);
  self->gds_file.load_tensor(t, offset);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPGdsFile_save_tensor(PyObject* _self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto self = (THCPGdsFile*)_self;
  PyObject* t_ = PyTuple_GetItem(args, 0);
  PyObject* offset_ = PyTuple_GetItem(args, 1);

  TORCH_CHECK(THPVariable_Check(t_));
  TORCH_CHECK(THPUtils_checkLong(offset_));
  auto& t = THPVariable_Unpack(t_);
  int64_t offset = THPUtils_unpackLong(offset_);
  self->gds_file.save_tensor(t, offset);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPGdsFile_load_storage(PyObject* _self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto self = (THCPGdsFile*)_self;
  PyObject* s_ = PyTuple_GetItem(args, 0);
  PyObject* offset_ = PyTuple_GetItem(args, 1);

  TORCH_CHECK(THPStorage_Check(s_));
  auto& s = THPStorage_Unpack(s_);
  int64_t offset = THPUtils_unpackLong(offset_);
  self->gds_file.load_storage(s, offset);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPGdsFile_save_storage(PyObject* _self, PyObject* args) {
  HANDLE_TH_ERRORS
  auto self = (THCPGdsFile*)_self;
  PyObject* s_ = PyTuple_GetItem(args, 0);
  PyObject* offset_ = PyTuple_GetItem(args, 1);

  TORCH_CHECK(THPStorage_Check(s_));
  auto& s = THPStorage_Unpack(s_);
  int64_t offset = THPUtils_unpackLong(offset_);
  self->gds_file.save_storage(s, offset);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static struct PyGetSetDef THCPGdsFile_properties[] = {
    // FIXME: add properties (perhaps getter for filename)
    // {"filename", (getter)THCPGdsFile_get_filename, nullptr, nullptr,
    // nullptr},
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static PyMethodDef THCPGdsFile_methods[] = {
    {(char*)"load_tensor", THCPGdsFile_load_tensor, METH_VARARGS, nullptr},
    {(char*)"save_tensor", THCPGdsFile_save_tensor, METH_VARARGS, nullptr},
    {(char*)"load_storage", THCPGdsFile_load_storage, METH_VARARGS, nullptr},
    {(char*)"save_storage", THCPGdsFile_save_storage, METH_VARARGS, nullptr},
    // {(char*)"register_buffer",
    //  THCPGdsFile_register_buffer,
    //  METH_VARARGS,
    //  nullptr},
    // {(char*)"deregister_buffer",
    //  THCPGdsFile_deregister_buffer,
    //  METH_VARARGS,
    //  nullptr},
    {nullptr}};

PyTypeObject THCPGdsFileType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._CudaGdsFileBase", /* tp_name */
    sizeof(THCPGdsFile), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THCPGdsFile_dealloc, /* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THCPGdsFile_methods, /* tp_methods */
    nullptr, /* tp_members */
    THCPGdsFile_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THCPGdsFile_pynew, /* tp_new */
};

PyObject* THCPModule_gds_register_buffer(PyObject* _self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* t_ = PyTuple_GetItem(args, 0);

  TORCH_CHECK(THPVariable_Check(t_) || THPStorage_Check(t_));
  if (THPVariable_Check(t_)) {
    auto& t = THPVariable_Unpack(t_);
    at::cuda::gds_register_buffer(t);
  } else {
    auto& t = THPStorage_Unpack(t_);
    at::cuda::gds_register_buffer(t);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_gds_deregister_buffer(PyObject* _self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* t_ = PyTuple_GetItem(args, 0);

  TORCH_CHECK(THPVariable_Check(t_) || THPStorage_Check(t_));
  if (THPVariable_Check(t_)) {
    auto& t = THPVariable_Unpack(t_);
    at::cuda::gds_deregister_buffer(t);
  } else {
    auto& t = THPStorage_Unpack(t_);
    at::cuda::gds_deregister_buffer(t);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

void THCPGdsFile_init(PyObject* module) {
  THCPGdsFileClass = (PyObject*)&THCPGdsFileType;
  if (PyType_Ready(&THCPGdsFileType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPGdsFileType);
  if (PyModule_AddObject(
          module, "_CudaGdsFileBase", (PyObject*)&THCPGdsFileType) < 0) {
    throw python_error();
  }
}
