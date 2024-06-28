#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/cuda/Module.h>
#include <torch/csrc/cuda/Stream.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime_api.h>
#include <structmember.h>

PyObject* THCPStreamClass = nullptr;

static PyObject* THCPStream_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS

  const auto current_device = c10::cuda::current_device();

  int priority = 0;
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;
  uint64_t stream_ptr = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "priority",
      "stream_id",
      "device_index",
      "device_type",
      "stream_ptr",
      nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|iLLLK",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &priority,
          &stream_id,
          &device_index,
          &device_type,
          &stream_ptr)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  if (stream_ptr) {
    TORCH_CHECK(
        priority == 0, "Priority was explicitly set for a external stream")
  }
  at::cuda::CUDAStream stream = (stream_id || device_index || device_type)
      ? at::cuda::CUDAStream::unpack3(
            stream_id,
            static_cast<c10::DeviceIndex>(device_index),
            static_cast<c10::DeviceType>(device_type))
      : stream_ptr ? at::cuda::getStreamFromExternal(
                         // NOLINTNEXTLINE(performance-no-int-to-ptr)
                         reinterpret_cast<cudaStream_t>(stream_ptr),
                         current_device)
                   : at::cuda::getStreamFromPool(priority);

  THCPStream* self = (THCPStream*)ptr.get();
  self->stream_id = static_cast<int64_t>(stream.id());
  self->device_index = static_cast<int64_t>(stream.device_index());
  self->device_type = static_cast<int64_t>(stream.device_type());
  new (&self->cuda_stream) at::cuda::CUDAStream(stream);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPStream_dealloc(THCPStream* self) {
  self->cuda_stream.~CUDAStream();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THCPStream_get_cuda_stream(THCPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->cuda_stream.stream());
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPStream_get_priority(THCPStream* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(self->cuda_stream.priority());
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPStream_priority_range(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto [least_priority, greatest_priority] =
      at::cuda::CUDAStream::priority_range();
  return Py_BuildValue("(ii)", least_priority, greatest_priority);
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPStream_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THCPStream*)_self;
  return PyBool_FromLong(self->cuda_stream.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPStream_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    auto self = (THCPStream*)_self;
    self->cuda_stream.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPStream_eq(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THCPStream*)_self;
  auto other = (THCPStream*)_other;
  return PyBool_FromLong(self->cuda_stream == other->cuda_stream);
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMemberDef THCPStream_members[] = {{nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyGetSetDef THCPStream_properties[] = {
    {"cuda_stream",
     (getter)THCPStream_get_cuda_stream,
     nullptr,
     nullptr,
     nullptr},
    {"priority", (getter)THCPStream_get_priority, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static PyMethodDef THCPStream_methods[] = {
    {"query", THCPStream_query, METH_NOARGS, nullptr},
    {"synchronize", THCPStream_synchronize, METH_NOARGS, nullptr},
    {"priority_range",
     THCPStream_priority_range,
     METH_STATIC | METH_NOARGS,
     nullptr},
    {"__eq__", THCPStream_eq, METH_O, nullptr},
    {nullptr}};

PyTypeObject THCPStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._CudaStreamBase", /* tp_name */
    sizeof(THCPStream), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THCPStream_dealloc, /* tp_dealloc */
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
    THCPStream_methods, /* tp_methods */
    THCPStream_members, /* tp_members */
    THCPStream_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THCPStream_pynew, /* tp_new */
};

void THCPStream_init(PyObject* module) {
  Py_INCREF(THPStreamClass);
  THCPStreamType.tp_base = THPStreamClass;
  THCPStreamClass = (PyObject*)&THCPStreamType;
  if (PyType_Ready(&THCPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPStreamType);
  if (PyModule_AddObject(
          module, "_CudaStreamBase", (PyObject*)&THCPStreamType) < 0) {
    throw python_error();
  }
}
