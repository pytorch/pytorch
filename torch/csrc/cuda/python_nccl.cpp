#include <torch/csrc/cuda/python_nccl.h>

#include <ATen/core/functional.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/utils/pybind.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>

using namespace at;
using namespace torch;
using namespace torch::cuda::nccl;
using namespace torch::cuda::nccl::detail;

static const char* COMM_CAPSULE_NAME = "torch.cuda.nccl.Communicator";

PyObject* THCPModule_nccl_version(PyObject* self, PyObject* args) {
  return PyLong_FromUnsignedLongLong(version());
}

PyObject* THCPModule_nccl_version_suffix(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  return PyBytes_FromString(version_suffix());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_unique_id(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  ncclUniqueId id;
  get_unique_id(id);
  return PyBytes_FromStringAndSize((char*)&id, NCCL_UNIQUE_ID_BYTES);
  END_HANDLE_TH_ERRORS
}

static ncclComm_t unpack_nccl_comm(PyObject* capsule) {
  ncclComm_t comm =
      (ncclComm_t)PyCapsule_GetPointer(capsule, COMM_CAPSULE_NAME);
  if (!comm)
    throw python_error();
  return comm;
}

static void destroy_nccl_comm(PyObject* capsule) {
  HANDLE_TH_ERRORS
  ncclComm_t comm = unpack_nccl_comm(capsule);
  {
    pybind11::gil_scoped_release no_gil;
    comm_destroy(comm);
  }
  END_HANDLE_TH_ERRORS_RET()
}

static std::vector<std::optional<at::cuda::CUDAStream>> unpack_streams(
    PyObject* obj,
    size_t size) {
  if (obj == Py_None) {
    return std::vector<std::optional<at::cuda::CUDAStream>>(size, std::nullopt);
  }
  auto streams = THPUtils_PySequence_to_CUDAStreamList(obj);
  if (streams.size() != size) {
    throw std::runtime_error(
        "number of streams is not equal to number of inputs");
  }
  return streams;
}

static at::Tensor extract_tensor(PyObject* obj);
static std::vector<at::Tensor> extract_tensors(PyObject* obj);

static std::vector<ncclComm_t> unpack_comms(PyObject* obj, size_t size) {
  if (obj == Py_None) {
    return std::vector<ncclComm_t>();
  }
  std::vector<ncclComm_t> comms;
  if (PyCapsule_CheckExact(obj)) {
    comms = {unpack_nccl_comm(obj)};
  } else {
    auto seq = THPObjectPtr(PySequence_Fast(obj, "comm is not a sequence"));
    if (!seq)
      throw python_error();
    auto size = PySequence_Fast_GET_SIZE(seq.get());
    comms = std::vector<ncclComm_t>(size);
    for (const auto i : c10::irange(size)) {
      comms[i] = unpack_nccl_comm(PySequence_Fast_GET_ITEM(seq.get(), i));
    }
  }
  if (comms.size() != size) {
    throw std::runtime_error(
        "number of communicators is not equal to number of inputs");
  }
  return comms;
}

PyObject* THCPModule_nccl_init_rank(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  int nranks = 0;
  const char* id = nullptr;
  Py_ssize_t id_len = 0;
  int rank = 0;

  if (!PyArg_ParseTuple(
          args, "is#i:nccl_init_rank", &nranks, &id, &id_len, &rank)) {
    return nullptr;
  }
  TORCH_CHECK(
      id_len == NCCL_UNIQUE_ID_BYTES,
      "invalid unique_id (expected ",
      NCCL_UNIQUE_ID_BYTES,
      " bytes, got ",
      id_len,
      ")");

  ncclUniqueId commId;
  memcpy(&commId, id, NCCL_UNIQUE_ID_BYTES);
  ncclComm_t comm = nullptr;
  {
    pybind11::gil_scoped_release no_gil;
    comm = comm_init_rank(nranks, commId, rank);
  }
  return PyCapsule_New(comm, COMM_CAPSULE_NAME, &destroy_nccl_comm);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_reduce(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_output = nullptr, *_streams = nullptr,
           *_comms = nullptr;
  int root = 0, op = 0;

  if (!PyArg_ParseTuple(
          args, "OOiiOO", &_inputs, &_output, &root, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_reduce",
        1,
        "(sequence[Tensor] inputs, Tensor output, int root,"
        " int op, sequence[torch.cuda.Stream or None]");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  auto output = extract_tensor(_output);
  std::vector<std::optional<at::cuda::CUDAStream>> streams =
      unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    torch::cuda::nccl::reduce(inputs, output, root, op, streams, user_comms);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_all_reduce(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_outputs = nullptr, *_streams = nullptr,
           *_comms = nullptr;
  int op = 0;

  if (!PyArg_ParseTuple(
          args, "OOiOO", &_inputs, &_outputs, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_all_reduce",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs, int op,"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    all_reduce(inputs, outputs, op, streams, user_comms);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_broadcast(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_streams = nullptr, *_comms = nullptr;
  int root = 0;

  if (!PyArg_ParseTuple(args, "OiOO", &_inputs, &root, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_broadcast",
        1,
        "(sequence[Tensor] inputs, int root"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  TORCH_CHECK(root >= 0 && (size_t)root < inputs.size(), "invalid root");
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    torch::cuda::nccl::broadcast(inputs, streams, user_comms);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_all_gather(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_outputs = nullptr, *_streams = nullptr,
           *_comms = nullptr;

  if (!PyArg_ParseTuple(
          args, "OOOO", &_inputs, &_outputs, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_all_gather",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    all_gather(inputs, outputs, streams, user_comms);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_reduce_scatter(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs = nullptr, *_outputs = nullptr, *_streams = nullptr,
           *_comms = nullptr;
  int op = 0;

  if (!PyArg_ParseTuple(
          args, "OOiOO", &_inputs, &_outputs, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_reduce_scatter",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs, int op"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    reduce_scatter(inputs, outputs, op, streams, user_comms);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static at::Tensor extract_tensor(PyObject* obj) {
  TORCH_CHECK_TYPE(
      THPVariable_Check(obj),
      "expected Tensor (got ",
      Py_TYPE(obj)->tp_name,
      ")");
  return THPVariable_Unpack(obj);
}

static std::vector<at::Tensor> extract_tensors(PyObject* obj) {
  auto seq = THPObjectPtr(PySequence_Fast(obj, "expected a sequence"));
  if (!seq)
    throw python_error();

  const Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  std::vector<at::Tensor> list;
  if (length >= 0) {
    list.reserve(length);
  }
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq.get(), i);
    TORCH_CHECK_TYPE(
        THPVariable_Check(item),
        "expected Tensor at ",
        i,
        " (got ",
        Py_TYPE(item)->tp_name,
        ")");
    list.emplace_back(THPVariable_Unpack(item));
  }
  return list;
}
