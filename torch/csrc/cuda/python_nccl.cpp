#include <torch/csrc/cuda/python_nccl.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/cuda/THCP.h>
#include <torch/csrc/cuda/nccl.h>
#include <ATen/core/functional.h>

#include <c10/cuda/CUDAGuard.h>

#include <nccl.h>

#include <sstream>
#include <unordered_map>

using namespace at;
using namespace torch;
using namespace torch::cuda::nccl;
using namespace torch::cuda::nccl::detail;

static const char* COMM_CAPSULE_NAME = "torch.cuda.nccl.Communicator";

PyObject* THCPModule_nccl_version(PyObject* self, PyObject* args) {
  return PyInt_FromLong(version());
}

PyObject* THCPModule_nccl_unique_id(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));
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
  /*
   * TODO(T30279827) Temporarily disable calling ncclCommDestroy
   * Calling ncclCommDestroy while program exiting is undefined
   * according to Nvidia, and lead to segfault in NCCL 2
   * (whether it is called before or after the CUDA runtime destructor).
   * Temporarily disable it in destructor to avoid segfault.
   * Following up with Nvidia for long term solution.
   */
  return;

  HANDLE_TH_ERRORS
  ncclComm_t comm = unpack_nccl_comm(capsule);
  {
    pybind11::gil_scoped_release no_gil;
    ncclCommDestroy(comm);
  }
  END_HANDLE_TH_ERRORS_RET()
}

static std::vector<c10::optional<at::cuda::CUDAStream>> unpack_streams(PyObject* obj, size_t size) {
  if (obj == Py_None) {
    return std::vector<c10::optional<at::cuda::CUDAStream>>(size, c10::nullopt);
  }
  auto streams = THPUtils_PySequence_to_CUDAStreamList(obj);
  if (streams.size() != size) {
    throw std::runtime_error(
        "number of streams is not equal to number of inputs");
  }
  return streams;
}

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
    for (int64_t i = 0; i < size; i++) {
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
  int nranks;
  const char* id;
  Py_ssize_t id_len;
  int rank;

  if (!PyArg_ParseTuple(
          args, "is#i:nccl_init_rank", &nranks, &id, &id_len, &rank)) {
    return nullptr;
  }
  THPUtils_assert(
      id_len == NCCL_UNIQUE_ID_BYTES,
      "invalid unqiue_id (expected %d bytes, got %zd)",
      NCCL_UNIQUE_ID_BYTES,
      id_len);

  ncclUniqueId commId;
  memcpy(&commId, id, NCCL_UNIQUE_ID_BYTES);
  ncclComm_t comm;
  {
    pybind11::gil_scoped_release no_gil;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, commId, rank));
  }
  return PyCapsule_New(comm, COMM_CAPSULE_NAME, &destroy_nccl_comm);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_reduce(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams, *_comms;
  int root, op;

  if (!PyArg_ParseTuple(
          args,
          "OOiiOO",
          &_inputs,
          &_outputs,
          &root,
          &op,
          &_streams,
          &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_reduce",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs, int root,"
        " int op, sequence[torch.cuda.Stream or None]");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  std::vector<c10::optional<at::cuda::CUDAStream>> streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    torch::cuda::nccl::reduce(inputs, outputs, root, op, streams, user_comms);
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_all_reduce(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams, *_comms;
  int op;

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
    check_inputs(inputs, outputs, 1, 1);
    size_t len = inputs.size();

    ncclDataType_t data_type = get_data_type(inputs[0]);

    int64_t count = inputs[0].numel();
    auto comms = user_comms.empty() ? get_communicators(inputs)
                                    : ArrayRef<ncclComm_t>(user_comms);
    AutoNcclGroup nccl_group_guard;
    at::cuda::OptionalCUDAGuard device_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      device_guard.set_index(device);
      auto stream = !streams[i]
          ? at::cuda::getCurrentCUDAStream(device).stream()
          : streams[i]->stream();
      NCCL_CHECK(ncclAllReduce(
          inputs[i].data_ptr(),
          outputs[i].data_ptr(),
          count,
          data_type,
          (ncclRedOp_t)op,
          comms[i],
          stream));
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_broadcast(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_streams, *_comms;
  int root;

  if (!PyArg_ParseTuple(args, "OiOO", &_inputs, &root, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_broadcast",
        1,
        "(sequence[Tensor] inputs, int root)");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  THPUtils_assert(root >= 0 && (size_t)root < inputs.size(), "invalid root");
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
  PyObject *_inputs, *_outputs, *_streams, *_comms;

  if (!PyArg_ParseTuple(
          args, "OOOO", &_inputs, &_outputs, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_all_gather",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    size_t len = inputs.size();
    check_inputs(inputs, outputs, len, 1);

    ncclDataType_t data_type = get_data_type(inputs[0]);

    int64_t count = inputs[0].numel();
    auto comms = user_comms.empty() ? get_communicators(inputs)
                                    : ArrayRef<ncclComm_t>(user_comms);
    AutoNcclGroup nccl_group_guard;
    at::cuda::OptionalCUDAGuard device_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      device_guard.set_index(device);
      auto stream = !streams[i]
          ? at::cuda::getCurrentCUDAStream(device).stream()
          : streams[i]->stream();
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
      NCCL_CHECK(ncclAllGather(
          inputs[i].data_ptr(),
          outputs[i].data_ptr(),
          count,
          data_type,
          comms[i],
          stream));
#else
      NCCL_CHECK(ncclAllGather(
          inputs[i].data_ptr(),
          count,
          data_type,
          outputs[i].data_ptr(),
          comms[i],
          stream));
#endif
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_nccl_reduce_scatter(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams, *_comms;
  int op;

  if (!PyArg_ParseTuple(
          args, "OOiOO", &_inputs, &_outputs, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "nccl_reduce_scatter",
        1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs, int op");
    return nullptr;
  }

  std::vector<at::Tensor> inputs = extract_tensors(_inputs);
  std::vector<at::Tensor> outputs = extract_tensors(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  {
    pybind11::gil_scoped_release no_gil;
    size_t len = inputs.size();
    check_inputs(inputs, outputs, 1, len);

    ncclDataType_t data_type = get_data_type(inputs[0]);

    int64_t count = inputs[0].numel() / len;
    auto comms = user_comms.empty() ? get_communicators(inputs)
                                    : ArrayRef<ncclComm_t>(user_comms);
    AutoNcclGroup nccl_group_guard;
    at::cuda::OptionalCUDAGuard device_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      device_guard.set_index(device);
      auto stream = !streams[i]
          ? at::cuda::getCurrentCUDAStream(device).stream()
          : streams[i]->stream();
      NCCL_CHECK(ncclReduceScatter(
          inputs[i].data_ptr(),
          outputs[i].data_ptr(),
          count,
          data_type,
          (ncclRedOp_t)op,
          comms[i],
          stream));
    }
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static std::vector<at::Tensor> extract_tensors(PyObject* obj) {
  auto seq = THPObjectPtr(PySequence_Fast(obj, "expected a sequence"));
  if (!seq)
    throw python_error();

  std::vector<at::Tensor> list;
  Py_ssize_t length = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < length; i++) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq.get(), i);
    if (!THPVariable_Check(item)) {
      throw TypeError(
          "expected Tensor at %d (got %s)", (int)i, Py_TYPE(item)->tp_name);
    }
    auto var = (THPVariable*)item;
    list.emplace_back(var->cdata);
  }
  return list;
}
