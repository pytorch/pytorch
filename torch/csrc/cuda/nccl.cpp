#include "nccl.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/Types.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/cuda/THCP.h"

#include <nccl.h>
#include <sstream>
#include <unordered_map>

using namespace at;

static const char* COMM_CAPSULE_NAME = "torch.cuda.nccl.Communicator";

static inline void CHECK(ncclResult_t status) {
  if (status != ncclSuccess) {
    std::stringstream err;
    err << "NCCL Error " << status << ": " << ncclGetErrorString(status);
    throw std::runtime_error(err.str());
  }
}

struct NcclCommList {
  std::unique_ptr<ncclComm_t[]> comms;
  int ndevices;
  NcclCommList(const std::vector<int>& devices)
    : comms(new ncclComm_t[devices.size()]), ndevices(devices.size()) {
    CHECK(ncclCommInitAll(comms.get(), devices.size(), devices.data()));
  }
  NcclCommList(NcclCommList&& foo) = default;
  ~NcclCommList() {
    if (comms) {
      for (int i = 0; i < ndevices; i++) {
        int dummy_var;
        if (cudaGetDevice(&dummy_var) != cudaSuccess) {
          /* there are cases when this destructor is called after the
           CUDA driver is already unloaded from the process.
           In these cases, skip ncclCommDestroy */
          return;
        }
        ncclCommDestroy(comms[i]);
      }
    }
  }
  ArrayRef<ncclComm_t> ref() const {
    return ArrayRef<ncclComm_t>(comms.get(), ndevices);
  }
};

struct AutoNcclGroup {
  AutoNcclGroup() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    CHECK(ncclGroupStart());
#endif
  }
  ~AutoNcclGroup() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    CHECK(ncclGroupEnd());
#endif
  }
};

// accesses to this object have to be guarded by THC's CudaFreeMutex
std::unordered_map<std::string, NcclCommList> _communicators;

static ArrayRef<ncclComm_t> _get_communicators(TensorList inputs) {
  std::stringstream hash_stream;
  std::vector<int> devs;
  for (auto& input : inputs) {
    int dev = input.get_device();
    hash_stream << dev << ",";
    devs.push_back(dev);
  }
  std::string hash = hash_stream.str();
  auto it = _communicators.find(hash);
  if (it == _communicators.end()) {
    return _communicators.emplace_hint(it, hash, devs)->second.ref();
  } else {
    return it->second.ref();
  }
}

static void _check_inputs(TensorList inputs, TensorList outputs, int input_multiplier, int output_multiplier) {
  // len(inputs) == len(outputs)
  size_t len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  if (len != outputs.size()) {
    std::stringstream err;
    err << "inputs and outputs sequences have to be of the same length, but got input of length " << len << " and output of length " << outputs.size();
    throw std::runtime_error(err.str());
  }

  std::unordered_set<int> devices;
  devices.reserve(len);
  int64_t numel = inputs[0].numel();
  auto& type = inputs[0].type();

  for (size_t i = 0; i < len; i++) {
    auto input = inputs[i];
    auto output = outputs[i];

    if (!(input.type().is_cuda() && !input.type().is_sparse()
        && output.type().is_cuda()  && !output.type().is_sparse())) {
      throw std::runtime_error("input and output elements have to be cuda dense Tensors");
    }

    if (!(type == input.type() && type == output.type())) {
      throw std::runtime_error("all inputs and outputs must be of the same Tensor type");
    }

    if (!input.is_contiguous() || !output.is_contiguous()) {
      throw std::runtime_error("all inputs and outputs have to be contiguous");
    }

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.find(input_device) != devices.end()) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.insert(input_device);

    // inputs and outputs must be on same device respectively
    if (input_device != output.get_device()) {
      throw std::runtime_error("input and output must be on the same device");
    }

    // all inputs must be same size
    if (input.numel() != numel) {
      throw std::runtime_error("all inputs must have the same number of elements");
    }

    if (output.numel() * output_multiplier != numel * input_multiplier) {
      throw std::runtime_error("output must be of size input_size * size_multiplier");
    }
  }
}

static ncclDataType_t _get_data_type(const Type& type) {
  if (type.backend() != kCUDA) {
    throw std::runtime_error("Unconvertible NCCL type");
  }
  switch (type.scalarType()) {
  case at::kFloat   : return ncclFloat;
  case at::kHalf    : return ncclHalf;
  case at::kDouble  : return ncclDouble;
  case at::kLong    : return ncclInt64;
  case at::kInt     : return ncclInt;
  case at::kChar    : return ncclChar;
  case at::kByte    : return ncclChar;
  default: throw std::runtime_error("Unconvertible NCCL type");
  }
}

PyObject * THCPModule_nccl_version(PyObject *self, PyObject *args) {
#if defined(NCCL_MAJOR)
  return PyInt_FromLong(NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH);
#else
  return PyInt_FromLong(1000);  // assume NCCL 1.0
#endif
}

PyObject * THCPModule_nccl_unique_id(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  ncclUniqueId id;
  CHECK(ncclGetUniqueId(&id));
  return PyBytes_FromStringAndSize((char*)&id, NCCL_UNIQUE_ID_BYTES);
  END_HANDLE_TH_ERRORS
}

static ncclComm_t unpack_nccl_comm(PyObject* capsule) {
  ncclComm_t comm = (ncclComm_t)PyCapsule_GetPointer(capsule, COMM_CAPSULE_NAME);
  if (!comm) throw python_error();
  return comm;
}

static void destroy_nccl_comm(PyObject* capsule) {
  HANDLE_TH_ERRORS
  ncclComm_t comm = unpack_nccl_comm(capsule);
  with_no_gil([&]{
    ncclCommDestroy(comm);
  });
  END_HANDLE_TH_ERRORS_RET()
}

static std::vector<THCStream*> unpack_streams(PyObject* obj, size_t size) {
  if (obj == Py_None) {
    return std::vector<THCStream*>(size, nullptr);
  }
  auto streams = THPUtils_PySequence_to_THCStreamList(obj);
  if (streams.size() != size) {
    throw std::runtime_error("number of streams is not equal to number of inputs");
  }
  return streams;
}

static std::vector<ncclComm_t> unpack_comms(PyObject* obj, size_t size) {
  if (obj == Py_None) {
    return std::vector<ncclComm_t>();
  }
  std::vector<ncclComm_t> comms;
  if (PyCapsule_CheckExact(obj)) {
    comms = { unpack_nccl_comm(obj) };
  } else {
    auto seq = THPObjectPtr(PySequence_Fast(obj, "comm is not a sequence"));
    if (!seq) throw python_error();
    auto size = PySequence_Fast_GET_SIZE(seq.get());
    comms = std::vector<ncclComm_t>(size);
    for (int64_t i = 0; i < size; i++) {
      comms[i] = unpack_nccl_comm(PySequence_Fast_GET_ITEM(seq.get(), i));
    }
  }
  if (comms.size() != size) {
    throw std::runtime_error("number of communicators is not equal to number of inputs");
  }
  return comms;
}

PyObject * THCPModule_nccl_init_rank(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  int nranks;
  const char* id;
  Py_ssize_t id_len;
  int rank;

  if (!PyArg_ParseTuple(args, "is#i:nccl_init_rank", &nranks, &id, &id_len, &rank)) {
    return NULL;
  }
  THPUtils_assert(id_len == NCCL_UNIQUE_ID_BYTES,
      "invalid unqiue_id (expected %d bytes, got %zd)",
      NCCL_UNIQUE_ID_BYTES, id_len);

  ncclUniqueId commId;
  memcpy(&commId, id, NCCL_UNIQUE_ID_BYTES);
  ncclComm_t comm;
  with_no_gil([&]{
    CHECK(ncclCommInitRank(&comm, nranks, commId, rank));
  });
  return PyCapsule_New(comm, COMM_CAPSULE_NAME, &destroy_nccl_comm);
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_nccl_reduce(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams, *_comms;
  int root, op;

  if (!PyArg_ParseTuple(args, "OOiiOO", &_inputs, &_outputs, &root, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(args, NULL, "nccl_reduce", 1,
			      "(sequence[Tensor] inputs, sequence[Tensor] outputs, int root,"
            " int op, sequence[torch.cuda.Stream or None]");
    return NULL;
  }

  std::vector<at::Tensor> inputs = THPUtils_PySequence_to_TensorList(_inputs);
  std::vector<at::Tensor> outputs = THPUtils_PySequence_to_TensorList(_outputs);
  std::vector<THCStream*> streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  THPUtils_assert(root >= 0 && (size_t)root < inputs.size(), "invalid root");

  with_no_gil([&]{
    _check_inputs(inputs, outputs, 1, 1);
    size_t len = inputs.size();

    ncclDataType_t data_type = _get_data_type(inputs[0].type());

    int64_t count = inputs[0].numel();
    std::lock_guard<std::mutex> lock(*(THCCachingAllocator_getCudaFreeMutex()));
    auto comms = user_comms.empty() ? _get_communicators(inputs) : ArrayRef<ncclComm_t>(user_comms);
    AutoGPU gpu_guard;
    AutoNcclGroup nccl_group_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      gpu_guard.setDevice(device);
      auto stream = (streams[i] == NULL) ? NULL : streams[i]->stream;
      CHECK(ncclReduce(inputs[i].data_ptr(), outputs[i].data_ptr(),
           count, data_type, (ncclRedOp_t) op, root, comms[i], stream));
    }
  });

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_nccl_all_reduce(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams, *_comms;
  int op;

  if (!PyArg_ParseTuple(args, "OOiOO", &_inputs, &_outputs, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(args, NULL, "nccl_all_reduce", 1,
        "(sequence[Tensor] inputs, sequence[Tensor] outputs, int op,"
        " sequence[torch.cuda.Stream] streams,"
        " sequence[torch.cuda.nccl.Communicator] comms)");
    return NULL;
  }

  std::vector<at::Tensor> inputs = THPUtils_PySequence_to_TensorList(_inputs);
  std::vector<at::Tensor> outputs = THPUtils_PySequence_to_TensorList(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  with_no_gil([&]{
    _check_inputs(inputs, outputs, 1, 1);
    size_t len = inputs.size();

    ncclDataType_t data_type = _get_data_type(inputs[0].type());

    int64_t count = inputs[0].numel();
    std::lock_guard<std::mutex> lock(*(THCCachingAllocator_getCudaFreeMutex()));
    auto comms = user_comms.empty() ? _get_communicators(inputs) : ArrayRef<ncclComm_t>(user_comms);
    AutoGPU gpu_guard;
    AutoNcclGroup nccl_group_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      gpu_guard.setDevice(device);
      auto stream = (streams[i] == NULL) ? NULL : streams[i]->stream;
      CHECK(ncclAllReduce(inputs[i].data_ptr(), outputs[i].data_ptr(),
          count, data_type, (ncclRedOp_t) op, comms[i], stream));
    }
  });

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_nccl_broadcast(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_streams, *_comms;
  int root;

  if (!PyArg_ParseTuple(args, "OiOO", &_inputs, &root, &_streams, &_comms)) {
    THPUtils_invalidArguments(args, NULL, "nccl_broadcast", 1,
			      "(sequence[Tensor] inputs, int root)");
    return NULL;
  }

  std::vector<at::Tensor> inputs = THPUtils_PySequence_to_TensorList(_inputs);
  THPUtils_assert(root >= 0 && (size_t)root < inputs.size(), "invalid root");
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  with_no_gil([&]{
    _check_inputs(inputs, inputs, 1, 1);
    size_t len = inputs.size();

    ncclDataType_t data_type = _get_data_type(inputs[0].type());

    int64_t count = inputs[0].numel();
    std::lock_guard<std::mutex> lock(*(THCCachingAllocator_getCudaFreeMutex()));
    auto comms = user_comms.empty() ? _get_communicators(inputs) : ArrayRef<ncclComm_t>(user_comms);
    AutoGPU gpu_guard;
    AutoNcclGroup nccl_group_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      gpu_guard.setDevice(device);
      auto stream = (streams[i] == NULL) ? NULL : streams[i]->stream;
      CHECK(ncclBcast(inputs[i].data_ptr(), count, data_type, root, comms[i], stream));
    }
  });

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_nccl_all_gather(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams, *_comms;

  if (!PyArg_ParseTuple(args, "OOOO", &_inputs, &_outputs, &_streams, &_comms)) {
    THPUtils_invalidArguments(args, NULL, "nccl_all_gather", 1,
			      "(sequence[Tensor] inputs, sequence[Tensor] outputs");
    return NULL;
  }

  std::vector<at::Tensor> inputs = THPUtils_PySequence_to_TensorList(_inputs);
  std::vector<at::Tensor> outputs = THPUtils_PySequence_to_TensorList(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  with_no_gil([&]{
    size_t len = inputs.size();
    _check_inputs(inputs, outputs, len, 1);

    ncclDataType_t data_type = _get_data_type(inputs[0].type());

    int64_t count = inputs[0].numel();
    std::lock_guard<std::mutex> lock(*(THCCachingAllocator_getCudaFreeMutex()));
    auto comms = user_comms.empty() ? _get_communicators(inputs) : ArrayRef<ncclComm_t>(user_comms);
    AutoGPU gpu_guard;
    AutoNcclGroup nccl_group_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      gpu_guard.setDevice(device);
      auto stream = (streams[i] == NULL) ? NULL : streams[i]->stream;
    #if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
      CHECK(ncclAllGather(inputs[i].data_ptr(), outputs[i].data_ptr(),
        count, data_type, comms[i], stream));
    #else
      CHECK(ncclAllGather(inputs[i].data_ptr(), count, data_type,
        outputs[i].data_ptr(), comms[i], stream));
    #endif
    }
  });

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject * THCPModule_nccl_reduce_scatter(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams, *_comms;
  int op;

  if (!PyArg_ParseTuple(args, "OOiOO", &_inputs, &_outputs, &op, &_streams, &_comms)) {
    THPUtils_invalidArguments(args, NULL, "nccl_reduce_scatter", 1,
			      "(sequence[Tensor] inputs, sequence[Tensor] outputs, int op");
    return NULL;
  }

  std::vector<at::Tensor> inputs = THPUtils_PySequence_to_TensorList(_inputs);
  std::vector<at::Tensor> outputs = THPUtils_PySequence_to_TensorList(_outputs);
  auto streams = unpack_streams(_streams, inputs.size());
  auto user_comms = unpack_comms(_comms, inputs.size());

  with_no_gil([&]{
    size_t len = inputs.size();
    _check_inputs(inputs, outputs, 1, len);

    ncclDataType_t data_type = _get_data_type(inputs[0].type());

    int64_t count = inputs[0].numel() / len;
    std::lock_guard<std::mutex> lock(*(THCCachingAllocator_getCudaFreeMutex()));
    auto comms = user_comms.empty() ? _get_communicators(inputs) : ArrayRef<ncclComm_t>(user_comms);
    AutoGPU gpu_guard;
    AutoNcclGroup nccl_group_guard;
    for (size_t i = 0; i < len; i++) {
      int device = inputs[i].get_device();
      gpu_guard.setDevice(device);
      auto stream = (streams[i] == NULL) ? NULL : streams[i]->stream;
      CHECK(ncclReduceScatter(inputs[i].data_ptr(), outputs[i].data_ptr(),
          count, data_type, (ncclRedOp_t) op, comms[i], stream));
    }
  });

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
