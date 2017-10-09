#include "nccl.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/Types.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/cuda/THCP.h"

#include <nccl.h>
#include <sstream>
#include <unordered_map>

static inline void CHECK(ncclResult_t status)
{
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
	ncclCommDestroy(comms[i]);
      }
    }
  }
};

// accesses to this object have to be guarded by THC's CudaFreeMutex
std::unordered_map<std::string, NcclCommList > _communicators;

static ncclComm_t* _get_communicator(std::vector<at::Tensor>& inputs) {
  int ndevices = inputs.size();
  std::stringstream hash_stream;
  std::vector<int> devs;
  for (int i = 0; i < ndevices; i++) {
    int dev = inputs[i].get_device();
    hash_stream <<  dev << ",";
    devs.push_back(dev);
  }
  std::string hash = hash_stream.str();
  auto it = _communicators.find(hash);
  if (it == _communicators.end()) {
    return _communicators.emplace_hint(it, hash, devs)->second.comms.get();
  } else {
    return it->second.comms.get();
  }
}

static void _check_inputs(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs, int size_multiplier) {
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
  auto type = inputs[0].type().ID();

  for (size_t i = 0; i < len; i++) {
    auto input = inputs[i];
    auto output = outputs[i];

    if (!(input.type().isCuda() && !input.type().isSparse()
	  && output.type().isCuda()  && !output.type().isSparse())) {
      throw std::runtime_error("input and output elements have to be cuda dense Tensors");
    }

    if (type != input.type().ID() || type != output.type().ID()) {
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
  
    // outputs have to be of size * size_multiplier
    if (output.numel() != (numel * size_multiplier)) {
      throw std::runtime_error("output must be of size input_size * size_multiplier");
    }
  }
}

static ncclDataType_t _get_data_type(at::TypeID type) {
  switch(type) {
  case at::TypeID::CUDAFloat   : return ncclFloat;
  case at::TypeID::CUDAHalf    : return ncclHalf;
  case at::TypeID::CUDADouble  : return ncclDouble;
  case at::TypeID::CUDALong    : return ncclInt64;
  case at::TypeID::CUDAInt     : return ncclInt;
  case at::TypeID::CUDAChar    : return ncclChar;
  case at::TypeID::CUDAByte    : return ncclChar;
  default: throw std::runtime_error("Unconvertible NCCL type");
  }
}

PyObject * THCPModule_nccl_reduce(PyObject *self, PyObject *args) {
  HANDLE_TH_ERRORS
  PyObject *_inputs, *_outputs, *_streams;
  int root, op;

  if (!PyArg_ParseTuple(args, "OOOii", &_inputs, &_outputs, &_streams, &root, &op)) {
    THPUtils_invalidArguments(args, NULL, "nccl_reduce", 1,
			      "(sequence[Tensor] inputs, sequence[Tensor]"
			      " outputs, sequence[torch.cuda.Stream or None], int root, int op");
    return NULL;
  }

  std::vector<at::Tensor> inputs = THPUtils_PySequence_to_TensorList(_inputs);
  std::vector<at::Tensor> outputs = THPUtils_PySequence_to_TensorList(_outputs);
  std::vector<THCStream*> streams = THPUtils_PySequence_to_THCStreamList(_streams);

  THPUtils_assert(inputs.size() == streams.size(), "number of streams is not equal to number of inputs");
  
  // we can safely release GIL after this line, no python API used
  AutoNoGIL no_gil;
  _check_inputs(inputs, outputs, 1);
  size_t len = inputs.size();

  ncclDataType_t data_type = _get_data_type(inputs[0].type().ID());

  int64_t count = inputs[0].numel();
  std::lock_guard<std::mutex> lock(*(THCCachingAllocator_getCudaFreeMutex()));
  ncclComm_t *comm = _get_communicator(inputs);
  AutoGPU gpu_guard;
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  CHECK(ncclGroupStart());
#endif
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].get_device();
    gpu_guard.setDevice(device);
    auto stream = (streams[i] == NULL) ? NULL : streams[i]->stream;
    CHECK(ncclReduce(inputs[i].data_ptr(), outputs[i].data_ptr(),
		     count, data_type, (ncclRedOp_t) op, root, comm[i], stream));
  }
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  CHECK(ncclGroupEnd());
#endif

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
