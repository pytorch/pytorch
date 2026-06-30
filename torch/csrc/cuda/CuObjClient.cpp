#include <c10/util/error.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/CuObjClient.h>
#include <torch/csrc/utils/pybind.h>

// Thin binding over NVIDIA cuObject (libcuobjclient) implementing the
// "manual RDMA token" pattern (cuObjClient API spec section 1.12.4). PyTorch
// owns only the RDMA data plane here: buffer registration and RDMA descriptor
// (token) lifetime. The S3 control plane (issuing the GET/PUT carrying the
// x-amz-rdma-token header) lives in Python so that no HTTP/S3 SDK is linked
// into libtorch. See torch/distributed/checkpoint/_cuobj_rdma_storage.py.
//
// NOTE: cuObject ships with CUDA Toolkit >= 13.1.1. The exact header name and
// enum spellings below are written against the published API spec and must be
// validated against the installed headers (the live cluster carries them); any
// drift is isolated entirely to this file.

#if defined(USE_CUOBJ)
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuobjclient.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace {

// cuObject get/put callbacks are unused in the manual-token pattern (Python
// drives the S3 request out of band), but the cuObjClient constructor requires
// an ops table. Provide stubs.
ssize_t cuobj_stub_get(
    const void* /*handle*/,
    char* /*ptr*/,
    size_t /*size*/,
    loff_t /*offset*/,
    const cufileRDMAInfo_t* /*rdma_info*/) {
  return -EOPNOTSUPP;
}

ssize_t cuobj_stub_put(
    const void* /*handle*/,
    const char* /*ptr*/,
    size_t /*size*/,
    loff_t /*offset*/,
    const cufileRDMAInfo_t* /*rdma_info*/) {
  return -EOPNOTSUPP;
}

// Process-wide cuObjClient. Lazily constructed; threading is guarded by the
// caller holding the GIL plus the token registry mutex below.
cuObjClient* getClient() {
  static CUObjOps_t ops = {cuobj_stub_get, cuobj_stub_put};
  static cuObjClient client(ops);
  return &client;
}

// Tokens returned by cuMemObjGetRDMAToken are owned by cuObject and must be
// released via cuMemObjPutRDMAToken once the S3 request has completed. We hand
// Python a copy of the descriptor string and keep the original pointer here so
// it can be freed by token value when Python is done with it.
std::mutex& tokenMutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, char*>& tokenRegistry() {
  static std::unordered_map<std::string, char*> r;
  return r;
}

std::string cuObjGetErrorString(cuObjErr_t status) {
  return std::string("cuObject error code ") +
      std::to_string(static_cast<int>(status));
}

} // namespace

static bool cuobj_available() {
  return getClient()->isConnected();
}

static void cuobj_register_buffer(const at::Storage& storage) {
  c10::cuda::OptionalCUDAGuard gpuGuard;
  if (storage.device().is_cuda()) {
    gpuGuard.set_index(storage.device().index());
  }
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();
  cuObjErr_t status = getClient()->cuMemObjGetDescriptor(dataPtr, nbytes);
  TORCH_CHECK(
      status == CU_OBJ_SUCCESS,
      "cuMemObjGetDescriptor failed: ",
      cuObjGetErrorString(status));
}

static void cuobj_deregister_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  cuObjErr_t status = getClient()->cuMemObjPutDescriptor(dataPtr);
  TORCH_CHECK(
      status == CU_OBJ_SUCCESS,
      "cuMemObjPutDescriptor failed: ",
      cuObjGetErrorString(status));
}

// Returns an RDMA descriptor (token) string for [offset, offset+size) of the
// registered storage. is_put selects PUT vs GET semantics. The caller must pass
// the returned string to cuobj_put_rdma_token after the S3 request completes.
static std::string cuobj_get_rdma_token(
    const at::Storage& storage,
    int64_t size,
    int64_t offset,
    bool is_put) {
  void* dataPtr = storage.mutable_data();
  char* desc = nullptr;
  cuObjOpType_t op = is_put ? CUOBJ_PUT : CUOBJ_GET;
  cuObjErr_t status = getClient()->cuMemObjGetRDMAToken(
      dataPtr,
      static_cast<size_t>(size),
      static_cast<size_t>(offset),
      op,
      &desc);
  TORCH_CHECK(
      status == CU_OBJ_SUCCESS && desc != nullptr,
      "cuMemObjGetRDMAToken failed: ",
      cuObjGetErrorString(status));
  std::string token(desc);
  {
    std::lock_guard<std::mutex> lock(tokenMutex());
    tokenRegistry()[token] = desc;
  }
  return token;
}

static void cuobj_put_rdma_token(const std::string& token) {
  char* desc = nullptr;
  {
    std::lock_guard<std::mutex> lock(tokenMutex());
    auto it = tokenRegistry().find(token);
    if (it == tokenRegistry().end()) {
      return;
    }
    desc = it->second;
    tokenRegistry().erase(it);
  }
  cuObjErr_t status = getClient()->cuMemObjPutRDMAToken(desc);
  TORCH_CHECK(
      status == CU_OBJ_SUCCESS,
      "cuMemObjPutRDMAToken failed: ",
      cuObjGetErrorString(status));
}

#endif

namespace torch::cuda::shared {

void initCuObjBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#if defined(USE_CUOBJ)
  m.def("_cuobj_available", &cuobj_available);
  m.def("_cuobj_register_buffer", &cuobj_register_buffer);
  m.def("_cuobj_deregister_buffer", &cuobj_deregister_buffer);
  m.def("_cuobj_get_rdma_token", &cuobj_get_rdma_token);
  m.def("_cuobj_put_rdma_token", &cuobj_put_rdma_token);
#endif
}

} // namespace torch::cuda::shared
