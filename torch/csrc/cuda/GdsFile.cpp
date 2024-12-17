#include <c10/util/error.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#if defined(USE_CUFILE)
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cufile.h>

namespace {
// To get error message for cuFileRead/Write APIs that return ssize_t (-1 for
// filesystem error and a negative CUfileOpError enum value otherwise).
template <
    class T,
    std::enable_if_t<std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                               : std::string(c10::utils::str_error(errno));
}

// To get error message for Buf/Handle registeration APIs that return
// CUfileError_t
template <
    class T,
    std::enable_if_t<!std::is_integral_v<T>, std::nullptr_t> = nullptr>
std::string cuGDSFileGetErrorString(T status) {
  std::string errStr = cuGDSFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status))
    errStr.append(".").append(
        cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}
} // namespace

void gds_load_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, (void*)dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileRead failed: ", cuGDSFileGetErrorString(ret));
}

void gds_save_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

void gds_register_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  CUfileError_t status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileBufRegister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

void gds_deregister_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  CUfileError_t status = cuFileBufDeregister(dataPtr);
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileBufDeregister failed: ",
      cuGDSFileGetErrorString(status));
  return;
}

int64_t gds_register_handle(int fd) {
  CUfileDescr_t cf_descr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUfileHandle_t cf_handle;
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    TORCH_CHECK(
        false,
        "cuFileHandleRegister failed: ",
        cuGDSFileGetErrorString(status));
  }

  // Returning cuFileHandle_t as int64_t
  return reinterpret_cast<int64_t>(cf_handle);
}

void gds_deregister_handle(int64_t handle) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  cuFileHandleDeregister(cf_handle);
}

#endif

namespace torch::cuda::shared {

void initGdsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

#if defined(USE_CUFILE)
  m.def("_gds_register_handle", &gds_register_handle);
  m.def("_gds_deregister_handle", &gds_deregister_handle);
  m.def("_gds_register_buffer", &gds_register_buffer);
  m.def("_gds_deregister_buffer", &gds_deregister_buffer);
  m.def("_gds_load_storage", &gds_load_storage);
  m.def("_gds_save_storage", &gds_save_storage);
#endif
}

} // namespace torch::cuda::shared
