#include <c10/util/error.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/GdsFile.h>
#include <torch/csrc/utils/pybind.h>

#if defined(USE_CUFILE)
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

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

// To get error message for Buf/Handle registration APIs that return
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

// Helper to get the correct stream and do some error handling.
c10::cuda::CUDAStream getGdsStream(
    const at::Storage& storage,
    std::optional<c10::Stream> stream) {
  // Make sure device of stream and device of storage device matches.
  TORCH_CHECK(
      !stream.has_value() || stream->device() == storage.device(),
      "Expected a stream on ",
      storage.device(),
      " but got a stream on ",
      stream->device());
  c10::cuda::CUDAStream cuda_stream = stream.has_value()
      ? c10::cuda::CUDAStream(*stream)
      : c10::cuda::getCurrentCUDAStream(storage.device().index());

  // cuFile's async API does not support CUDA graphs:
  // https://docs.nvidia.com/gpudirect-storage/release-notes/index.html#known-limitations
  cudaStreamCaptureStatus status{};
  C10_CUDA_CHECK(cudaStreamIsCapturing(cuda_stream.stream(), &status));
  TORCH_CHECK(
      status == cudaStreamCaptureStatusNone,
      "GDS load_storage/save_storage is not supported during CUDA graph capture");
  return cuda_stream;
}
} // namespace

void gds_load_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset,
    std::optional<c10::Stream> stream) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());
  c10::cuda::CUDAStream cuda_stream = getGdsStream(storage, stream);

  void* dataPtr = storage.mutable_data();
  size_t nbytes = storage.nbytes();
  off_t buf_offset = 0;
  ssize_t bytes_read = 0;

  // Use the stream-ordered API to avoid race conditions with other async
  // operations enqueued in the stream.
  CUfileError_t status = cuFileReadAsync(
      cf_handle,
      dataPtr,
      &nbytes,
      &offset,
      &buf_offset,
      &bytes_read,
      cuda_stream.stream());
  // Synchronize immediately after to avoid race conditions with other host
  // operations, such as open(file).read().
  cuda_stream.synchronize();
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileReadAsync failed to enqueue: ",
      cuGDSFileGetErrorString(status));
  TORCH_CHECK(
      bytes_read >= 0,
      "cuFileReadAsync I/O failed: ",
      cuGDSFileGetErrorString(bytes_read));
}

void gds_save_storage(
    int64_t handle,
    const at::Storage& storage,
    off_t offset,
    std::optional<c10::Stream> stream) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());
  c10::cuda::CUDAStream cuda_stream = getGdsStream(storage, stream);

  void* dataPtr = storage.mutable_data();
  size_t nbytes = storage.nbytes();
  off_t buf_offset = 0;
  ssize_t bytes_written = 0;

  // Use the stream-ordered API to avoid race conditions with other async
  // operations enqueued in the stream.
  CUfileError_t status = cuFileWriteAsync(
      cf_handle,
      dataPtr,
      &nbytes,
      &offset,
      &buf_offset,
      &bytes_written,
      cuda_stream.stream());
  // Synchronize immediately after to avoid race conditions with other host
  // operations, such as open(file).read().
  cuda_stream.synchronize();
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      "cuFileWriteAsync failed to enqueue: ",
      cuGDSFileGetErrorString(status));
  TORCH_CHECK(
      bytes_written >= 0,
      "cuFileWriteAsync I/O failed: ",
      cuGDSFileGetErrorString(bytes_written));
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
  CUfileHandle_t cf_handle{};
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
  m.def(
      "_gds_load_storage",
      &gds_load_storage,
      py::arg("handle"),
      py::arg("storage"),
      py::arg("offset"),
      py::arg("stream") = std::nullopt,
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "_gds_save_storage",
      &gds_save_storage,
      py::arg("handle"),
      py::arg("storage"),
      py::arg("offset"),
      py::arg("stream") = std::nullopt,
      py::call_guard<py::gil_scoped_release>());
#endif
}

} // namespace torch::cuda::shared
