#include <c10/util/error.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/cuda/GdsFile.h>
#include <torch/csrc/utils/pybind.h>

#if defined(USE_CUFILE)
#include <ATen/ATen.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <cufile.h>

#if defined(USE_ROCM)
#include <unistd.h>
#include <climits>
#endif

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

// hipFile silently truncates a transfer to the kernel's MAX_RW_COUNT (INT_MAX
// rounded down to a page), so reject larger ones rather than transfer part of
// the storage. Remove once hipFile supports larger transfers.
void check_transfer_size(
    [[maybe_unused]] const char* fn_name,
    [[maybe_unused]] size_t nbytes) {
#if defined(USE_ROCM)
  static const size_t max_nbytes = static_cast<size_t>(INT_MAX) &
      ~(static_cast<size_t>(sysconf(_SC_PAGESIZE)) - 1);
  TORCH_CHECK(
      nbytes <= max_nbytes,
      fn_name,
      " transfers at most ",
      max_nbytes,
      " bytes per call on ROCm, but the storage has ",
      nbytes,
      " bytes");
#endif
}

using GdsAsyncIOFn = decltype(&cuFileReadAsync);

// Enqueues fn on the current stream of the storage's device and returns a
// 0-dim tensor that holds the number of bytes transferred once the stream has
// passed the I/O.
at::Tensor gds_io_async(
    GdsAsyncIOFn fn,
    const char* fn_name,
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  CUfileHandle_t cf_handle = reinterpret_cast<CUfileHandle_t>(handle);
  c10::cuda::CUDAGuard gpuGuard(storage.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  // The stream APIs do not support CUDA graphs. Reject before enqueueing
  // anything so the caller's capture stays valid.
  TORCH_CHECK(
      !stream.is_capturing(),
      fn_name,
      " is not supported during CUDA graph capture");
  check_transfer_size(fn_name, storage.nbytes());

  // The size, offsets and byte count are accessed after this call returns.
  // Keep them in pinned memory that the caching host allocator does not reuse
  // until the stream passes the event recorded below, even if the caller drops
  // the result.
  static_assert(sizeof(size_t) == sizeof(int64_t));
  at::Tensor params =
      at::empty({4}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  int64_t* p = params.mutable_data_ptr<int64_t>();
  p[0] = static_cast<int64_t>(storage.nbytes());
  p[1] = offset;
  p[2] = 0;
  p[3] = 0;

  CUfileError_t status =
      fn(cf_handle,
         storage.mutable_data(),
         reinterpret_cast<size_t*>(&p[0]),
         &p[1],
         &p[2],
         &p[3],
         stream.stream());
  // Record even if fn failed, since it may have enqueued work that uses p.
  TORCH_INTERNAL_ASSERT(at::getHostAllocator(at::kCUDA)->record_event(
      p, params.storage().data_ptr().get_context(), stream.unwrap()));
  TORCH_CHECK(
      status.err == CU_FILE_SUCCESS,
      fn_name,
      " failed: ",
      cuGDSFileGetErrorString(status));
  return params.select(0, 3);
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
  check_transfer_size("cuFileRead", nbytes);

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, dataPtr, nbytes, offset, 0);
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
  check_transfer_size("cuFileWrite", nbytes);

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

at::Tensor gds_load_storage_async(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  return gds_io_async(
      cuFileReadAsync, "cuFileReadAsync", handle, storage, offset);
}

at::Tensor gds_save_storage_async(
    int64_t handle,
    const at::Storage& storage,
    off_t offset) {
  return gds_io_async(
      cuFileWriteAsync, "cuFileWriteAsync", handle, storage, offset);
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
  m.def("_gds_load_storage", &gds_load_storage);
  m.def("_gds_save_storage", &gds_save_storage);
  m.def("_gds_load_storage_async", &gds_load_storage_async);
  m.def("_gds_save_storage_async", &gds_save_storage_async);
#endif
}

} // namespace torch::cuda::shared
