#include <ATen/cuda/CUDAGdsFile.h>

// torch
#include <c10/cuda/CUDAGuard.h>

// cuda
#include <cuda_runtime.h>
#include <cufile.h>
#include <chrono>
#include <iostream>

// file io
#include <unistd.h>
#include <fcntl.h>


namespace at::cuda {

// POSIX
template <
    class T,
    typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type =
        nullptr>
std::string cuGDSFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                               : std::string(std::strerror(errno));
}

// CUfileError_t
template <
    class T,
    typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type =
        nullptr>
std::string cuGDSFileGetErrorString(T status) {
  std::string errStr = cuGDSFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status))
    errStr.append(".").append(cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}

void gds_register_buffer(const at::Tensor& tensor) {
  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  CUfileError_t status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufRegister failed: ", cuGDSFileGetErrorString(status));
  return;
}

void gds_register_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  CUfileError_t status = cuFileBufRegister(dataPtr, nbytes, 0);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufRegister failed: ", cuGDSFileGetErrorString(status));
  return;
}

void gds_deregister_buffer(const at::Tensor& tensor) {
  void* dataPtr = tensor.data_ptr();
  CUfileError_t status = cuFileBufDeregister(dataPtr);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufDeregister failed: ", cuGDSFileGetErrorString(status));
  return;
}

void gds_deregister_buffer(const at::Storage& storage) {
  void* dataPtr = storage.mutable_data();
  CUfileError_t status = cuFileBufDeregister(dataPtr);
  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufDeregister failed: ", cuGDSFileGetErrorString(status));
  return;
}

GDSFile::GDSFile() : is_open(false) {};

GDSFile::GDSFile(c10::string_view filename, c10::string_view mode) : filename(filename), mode(mode), is_open(false) {
  open(filename, mode);
}

GDSFile::~GDSFile() {
  if (is_open) {
    close();
  }
}

void GDSFile::open(c10::string_view other_filename, c10::string_view other_mode) {
  TORCH_CHECK(is_open == false, "file", filename, "is already open");
  if (!filename.empty()) {
    TORCH_CHECK(other_filename == filename, "file", filename, "is already open with mode", mode);
  }
  if (!mode.empty()) {
    TORCH_CHECK(other_mode == mode, "file", filename, "is already open with mode", mode);
  }

  // Open the binary file
  if (mode == "r") {
    // for reading
    fd = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
  } else if (mode == "w") {
    // for writing
    fd = ::open(filename.c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0664);
  } else {
    TORCH_CHECK(false, "only r and w modes are currently supported, but got:", mode);
  }
  TORCH_CHECK(fd >= 0, "fcntl cannot open file: ", filename);

  // Register cuGDSFile handle
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    TORCH_CHECK(false, "cuFileHandleRegister failed: ", cuGDSFileGetErrorString(status));
  }
  is_open = true;
}

void GDSFile::close() {
  // Deregister cuGDSFile handle and close the file
  if (is_open) {
      cuFileHandleDeregister(cf_handle);
      ::close(fd);
      fd = -1;
  }
  is_open = false;
}

void GDSFile::load_tensor(const at::Tensor& tensor, off_t offset) {
  TORCH_CHECK(mode == "r", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, (void*)dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileRead failed: ", cuGDSFileGetErrorString(ret));
}

void GDSFile::load_storage(const at::Storage& storage, off_t offset) {
  TORCH_CHECK(mode == "r", filename, " was opened for read only");
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Read the binary file
  ssize_t ret = cuFileRead(cf_handle, (void*)dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileRead failed: ", cuGDSFileGetErrorString(ret));
}

void GDSFile::save_tensor(const at::Tensor& tensor, off_t offset) {
  TORCH_CHECK(mode == "w", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtr = tensor.data_ptr();
  const size_t nbytes = tensor.nbytes();

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

void GDSFile::save_storage(const at::Storage& storage, off_t offset) {
  TORCH_CHECK(mode == "w", filename, " was opened for write only");
  c10::cuda::CUDAGuard gpuGuard(storage.device());

  // FIXME: check whether storage.mutable_data() is the correct API to call here
  void* dataPtr = storage.mutable_data();
  const size_t nbytes = storage.nbytes();

  // Write device memory contents to the file
  ssize_t ret = cuFileWrite(cf_handle, dataPtr, nbytes, offset, 0);
  TORCH_CHECK(ret >= 0, "cuFileWrite failed: ", cuGDSFileGetErrorString(ret));
}

} // namespace at::cuda
