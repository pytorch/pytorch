#pragma once

#include <cufile.h>
#include <c10/util/string_view.h>

#include <ATen/Tensor.h>
#include <ATen/Storage.h>

namespace at::cuda {
  TORCH_CUDA_CPP_API void gds_register_buffer(const at::Tensor& tensor);
  TORCH_CUDA_CPP_API void gds_register_buffer(const at::Storage& storage);
  TORCH_CUDA_CPP_API void gds_deregister_buffer(const at::Tensor& tensor);
  TORCH_CUDA_CPP_API void gds_deregister_buffer(const at::Storage& storage);

  class TORCH_CUDA_CPP_API GDSFile {
    public:
    GDSFile();
    GDSFile(c10::string_view filename, c10::string_view mode);
    ~GDSFile();

    void open(c10::string_view filename, c10::string_view mode);
    void close();

    void load_tensor(const at::Tensor& tensor, off_t offset);
    void save_tensor(const at::Tensor& tensor, off_t offset);

    void load_storage(const at::Storage& storage, off_t offset);
    void save_storage(const at::Storage& storage, off_t offset);

    private:
    std::string filename;
    std::string mode;

    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;

    int fd = -1;
    bool is_open = false;
  };
} // namespace at::cuda
