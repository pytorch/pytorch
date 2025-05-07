#include <ATen/native/quantized/cpu/qlinear.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/GradMode.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/mkldnn_tensor.h>
#include <torch/csrc/inductor/aoti_torch/oss_proxy_executor.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/stable/library.h>
#include <torch/library.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else

#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_mm.h>
#include <ATen/ops/_wrapped_linear_prepack.h>
#include <ATen/ops/_wrapped_quantized_linear_prepacked.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/index_put.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/scatter_reduce.h>
#include <ATen/ops/view_as_real_ops.h>
#include <ATen/ops/view_ops.h>

#endif

#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <climits>

#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

// HACK for failed builds in ARVR, where it cannot find these symbols within
// std::experimental::filesystem
namespace {
std::string get_current_path() {
#ifdef _WIN32
  return fs::current_path().string();
#else
  // NOLINTNEXTLINE(*array*)
  char currentPath[PATH_MAX]{};
  if (getcwd(currentPath, sizeof(currentPath)) != nullptr) {
    return std::string(currentPath);
  } else {
    throw std::runtime_error("Failed to get current path");
  }
#endif
}

bool file_exists(std::string& path) {
#ifdef _WIN32
  return fs::exists(path);
#else
  struct stat rc {};
  return lstat(path.c_str(), &rc) == 0;
#endif
}

bool create_directories(const std::string& path) {
#ifdef _WIN32
  return fs::create_directories(path);
#else
  if (mkdir(path.c_str(), 0777) == -1) {
    throw std::runtime_error("Failed to create directory");
  }
  return true;
#endif
}
} // namespace

using namespace torch::aot_inductor;

namespace {
static c10::Device c10_device(int32_t device_type, int32_t device_index) {
  if (device_type == aoti_torch_device_type_cpu()) {
    return c10::Device(static_cast<c10::DeviceType>(device_type));
  } else {
    return c10::Device(
        static_cast<c10::DeviceType>(device_type),
        static_cast<c10::DeviceIndex>(device_index));
  }
}
} // namespace

const int AOTI_TORCH_MAX_NUMEL_TO_PRINT = 64;

#define AOTI_TORCH_DEVICE_TYPE_IMPL(device_str, device_type) \
  int32_t aoti_torch_device_type_##device_str() {            \
    return (int32_t)c10::DeviceType::device_type;            \
  }

AOTI_TORCH_DEVICE_TYPE_IMPL(cpu, CPU)
AOTI_TORCH_DEVICE_TYPE_IMPL(cuda, CUDA)
AOTI_TORCH_DEVICE_TYPE_IMPL(meta, Meta)
AOTI_TORCH_DEVICE_TYPE_IMPL(xpu, XPU)
AOTI_TORCH_DEVICE_TYPE_IMPL(privateuse1, PrivateUse1)
#undef AOTI_TORCH_DEVICE_TYPE_IMPL

#define AOTI_TORCH_DTYPE_IMPL(dtype, stype) \
  int32_t aoti_torch_dtype_##dtype() {      \
    return (int32_t)c10::ScalarType::stype; \
  }

AOTI_TORCH_DTYPE_IMPL(float8_e5m2, Float8_e5m2)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fn, Float8_e4m3fn)
AOTI_TORCH_DTYPE_IMPL(float8_e5m2fnuz, Float8_e5m2fnuz)
AOTI_TORCH_DTYPE_IMPL(float8_e4m3fnuz, Float8_e4m3fnuz)
AOTI_TORCH_DTYPE_IMPL(bfloat16, BFloat16)
AOTI_TORCH_DTYPE_IMPL(float16, Half)
AOTI_TORCH_DTYPE_IMPL(float32, Float)
AOTI_TORCH_DTYPE_IMPL(float64, Double)
AOTI_TORCH_DTYPE_IMPL(uint8, Byte)
AOTI_TORCH_DTYPE_IMPL(uint16, UInt16)
AOTI_TORCH_DTYPE_IMPL(uint32, UInt32)
AOTI_TORCH_DTYPE_IMPL(uint64, UInt64)
AOTI_TORCH_DTYPE_IMPL(int8, Char)
AOTI_TORCH_DTYPE_IMPL(int16, Short)
AOTI_TORCH_DTYPE_IMPL(int32, Int)
AOTI_TORCH_DTYPE_IMPL(int64, Long)
AOTI_TORCH_DTYPE_IMPL(bool, Bool)
AOTI_TORCH_DTYPE_IMPL(complex32, ComplexHalf)
AOTI_TORCH_DTYPE_IMPL(complex64, ComplexFloat)
AOTI_TORCH_DTYPE_IMPL(complex128, ComplexDouble)
#undef AOTI_TORCH_DTYPE_IMPL

#define AOTI_TORCH_LAYOUT_IMPL(name, enum) \
  int32_t aoti_torch_layout_##name() {     \
    return (int32_t)at::Layout::enum;      \
  }

AOTI_TORCH_LAYOUT_IMPL(strided, Strided)
AOTI_TORCH_LAYOUT_IMPL(sparse_coo, Sparse)
AOTI_TORCH_LAYOUT_IMPL(sparse_csr, SparseCsr)
AOTI_TORCH_LAYOUT_IMPL(sparse_csc, SparseCsc)
AOTI_TORCH_LAYOUT_IMPL(sparse_bsr, SparseBsr)
AOTI_TORCH_LAYOUT_IMPL(sparse_bsc, SparseBsc)
AOTI_TORCH_LAYOUT_IMPL(_mkldnn, Mkldnn)
AOTI_TORCH_LAYOUT_IMPL(jagged, Jagged)
#undef AOTI_TORCH_LAYOUT_IMPL

#define AOTI_TORCH_MEMORY_FORMAT_IMPL(name, enum) \
  int32_t aoti_torch_memory_format_##name() {     \
    return (int32_t)at::MemoryFormat::enum;       \
  }

AOTI_TORCH_MEMORY_FORMAT_IMPL(contiguous_format, Contiguous)
AOTI_TORCH_MEMORY_FORMAT_IMPL(channels_last, ChannelsLast)
AOTI_TORCH_MEMORY_FORMAT_IMPL(channels_last_3d, ChannelsLast3d)
AOTI_TORCH_MEMORY_FORMAT_IMPL(preserve_format, Preserve)
#undef AOTI_TORCH_MEMORY_FORMAT_IMPL

#define AOTI_TORCH_ITEM_IMPL(dtype, ctype)                     \
  AOTITorchError aoti_torch_item_##dtype(                      \
      AtenTensorHandle tensor, ctype* ret_value) {             \
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({               \
      at::Tensor* t = tensor_handle_to_tensor_pointer(tensor); \
      *ret_value = t->item().to<ctype>();                      \
    });                                                        \
  }

AOTI_TORCH_ITEM_IMPL(float16, c10::Half)
AOTI_TORCH_ITEM_IMPL(float32, float)
AOTI_TORCH_ITEM_IMPL(float64, double)
AOTI_TORCH_ITEM_IMPL(uint8, uint8_t)
AOTI_TORCH_ITEM_IMPL(uint16, uint16_t)
AOTI_TORCH_ITEM_IMPL(uint32, uint32_t)
AOTI_TORCH_ITEM_IMPL(uint64, uint64_t)
AOTI_TORCH_ITEM_IMPL(int8, int8_t)
AOTI_TORCH_ITEM_IMPL(int16, int16_t)
AOTI_TORCH_ITEM_IMPL(int32, int32_t)
AOTI_TORCH_ITEM_IMPL(int64, int64_t)
AOTI_TORCH_ITEM_IMPL(bool, bool)
AOTI_TORCH_ITEM_IMPL(bfloat16, c10::BFloat16)
AOTI_TORCH_ITEM_IMPL(complex64, c10::complex<float>)
#undef AOTI_TORCH_ITEM_IMPL

#define AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(dtype, ctype, ttype)                  \
  AOTITorchError aoti_torch_scalar_to_tensor_##dtype(                          \
      ctype value, AtenTensorHandle* ret_new_tensor) {                         \
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({                               \
      *ret_new_tensor =                                                        \
          new_tensor_handle(at::scalar_tensor(value, c10::ScalarType::ttype)); \
    });                                                                        \
  }

AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float32, float, Float)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(float64, double, Double)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint8, uint8_t, Byte)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint16, uint16_t, UInt16)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint32, uint32_t, UInt32)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(uint64, uint64_t, UInt64)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int8, int8_t, Char)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int16, int16_t, Short)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int32, int32_t, Int)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(int64, int64_t, Long)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(bool, bool, Bool)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(complex64, c10::complex<float>, ComplexFloat)
AOTI_TORCH_SCALAR_TO_TENSOR_IMPL(
    complex128,
    c10::complex<double>,
    ComplexDouble)
#undef AOTI_TORCH_SCALAR_TO_TENSOR_IMPL

#ifndef C10_MOBILE
#include <torch/version.h>
uint64_t aoti_torch_abi_version() {
  return TORCH_ABI_VERSION;
}
#endif // C10_MOBILE

bool aoti_torch_grad_mode_is_enabled() {
  return c10::GradMode::is_enabled();
}

void aoti_torch_grad_mode_set_enabled(bool enabled) {
  return c10::GradMode::set_enabled(enabled);
}

AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    delete t;
  });
}

AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor,
    void** ret_data_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    if (t->is_mkldnn()) {
      *ret_data_ptr = data_ptr_from_mkldnn(t);
    } else {
      *ret_data_ptr = t->data_ptr();
    }
  });
}

AOTITorchError aoti_torch_get_storage_size(
    AtenTensorHandle tensor,
    int64_t* ret_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_size = t->storage().nbytes();
  });
}

AOTITorchError aoti_torch_get_dim(AtenTensorHandle tensor, int64_t* ret_dim) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_dim = t->dim();
  });
}

AOTITorchError aoti_torch_get_numel(
    AtenTensorHandle tensor,
    int64_t* ret_numel) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_numel = t->numel();
  });
}

AOTITorchError aoti_torch_get_storage_numel(
    AtenTensorHandle tensor,
    int64_t* ret_numel) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    TORCH_INTERNAL_ASSERT(t->has_storage());
    auto dtype_size = t->dtype().itemsize();
    size_t nbytes = t->storage().nbytes();
    TORCH_INTERNAL_ASSERT(nbytes % dtype_size == 0);
    auto numel = nbytes / dtype_size;
    *ret_numel = numel;
  });
}

AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor,
    int64_t** ret_sizes) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    *ret_sizes = const_cast<int64_t*>(t->sizes().data());
  });
}

AOTITorchError aoti_torch_get_size(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_size = t->size(d);
  });
}

AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor,
    int64_t** ret_strides) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    *ret_strides = const_cast<int64_t*>(t->strides().data());
  });
}

AOTITorchError aoti_torch_get_stride(
    AtenTensorHandle tensor,
    int64_t d,
    int64_t* ret_stride) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_stride = t->stride(d);
  });
}

AOTITorchError aoti_torch_get_dtype(
    AtenTensorHandle tensor,
    int32_t* ret_dtype) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_dtype = static_cast<int32_t>(t->scalar_type());
  });
}

AOTITorchError aoti_torch_get_device_type(
    AtenTensorHandle tensor,
    int32_t* ret_device_type) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_device_type = static_cast<int32_t>(t->device().type());
  });
}

AOTITorchError aoti_torch_get_device_index(
    AtenTensorHandle tensor,
    int32_t* ret_device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_device_index = static_cast<int16_t>(t->device().index());
  });
}

AOTITorchError aoti_torch_get_storage_offset(
    AtenTensorHandle tensor,
    int64_t* ret_storage_offset) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    *ret_storage_offset = t->storage_offset();
  });
}

AOTITorchError aoti_torch_new_tensor_handle(
    AtenTensorHandle orig_handle,
    AtenTensorHandle* new_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(orig_handle);
    *new_handle = new_tensor_handle(at::Tensor(*t));
  });
}

AOTITorchError aoti_torch__reinterpret_tensor(
    AtenTensorHandle self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t offset_increment,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    *ret_new_tensor = new_tensor_handle(torch::inductor::_reinterpret_tensor(
        *self_tensor, sizes, strides, offset_increment));
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    if (c10::DeviceType(device_type) == c10::DeviceType::CPU) {
      *ret_new_tensor = new_tensor_handle(at::detail::empty_strided_cpu(
          sizes, strides, static_cast<c10::ScalarType>(dtype)));
    } else {
      c10::Device device = c10_device(device_type, device_index);
      c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
          static_cast<c10::ScalarType>(dtype));
      *ret_new_tensor =
          new_tensor_handle(at::empty_strided(sizes, strides, options));
    }
  });
}

AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    c10::Device device = c10_device(device_type, device_index);
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(dtype));
    *ret_new_tensor = new_tensor_handle(
        // data == nullptr can happen for a 0-size tensor
        (data != nullptr) ? at::for_blob(data, sizes)
                                .strides(strides)
                                .storage_offset(storage_offset)
                                .options(options)
                                .make_tensor()
                          : at::empty_strided(sizes, strides, options));
  });
}

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (layout == static_cast<int32_t>(at::kMkldnn)) {
      c10::IntArrayRef sizes(sizes_ptr, ndim);
      c10::IntArrayRef strides(strides_ptr, ndim);
      c10::Device device = c10_device(device_type, device_index);
      // get a mkldnn tensor wrapped by a torch Tensor(OpaqueTensorImpl),
      // which used by later mkldnn op.
      *ret_new_tensor = new_tensor_handle(mkldnn_tensor_from_data_ptr(
          data,
          sizes,
          static_cast<c10::ScalarType>(dtype),
          device,
          opaque_metadata,
          opaque_metadata_size));
    } else {
      aoti_torch_create_tensor_from_blob(
          data,
          ndim,
          sizes_ptr,
          strides_ptr,
          storage_offset,
          dtype,
          device_type,
          device_index,
          ret_new_tensor);
    }
  });
}

AOTITorchError aoti_torch__embedding_bag(
    AtenTensorHandle weight,
    AtenTensorHandle indices,
    AtenTensorHandle offsets,
    int32_t scale_grad_by_freq,
    int32_t mode,
    int32_t sparse,
    AtenTensorHandle per_sample_weights, // optional argument
    int32_t include_last_offset,
    int32_t padding_idx,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto [r0, r1, r2, r3] = at::_embedding_bag(
        *tensor_handle_to_tensor_pointer(weight),
        *tensor_handle_to_tensor_pointer(indices),
        *tensor_handle_to_tensor_pointer(offsets),
        scale_grad_by_freq,
        mode,
        sparse,
        pointer_to_optional(
            tensor_handle_to_tensor_pointer(per_sample_weights)),
        include_last_offset,
        padding_idx);

    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
    *ret2 = new_tensor_handle(std::move(r2));
    *ret3 = new_tensor_handle(std::move(r3));
  });
}

AOTITorchError aoti_torch__fft_c2c(
    AtenTensorHandle self,
    const int64_t* dim_ptr,
    int64_t dim_size,
    int64_t normalization,
    int32_t forward,
    AtenTensorHandle* ret // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto dim = c10::IntArrayRef(dim_ptr, dim_size);
    *ret = new_tensor_handle(at::_fft_c2c(
        *tensor_handle_to_tensor_pointer(self), dim, normalization, forward));
  });
}

AOTITorchError aoti_torch__scaled_dot_product_flash_attention_v2(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    int is_causal,
    int return_debug_mask,
    double* scale, // optional argument
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3, // returns new reference
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // returns new reference
    AtenTensorHandle* ret7, // returns new reference
    AtenTensorHandle* ret8 // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* query_tensor = tensor_handle_to_tensor_pointer(query);
    at::Tensor* key_tensor = tensor_handle_to_tensor_pointer(key);
    at::Tensor* value_tensor = tensor_handle_to_tensor_pointer(value);
    auto optional_scale = pointer_to_optional(scale);
    auto [r0, r1, r2, r3, r4, r5, r6, r7, r8] =
        at::_scaled_dot_product_flash_attention(
            *query_tensor,
            *key_tensor,
            *value_tensor,
            dropout_p,
            is_causal,
            return_debug_mask,
            optional_scale);

    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
    // ret2 and ret3 may be null
    if (ret2) {
      *ret2 = new_tensor_handle(std::move(r2));
    }
    if (ret3) {
      *ret3 = new_tensor_handle(std::move(r3));
    }
    *ret4 = r4.expect_int();
    *ret5 = r5.expect_int();
    *ret6 = new_tensor_handle(std::move(r6));
    *ret7 = new_tensor_handle(std::move(r7));
    *ret8 = new_tensor_handle(std::move(r8));
  });
}

AOTITorchError aoti_torch__scaled_dot_product_flash_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    double scale,
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3, // returns new reference
    int64_t* ret4,
    int64_t* ret5,
    AtenTensorHandle* ret6, // returns new reference
    AtenTensorHandle* ret7, // returns new reference
    AtenTensorHandle* ret8 // returns new reference
) {
  return aoti_torch__scaled_dot_product_flash_attention_v2(
      query,
      key,
      value,
      dropout_p,
      is_causal,
      return_debug_mask,
      &scale,
      ret0,
      ret1,
      ret2,
      ret3,
      ret4,
      ret5,
      ret6,
      ret7,
      ret8);
}

AOTITorchError aoti_torch__scaled_dot_product_efficient_attention(
    AtenTensorHandle query,
    AtenTensorHandle key,
    AtenTensorHandle value,
    AtenTensorHandle attn_bias, // optional argument
    int compute_log_sumexp,
    double dropout_p,
    int is_causal,
    double* scale, // optional argument
    AtenTensorHandle* ret0, // returns new reference
    AtenTensorHandle* ret1, // returns new reference
    AtenTensorHandle* ret2, // returns new reference
    AtenTensorHandle* ret3 // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* query_tensor = tensor_handle_to_tensor_pointer(query);
    at::Tensor* key_tensor = tensor_handle_to_tensor_pointer(key);
    at::Tensor* value_tensor = tensor_handle_to_tensor_pointer(value);
    auto optional_attn_bias =
        pointer_to_optional(tensor_handle_to_tensor_pointer(attn_bias));
    auto optional_scale = pointer_to_optional(scale);
    auto [r0, r1, r2, r3] = at::_scaled_dot_product_efficient_attention(
        *query_tensor,
        *key_tensor,
        *value_tensor,
        optional_attn_bias,
        compute_log_sumexp,
        dropout_p,
        is_causal,
        optional_scale);
    *ret0 = new_tensor_handle(std::move(r0));
    *ret1 = new_tensor_handle(std::move(r1));
    *ret2 = new_tensor_handle(std::move(r2));
    *ret3 = new_tensor_handle(std::move(r3));
  });
}

AOTITorchError aoti_torch_convolution(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias, // optional argument
    const int64_t* stride_ptr,
    int64_t stride_size,
    const int64_t* padding_ptr,
    int64_t padding_size,
    const int64_t* dilation_ptr,
    int64_t dilation_size,
    int transposed,
    const int64_t* output_padding_ptr,
    int64_t output_padding_size,
    int64_t groups,
    AtenTensorHandle* out // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* input_tensor = tensor_handle_to_tensor_pointer(input);
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);
    auto optional_bias = pointer_to_optional(bias_tensor);
    c10::IntArrayRef stride(stride_ptr, stride_size);
    c10::IntArrayRef padding(padding_ptr, padding_size);
    c10::IntArrayRef dilation(dilation_ptr, dilation_size);
    c10::IntArrayRef output_padding(output_padding_ptr, output_padding_size);

    *out = new_tensor_handle(at::convolution(
        *input_tensor,
        *weight_tensor,
        optional_bias,
        stride,
        padding,
        dilation,
        static_cast<bool>(transposed),
        output_padding,
        groups));
  });
}

AOTITorchError aoti_torch_new_uninitialized_tensor(AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = new at::Tensor();
    *ret = tensor_pointer_to_tensor_handle(out_tensor);
  });
}

AOTITorchError aoti_torch__scaled_mm(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle bias,
    int32_t* out_dtype,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle scale_result,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0,
    AtenTensorHandle* ret1) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);
    at::Tensor* scale_a_tensor = tensor_handle_to_tensor_pointer(scale_a);
    at::Tensor* scale_b_tensor = tensor_handle_to_tensor_pointer(scale_b);
    at::Tensor* scale_result_tensor =
        tensor_handle_to_tensor_pointer(scale_result);
    auto r0 = at::_scaled_mm(
        *self_tensor,
        *mat2_tensor,
        *scale_a_tensor,
        *scale_b_tensor,
        pointer_to_optional(bias_tensor),
        pointer_to_optional(scale_result_tensor),
        pointer_to_optional<c10::ScalarType>(out_dtype),
        use_fast_accum);
    *ret0 = new_tensor_handle(std::move(r0));
  });
}

AOTITorchError aoti_torch__scaled_mm_v2(
    AtenTensorHandle self,
    AtenTensorHandle mat2,
    AtenTensorHandle scale_a,
    AtenTensorHandle scale_b,
    AtenTensorHandle bias,
    AtenTensorHandle scale_result,
    int32_t* out_dtype,
    int8_t use_fast_accum,
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);
    at::Tensor* scale_a_tensor = tensor_handle_to_tensor_pointer(scale_a);
    at::Tensor* scale_b_tensor = tensor_handle_to_tensor_pointer(scale_b);
    at::Tensor* scale_result_tensor =
        tensor_handle_to_tensor_pointer(scale_result);
    auto r0 = at::_scaled_mm(
        *self_tensor,
        *mat2_tensor,
        *scale_a_tensor,
        *scale_b_tensor,
        pointer_to_optional(bias_tensor),
        pointer_to_optional(scale_result_tensor),
        pointer_to_optional<c10::ScalarType>(out_dtype),
        use_fast_accum);
    *ret0 = new_tensor_handle(std::move(r0));
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_tensor_copy_(
    AtenTensorHandle src,
    AtenTensorHandle dst) {
  return aoti_torch_copy_(dst, src, /*non_blocking=*/0);
}

AOTITorchError aoti_torch_assign_tensors(
    AtenTensorHandle src,
    AtenTensorHandle dst) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::Tensor* dst_tensor = tensor_handle_to_tensor_pointer(dst);
    *dst_tensor = *src_tensor;
  });
}

AOTITorchError aoti_torch_assign_tensors_out(
    AtenTensorHandle src,
    AtenTensorHandle* ret_dst) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* src_tensor_ptr = tensor_handle_to_tensor_pointer(src);
    at::Tensor dst_tensor = *src_tensor_ptr;
    *ret_dst = new_tensor_handle(std::move(dst_tensor));
  });
}

AOTITorchError aoti_torch_clone(AtenTensorHandle self, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *ret = new_tensor_handle(self_tensor->clone());
  });
}

AOTITorchError aoti_torch_as_strided(
    AtenTensorHandle self,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    int64_t ndim = self_tensor->dim();
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    at::Tensor ret_tensor = self_tensor->as_strided(sizes, strides);
    *ret = new_tensor_handle(std::move(ret_tensor));
  });
}

AOTITorchError aoti_torch_clone_preserve_strides(
    AtenTensorHandle self,
    AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // To mimic clone_preserve_strides which is used in copy_misaligned_inputs
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    int64_t needed_size = 1;
    for (int i = 0; i < self_tensor->dim(); i++) {
      if (self_tensor->size(i) == 0) {
        needed_size = 0;
        break;
      }
      needed_size += (self_tensor->size(i) - 1) * self_tensor->stride(i);
    }
    at::Tensor ret_tensor =
        self_tensor->as_strided({needed_size}, {1})
            .clone()
            .as_strided(self_tensor->sizes(), self_tensor->strides());
    *ret = new_tensor_handle(std::move(ret_tensor));
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    float beta,
    float alpha) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat1_tensor = tensor_handle_to_tensor_pointer(mat1);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::addmm_out(
        *out_tensor, *self_tensor, *mat1_tensor, *mat2_tensor, beta, alpha);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_bmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::bmm_out(*out_tensor, *self_tensor, *mat2_tensor);
  });
}

AOTITorchError aoti_torch_copy_(
    AtenTensorHandle self,
    AtenTensorHandle src,
    int32_t non_blocking) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    tensor_handle_to_tensor_pointer(self)->copy_(
        *tensor_handle_to_tensor_pointer(src), non_blocking);
  });
}

// TODO: implement a more efficient version instead of calling into aten
AOTITorchError aoti_torch_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat2) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* mat2_tensor = tensor_handle_to_tensor_pointer(mat2);
    at::mm_out(*out_tensor, *self_tensor, *mat2_tensor);
  });
}

AOTITorchError aoti_torch__mm_plus_mm_out(
    AtenTensorHandle out,
    AtenTensorHandle a,
    AtenTensorHandle b,
    AtenTensorHandle c,
    AtenTensorHandle d) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* a_tensor = tensor_handle_to_tensor_pointer(a);
    at::Tensor* b_tensor = tensor_handle_to_tensor_pointer(b);
    at::Tensor* c_tensor = tensor_handle_to_tensor_pointer(c);
    at::Tensor* d_tensor = tensor_handle_to_tensor_pointer(d);
    torch::inductor::_mm_plus_mm_out(
        *out_tensor, *a_tensor, *b_tensor, *c_tensor, *d_tensor);
  });
}

AOTITorchError aoti_torch_cpu_wrapped_fbgemm_pack_gemm_matrix_fp16(
    AtenTensorHandle weight,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);

    *out = new_tensor_handle(at::fbgemm_pack_gemm_matrix_fp16(*weight_tensor));
  });
}

AOTITorchError aoti_torch_cpu__wrapped_linear_prepack(
    AtenTensorHandle weight,
    AtenTensorHandle weight_scale,
    AtenTensorHandle weight_zero_point,
    AtenTensorHandle bias,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* weight_scale_tensor =
        tensor_handle_to_tensor_pointer(weight_scale);
    at::Tensor* weight_zero_point_tensor =
        tensor_handle_to_tensor_pointer(weight_zero_point);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);

    *out = new_tensor_handle(at::_wrapped_linear_prepack(
        *weight_tensor,
        *weight_scale_tensor,
        *weight_zero_point_tensor,
        *bias_tensor));
  });
}

AOTITorchError aoti_torch_cpu_wrapped_fbgemm_linear_fp16_weight(
    AtenTensorHandle input,
    AtenTensorHandle weight,
    AtenTensorHandle bias,
    int64_t out_channel,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* input_tensor = tensor_handle_to_tensor_pointer(input);
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* bias_tensor = tensor_handle_to_tensor_pointer(bias);

    *out = new_tensor_handle(at::fbgemm_linear_fp16_weight_fp32_activation(
        *input_tensor, *weight_tensor, *bias_tensor));
  });
}

AOTITorchError aoti_torch_cpu__wrapped_quantized_linear_prepacked(
    AtenTensorHandle input,
    AtenTensorHandle input_scale,
    AtenTensorHandle input_zero_point,
    AtenTensorHandle weight,
    AtenTensorHandle out_scale,
    AtenTensorHandle out_zeropoint,
    int64_t out_channel,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* input_tensor = tensor_handle_to_tensor_pointer(input);
    at::Tensor* input_scale_tensor =
        tensor_handle_to_tensor_pointer(input_scale);
    at::Tensor* input_zero_point_tensor =
        tensor_handle_to_tensor_pointer(input_zero_point);
    at::Tensor* weight_tensor = tensor_handle_to_tensor_pointer(weight);
    at::Tensor* out_scale_tensor = tensor_handle_to_tensor_pointer(out_scale);
    at::Tensor* out_zeropoint_tensor =
        tensor_handle_to_tensor_pointer(out_zeropoint);
    *out = new_tensor_handle(at::_wrapped_quantized_linear_prepacked(
        *input_tensor,
        *input_scale_tensor,
        *input_zero_point_tensor,
        *weight_tensor,
        *out_scale_tensor,
        *out_zeropoint_tensor,
        out_channel));
  });
}

AOTITorchError aoti_torch_nonzero(
    AtenTensorHandle self,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *out = new_tensor_handle(at::nonzero(*self_tensor));
  });
}

AOTITorchError aoti_torch_repeat_interleave_Tensor(
    AtenTensorHandle repeats,
    int64_t* output_size,
    AtenTensorHandle* out) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* repeats_tensor = tensor_handle_to_tensor_pointer(repeats);
    *out = new_tensor_handle(at::_ops::repeat_interleave_Tensor::call(
        *repeats_tensor, pointer_to_optional<c10::SymInt>(output_size)));
  });
}

// Function to check existence of inf and NaN
AOTITorchError aoti_torch_check_inf_and_nan(
    const char* tensor_name,
    AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* check_tensor = tensor_handle_to_tensor_pointer(tensor);

    assert_inf_and_nan(tensor_name, *check_tensor);
  });
}

AOTITorchError aoti_torch_scatter_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* index_tensor = tensor_handle_to_tensor_pointer(index);
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::scatter_out(*out_tensor, *self_tensor, dim, *index_tensor, *src_tensor);
  });
}

AOTITorchError aoti_torch_scatter_reduce_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    int64_t dim,
    AtenTensorHandle index,
    AtenTensorHandle src,
    const char* reduce,
    int32_t include_self) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* index_tensor = tensor_handle_to_tensor_pointer(index);
    at::Tensor* src_tensor = tensor_handle_to_tensor_pointer(src);
    at::scatter_reduce_out(
        *out_tensor,
        *self_tensor,
        dim,
        *index_tensor,
        *src_tensor,
        reduce,
        (bool)include_self);
  });
}

AOTITorchError aoti_torch_index_put_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    const AtenTensorHandle* indices,
    const uint32_t num_indices,
    // NOLINTNEXTLINE(misc-misplaced-const)
    const AtenTensorHandle values,
    bool accumulate) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::List<std::optional<at::Tensor>> indices_;
    indices_.reserve(num_indices);
    for (size_t i = 0; i < num_indices; i++) {
      indices_.emplace_back(
          pointer_to_optional(tensor_handle_to_tensor_pointer(indices[i])));
    }
    at::Tensor* out_tensor = tensor_handle_to_tensor_pointer(out);
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    at::Tensor* values_tensor = tensor_handle_to_tensor_pointer(values);
    at::index_put_out(
        *out_tensor, *self_tensor, indices_, *values_tensor, accumulate);
  });
}

AOTITorchError aoti_torch_view_as_real(
    AtenTensorHandle self,
    AtenTensorHandle* ret // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(
        at::_ops::view_as_real::call(*tensor_handle_to_tensor_pointer(self)));
  });
}

AOTITorchError aoti_torch_view_dtype(
    AtenTensorHandle self,
    int32_t dtype,
    AtenTensorHandle* ret // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    *ret = new_tensor_handle(at::_ops::view_dtype::call(
        *self_tensor, static_cast<c10::ScalarType>(dtype)));
  });
}

void aoti_torch_save_tensor_handle(
    AtenTensorHandle self,
    const char* tensor_name,
    const char* launch_prefix,
    const char* kernel_name) {
  at::Tensor* t = tensor_handle_to_tensor_pointer(self);
#ifndef C10_MOBILE
  // Save tensor to tmp .pt file for tensors and can be torch.load'ed later
  std::string cwd = get_current_path();
  std::string tmp_folder = cwd + "/tmp/aoti_torch/";
  if (!file_exists(tmp_folder)) {
    std::cout
        << "aoti_torch_save_tensor_handle: Path does not exist, creating it..."
        << tmp_folder << '\n';

    if (!create_directories(tmp_folder)) {
      std::cout << "aoti_torch_save_tensor_handle: Error creating directory: "
                << tmp_folder << '\n';
      return;
    }
  }
  std::string tensor_filepath_to_save = tmp_folder + launch_prefix + "_" +
      kernel_name + "_" + tensor_name + "_" + t->device().str() + ".pt";

  auto bytes = torch::jit::pickle_save(c10::IValue(*t));
  std::ofstream fout(tensor_filepath_to_save, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  fout.close();

  std::cout << "aoti_torch_save_tensor_handle: Saved tensor to "
            << tensor_filepath_to_save << '\n';
#endif // !defined(C10_MOBILE)
}

void aoti_torch_print_tensor_handle(AtenTensorHandle self, const char* msg) {
  at::Tensor* t = tensor_handle_to_tensor_pointer(self);

  // Display message
  std::cout << "[";
  if (msg) {
    std::cout << "  " << msg;
  }
  std::cout << "  "
            << "]:" << '\n';

  // Print exact tensor values for small size tensors
  const int64_t numel = t->numel();
  if (numel <= AOTI_TORCH_MAX_NUMEL_TO_PRINT) {
    std::cout << *t << "\n";
  }

  // Print summary stats of the tensor
  std::cout << "Number of elements: " << numel << '\n';

  // Print dtypes and for float types, print exact precision
  auto scalarType = t->scalar_type();
  if (scalarType == at::ScalarType::Float) {
    std::cout << "Dtype: float32" << std::endl;
  } else if (scalarType == at::ScalarType::Half) {
    std::cout << "Dtype: float16" << std::endl;
  } else if (scalarType == at::ScalarType::BFloat16) {
    std::cout << "Dtype: bfloat16" << std::endl;
  } else {
    std::cout << "Dtype: " << t->dtype() << '\n';
  }

  if (numel > 0) {
    // torch/aten `mean()` function only supports float and complex dtypes
    // See:
    // https://github.com/pytorch/pytorch/blob/a0e062c6f1a03ec93e87413e42c4d0b336518131/aten/src/ATen/native/ReduceOps.cpp#L304-L309
    auto mean_value = [t](at::ScalarType dtype) {
      return t->to(dtype).mean().item();
    };
    bool is_complex_type =
        at::isComplexType(at::typeMetaToScalarType(t->dtype()));
    at::ScalarType float_dtype =
        is_complex_type ? at::kComplexFloat : at::kFloat;
    std::cout << "Mean value: " << mean_value(float_dtype) << '\n';
    if (!is_complex_type) {
      // "min_all_cuda" function is not implemented for 'ComplexFloat' type.
      // (similar for max) Skip printing min/max value for complex type tensors
      // here If encountered complex dtypes (rare occasions), suggest to print
      // out the whole value of the tensor.
      std::cout << "Min value: " << t->min().item<float>() << '\n';
      std::cout << "Max value: " << t->max().item<float>() << '\n';
    }
  }
  std::cout << "Device: " << t->device() << '\n';
  std::cout << "Size: " << t->sizes() << '\n';
  std::cout << "Stride: " << t->strides() << '\n';
  std::cout << "Layout: " << t->layout() << '\n';
  std::cout << "Is contiguous: " << t->is_contiguous() << '\n';
  std::cout << "Requires grad: " << t->requires_grad() << '\n';

  std::cout << '\n';
}

// ProxyExecutor
AOTITorchError aoti_torch_proxy_executor_call_function(
    AOTIProxyExecutorHandle proxy_executor,
    int extern_node_index,
    int num_ints,
    int64_t* flatten_int_args,
    int num_tensors,
    AtenTensorHandle* flatten_tensor_args) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (!proxy_executor) {
      throw std::runtime_error(
          "Unable to find a proxy executor to run custom ops. Please check if "
          "there is a json file generated in the same directory as the so, or use "
          "torch._inductor.aoti_compile_and_package to package everything into a "
          "PT2 artifact.");
    }
    ProxyExecutor* executor = reinterpret_cast<ProxyExecutor*>(proxy_executor);
    executor->call_function(
        extern_node_index,
        num_ints,
        flatten_int_args,
        num_tensors,
        flatten_tensor_args);
  });
}

void aoti_torch_check(
    bool cond,
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  if (C10_UNLIKELY_OR_CONST(!cond)) {
    ::c10::detail::torchCheckFail(func, file, line, msg);
  }
}

void aoti_torch_warn(
    const char* func,
    const char* file,
    uint32_t line,
    const char* msg) {
  ::c10::warn(::c10::Warning(
      ::c10::UserWarning(),
      {func, file, static_cast<uint32_t>(line)},
      msg,
      false));
}

AOTITorchError aoti_torch__alloc_from_pool(
    AtenTensorHandle self,
    int64_t offset_bytes,
    int32_t dtype,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    AtenTensorHandle* ret_new_tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* self_tensor = tensor_handle_to_tensor_pointer(self);
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    *ret_new_tensor = new_tensor_handle(torch::inductor::_alloc_from_pool(
        *self_tensor,
        offset_bytes,
        static_cast<c10::ScalarType>(dtype),
        sizes,
        strides));
  });
}

AOTITorchError aoti_torch_zero_(AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t = tensor_handle_to_tensor_pointer(tensor);
    t->zero_();
  });
}

static StableIValue from_ivalue(
    const c10::TypePtr& type,
    const c10::IValue& ivalue) {
  switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      AtenTensorHandle ath = torch::aot_inductor::new_tensor_handle(
          std::move(const_cast<at::Tensor&>(ivalue.toTensor())));
      return from(ath);
    }
    case c10::TypeKind::IntType: {
      return from(ivalue.toInt());
    }
    case c10::TypeKind::FloatType: {
      return from(ivalue.toDouble());
    }
    case c10::TypeKind::BoolType: {
      return from(ivalue.toBool());
    }
    case c10::TypeKind::ScalarTypeType: {
      return from(ivalue.toScalarType());
    }
    case c10::TypeKind::DeviceObjType: {
      return from(ivalue.toDevice());
    }
    case c10::TypeKind::LayoutType: {
      return from(ivalue.toLayout());
    }
    case c10::TypeKind::MemoryFormatType: {
      return from(ivalue.toMemoryFormat());
    }
    case c10::TypeKind::OptionalType: {
      auto inner_type = type->castRaw<at::OptionalType>()->getElementType();

      // ideally, if we had the C++ type corresponding to inner_type, which we
      // will denote as inner_type::t (does not actually exist), we would be
      // able to follow the patterned semantic of every other case here in one
      // line:
      //
      // return from<std::optional<inner_type::t>>(ivalue.toInnerTypeT()));
      //
      // BUT we do NOT have that type inner_type::t readily available, so we
      // will manually unwrap and recursively call. This implementation MUST
      // be kept in sync with from<std::optional<T>> function in
      // torch/csrc/stable/library.h
      if (ivalue.isNone()) {
        return from(std::nullopt);
      }
      StableIValue* sivp = new StableIValue(from_ivalue(inner_type, ivalue));
      return from(sivp);
    }
    default: {
      TORCH_CHECK(
          false,
          "Not yet supported conversion from IValue to StableIValue for schema type: ",
          type->str());
    }
  }
}

static c10::IValue to_ivalue(
    const c10::TypePtr& type,
    const StableIValue stable_ivalue) {
  switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      auto ret_raiiath = torch::aot_inductor::RAIIAtenTensorHandle(
          to<AtenTensorHandle>(stable_ivalue));
      return (c10::IValue(*torch::aot_inductor::tensor_handle_to_tensor_pointer(
          ret_raiiath.get())));
    }
    case c10::TypeKind::IntType: {
      return c10::IValue(to<int64_t>(stable_ivalue));
    }
    case c10::TypeKind::FloatType: {
      return c10::IValue(to<double>(stable_ivalue));
    }
    case c10::TypeKind::BoolType: {
      return c10::IValue(to<bool>(stable_ivalue));
    }
    case c10::TypeKind::ScalarTypeType: {
      return c10::IValue(to<c10::ScalarType>(stable_ivalue));
    }
    case c10::TypeKind::DeviceObjType: {
      return c10::IValue(to<c10::Device>(stable_ivalue));
    }
    case c10::TypeKind::LayoutType: {
      return c10::IValue(to<c10::Layout>(stable_ivalue));
    }
    case c10::TypeKind::MemoryFormatType: {
      return c10::IValue(to<c10::MemoryFormat>(stable_ivalue));
    }
    case c10::TypeKind::OptionalType: {
      auto inner_type = type->castRaw<at::OptionalType>()->getElementType();

      // ideally, if we had the C++ type corresponding to inner_type, which we
      // will denote as inner_type::t (does not actually exist), we would be
      // able to follow the patterned semantic of every other case here in one
      // line:
      //
      // return c10::IValue(to<std::optional<inner_type::t>>(stable_ivalue));
      //
      // BUT we do NOT have that type inner_type::t readily available, so we
      // will manually unwrap and recursively call. This implementation MUST
      // be kept in sync with the to<T> function in
      // torch/csrc/stable/library.h
      if (stable_ivalue == from(std::nullopt)) {
        return c10::IValue();
      }
      auto sivp = to<StableIValue*>(stable_ivalue);
      auto ival = to_ivalue(inner_type, *sivp);
      delete sivp;
      return ival;
    }
    default: {
      TORCH_CHECK(
          false,
          "Not yet supported conversion from StableIValue to IValue for schema type: ",
          type->str());
    }
  }
}

class StableIValueBoxedKernel : public c10::OperatorKernel {
 public:
  StableIValueBoxedKernel(void (*fn)(StableIValue*, uint64_t, uint64_t))
      : fn_(fn) {}

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    const auto num_returns = schema.returns().size();
    const auto num_arguments = schema.arguments().size();

    auto ministack =
        std::make_unique<StableIValue[]>(std::max(num_arguments, num_returns));

    for (const auto idx : c10::irange(num_arguments)) {
      const auto ministack_idx = num_arguments - idx - 1;
      const c10::TypePtr& arg_type = schema.arguments()[ministack_idx].type();
      ministack[ministack_idx] = from_ivalue(arg_type, torch::jit::pop(stack));
    }

    // boxed function is going to take a stack of StableIValues, cast them to
    // our schema values, and run the function and modify the StableIValue stack
    fn_(ministack.get(), num_arguments, num_returns);

    // read the output from the end of the stack and wrap that back into
    // IValue from StableIValue
    for (size_t idx = 0; idx < num_returns; idx++) {
      const c10::TypePtr& ret_type = schema.returns()[idx].type();
      torch::jit::push(stack, to_ivalue(ret_type, ministack[idx]));
    }
  }

 private:
  void (*fn_)(StableIValue*, uint64_t, uint64_t);
};

AOTITorchError aoti_torch_library_init_impl(
    const char* ns,
    const char* k,
    const char* file,
    uint32_t line,
    TorchLibraryHandle* ret_new_torch_lib) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret_new_torch_lib =
        reinterpret_cast<TorchLibraryOpaque*>(new torch::Library(
            torch::Library::Kind::IMPL,
            std::string(ns),
            c10::parseDispatchKey(std::string(k)),
            file,
            line));
  });
}

AOTITorchError aoti_torch_library_init_def(
    const char* ns,
    const char* file,
    uint32_t line,
    TorchLibraryHandle* ret_new_torch_lib) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret_new_torch_lib =
        reinterpret_cast<TorchLibraryOpaque*>(new torch::Library(
            torch::Library::Kind::DEF,
            std::string(ns),
            std::nullopt,
            file,
            line));
  });
}

AOTITorchError aoti_torch_library_init_fragment(
    const char* ns,
    const char* file,
    uint32_t line,
    TorchLibraryHandle* ret_new_torch_lib) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret_new_torch_lib =
        reinterpret_cast<TorchLibraryOpaque*>(new torch::Library(
            torch::Library::Kind::FRAGMENT,
            std::string(ns),
            std::nullopt,
            file,
            line));
  });
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t)) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<torch::Library*>(self)->impl(
        name,
        torch::CppFunction::makeFromBoxedFunctor(
            std::make_unique<StableIValueBoxedKernel>(fn)));
  });
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_library_def(TorchLibraryHandle self, const char* name) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { reinterpret_cast<torch::Library*>(self)->def(name); });
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_library_object(TorchLibraryHandle tlh) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<torch::Library*>(tlh); });
}

AOTITorchError aoti_torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto op =
        c10::Dispatcher::singleton().findSchemaOrThrow(opName, overloadName);
    const auto& schema = op.schema();
    const auto num_returns = schema.returns().size();
    const auto num_arguments = schema.arguments().size();

    torch::jit::Stack ivalue_stack;
    // we will only need max(num_args, num_returns)
    ivalue_stack.reserve(std::max(num_arguments, num_returns));

    // convert StableIValue stack to c10::IValue stack
    for (const auto idx : c10::irange(num_arguments)) {
      auto stable_ivalue = stack[idx];
      auto arg_type = schema.arguments()[idx].type();
      torch::jit::push(ivalue_stack, to_ivalue(arg_type, stable_ivalue));
    }

    op.callBoxed(ivalue_stack);

    // there should then be num_returns IValues on the stack, which
    // we will convert to StableIValue and repopulate user input stack
    for (const auto idx : c10::irange(num_returns)) {
      const auto stack_idx = num_returns - idx - 1;
      const c10::TypePtr& ret_type = schema.returns()[idx].type();
      stack[stack_idx] = from_ivalue(ret_type, torch::jit::pop(ivalue_stack));
    }
  });
}
