#include <ATen/native/mkldnn/LlgaTensorImpl.h>
#include <c10/core/CPUAllocator.h>

namespace at {

LlgaTensorImpl::LlgaTensorImpl(
    Storage&& storage,
    const caffe2::TypeMeta& data_type,
    const LlgaTensorDesc& desc)
    : TensorImpl(
          std::move(storage),
          c10::DispatchKeySet(DispatchKey::MkldnnCPU),
          data_type),
      desc_(desc) {}

// The following are publically exposed as methods of Tensor
IntArrayRef LlgaTensorImpl::strides() const {
  TORCH_CHECK(false, "Cannot get strides of LlgaTensorImpl");
}
int64_t LlgaTensorImpl::stride(int64_t d) const {
  TORCH_CHECK(false, "Cannot get strides of LlgaTensorImpl");
}
bool LlgaTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  TORCH_CHECK(false, "Cannot get query is_contiguous on LlgaTensorImpl");
}
const Storage& LlgaTensorImpl::storage() const {
  TORCH_CHECK(false, "Cannot access the storage() of LlgaTensorImpl");
}
int64_t LlgaTensorImpl::storage_offset() const {
  TORCH_CHECK(false, "Cannot access the storage_offset() of LlgaTensorImpl");
}

// The following are some internal inherited methods that we do not support.
// They should never get called.
void LlgaTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_INTERNAL_ASSERT(false, "Cannot set_size for LlgaTensorImpl");
}
void LlgaTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_INTERNAL_ASSERT(false, "Cannot set_stride for LlgaTensorImpl");
}
void LlgaTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_INTERNAL_ASSERT(false, "Cannot set_storage_offset for LlgaTensorImpl");
}
bool LlgaTensorImpl::has_storage() const {
  return true;
}

Tensor empty_llga(const LlgaTensorDesc& desc, const TensorOptions& options) {
  auto sizes = desc.sizes();
  auto nbytes = desc.storage_size();

  auto allocator = at::GetCPUAllocator();
  auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      nbytes,
      allocator->allocate(nbytes),
      allocator,
      /*resizable=*/false);

  return at::detail::make_tensor<LlgaTensorImpl>(
      std::move(storage_impl), options.dtype(), desc);
}

const LlgaTensorDesc& get_llga_desc(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_mkldnn(), "get_llga_desc expects Mkldnn tensor input");
  return static_cast<LlgaTensorImpl*>(tensor.unsafeGetTensorImpl())->desc();
}

dnnl::graph::tensor llga_from_aten_tensor(const Tensor& tensor) {
  return {get_llga_desc(tensor).logical_tensor(), tensor.data_ptr()};
}

using data_type = dnnl::graph::logical_tensor::data_type;

data_type getLlgaDataType(at::ScalarType dt) {
  switch (dt) {
    case at::ScalarType::Float:
      return data_type::f32;
    case at::ScalarType::BFloat16:
      return data_type::bf16;
    case at::kInt:
      return data_type::s32;
    case at::ScalarType::QInt8:
      return data_type::s8;
    case at::ScalarType::QUInt8:
      return data_type::u8;
    default:
      TORCH_CHECK(false, "Not support data type ", dt);
  }
}

LlgaTensorDesc LlgaTensorDesc::supplementTensorInfo(const at::Tensor& t) const {
  if (t.is_mkldnn()) {
    // if input tensor is of mkldnn, it's originated from an upstream
    // LLGA partition which carries opaque layout info
    return get_llga_desc(t).tid(tid_);
  } else {
    // if input tensor is not an mkldnn tensor, use default layout
    auto sizes = t.sizes().vec();
    auto strides = t.strides().vec();
    auto dtype = getLlgaDataType(t.scalar_type());
    return {tid_, sizes, strides, dtype};
  }
}

at::ScalarType LlgaTensorDesc::aten_scalar_type() const {
  switch (dtype_) {
    case data_type::f32:
      return at::ScalarType::Float;
    case data_type::bf16:
      return at::ScalarType::BFloat16;
    case data_type::s32:
      return at::kInt;
    case data_type::s8:
      return at::ScalarType::QInt8;
    case data_type::u8:
      return at::ScalarType::QUInt8;
    default:
      TORCH_CHECK(false, "Invalid data type ", static_cast<size_t>(dtype_));
  }
}

} // namespace at
