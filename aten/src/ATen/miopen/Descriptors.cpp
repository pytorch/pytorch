#include <ATen/miopen/Descriptors.h>
#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <iostream>

namespace at { namespace native {

namespace {

inline miopenDataType_t getDataType(const at::Tensor& t) {
  auto scalar_type = t.scalar_type();
  if (scalar_type == at::kFloat) {
    return miopenFloat;
  } else if (scalar_type == at::kHalf) {
    return miopenHalf;
  } else if (scalar_type == at::kBFloat16) {
    return miopenBFloat16;
  } else {
    TORCH_CHECK(
        false,
        "TensorDescriptor does not support ", scalar_type);
  }
}

} // anonymous namespace

constexpr size_t MIOPEN_DIM_MAX = 5;

void TensorDescriptor::set(const at::Tensor &t, at::MemoryFormat memory_format, size_t pad) {
  set(getDataType(t), t.sizes(), t.strides(), pad,
    memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d);
}

void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  auto memory_format = t.suggest_memory_format();
  set(getDataType(t), t.sizes(), t.strides(), pad,
    memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d);
}

void TensorDescriptor::set(miopenDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad) {
  set(datatype, t_sizes, t_strides, pad,
    is_channels_last_strides_2d(t_sizes, t_strides) ||
    is_channels_last_strides_3d(t_sizes, t_strides));
}

void TensorDescriptor::set(miopenDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad, bool nhwc) {
  size_t dim = t_sizes.size();
  if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
    TORCH_CHECK(false, "MIOpen supports only up to ", MIOPEN_DIM_MAX, " dimensions");
  // Use size_t (64-bit) to support large tensors with numel > int32_max
  size_t size[MIOPEN_DIM_MAX];
  size_t stride[MIOPEN_DIM_MAX];
  for (const auto i : c10::irange(dim)) {
    size[i] = static_cast<size_t>(t_sizes[i]);
    stride[i] = static_cast<size_t>(t_strides[i]);
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = 1;
    stride[i] = 1;
  }
  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride, nhwc);
}

std::string miopenTypeToString(miopenDataType_t dtype) {
  switch (dtype) {
    case miopenFloat:
      return "miopenFloat";
    case miopenHalf:
      return "miopenHalf";
    case miopenBFloat16:
      return "miopenBFloat16";
    default:
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ')';
      return oss.str();
  }
}

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << '\n';
  int nbDims = 0;
  int dimA[MIOPEN_DIM_MAX];
  int strideA[MIOPEN_DIM_MAX];
  miopenDataType_t dtype;
  miopenGetTensorDescriptorSize(d.desc(), &nbDims);
  miopenGetTensorDescriptor(d.desc(), &dtype, dimA, strideA);
  out << "    type = " << miopenTypeToString(dtype) << '\n';
  out << "    nbDims = " << nbDims << '\n';
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << '\n';
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << '\n';
  return out;
}

void TensorDescriptor::print() { std::cout << *this; }

void FilterDescriptor::set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad) {
  auto dim = t.ndimension();
  if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
  TORCH_CHECK(false, "MIOpen supports only up to ", MIOPEN_DIM_MAX, " dimensions");
  // NB: It is possible for this test to be insufficient, because the
  // Tensor passed in to set the filter descriptor may not be the actual
  // Tensor whose data pointer is passed to cuDNN.  Nevertheless,
  // that is the common case, so we can catch most client errors with this test.
  TORCH_CHECK(t.is_contiguous(memory_format),
    "MIOpen filters (a.k.a. weights) must be contiguous in desired memory_format\n",
    "Weight sizes: ", t.sizes(), '\n',
    "Weight strides: ", t.strides(), '\n',
    "cuDNN suggested memory_format: ", memory_format);

  // Use size_t (64-bit) to support large tensors
  size_t size[MIOPEN_DIM_MAX];
  size_t stride[MIOPEN_DIM_MAX];
  for (const auto i : c10::irange(dim)) {
    size[i] = static_cast<size_t>(t.size(i));
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = 1;
  }

  for (int i = pad; i >= dim; --i ) {
      stride[i] = 1;
  }
  for (int i = dim-1 ; i >=0; --i ) {
      // Pass-through
      stride[i] = static_cast<size_t>(t.stride(i));
  }

  dim = std::max<int64_t>(dim, pad);
  set(getDataType(t), static_cast<int>(dim), size, stride,
    memory_format == at::MemoryFormat::ChannelsLast ||
    memory_format == at::MemoryFormat::ChannelsLast3d);
}

}}
