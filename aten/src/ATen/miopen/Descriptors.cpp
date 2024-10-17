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
  throw std::runtime_error("TensorDescriptor only supports float, half and bfloat16 tensors");
  }
}

} // anonymous namespace


void TensorDescriptor::set(const at::Tensor &t, size_t pad) {
  set(getDataType(t), t.sizes(), t.strides(), pad);
}

constexpr size_t MIOPEN_DIM_MAX = 5;

void TensorDescriptor::set(miopenDataType_t datatype, IntArrayRef t_sizes, IntArrayRef t_strides, size_t pad) {
  size_t dim = t_sizes.size();
  if (dim > MIOPEN_DIM_MAX || pad > MIOPEN_DIM_MAX)
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  int size[MIOPEN_DIM_MAX];
  int stride[MIOPEN_DIM_MAX];
  for (const auto i : c10::irange(dim)) {
    size[i] = static_cast<int>(t_sizes[i]);
    stride[i] = static_cast<int>(t_strides[i]);
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = 1;
    stride[i] = 1;
  }
  set(datatype, static_cast<int>(std::max(dim, pad)), size, stride);
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
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}

std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims = 4;
  int dimA[MIOPEN_DIM_MAX];
  int strideA[MIOPEN_DIM_MAX];
  miopenDataType_t dtype;
  miopenGetTensorDescriptor(d.desc(), &dtype, dimA, strideA);
  out << "    type = " << miopenTypeToString(dtype) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void TensorDescriptor::print() { std::cout << *this; }

void FilterDescriptor::set(const at::Tensor &t, const at::MemoryFormat memory_format, int64_t pad) {
  auto dim = t.ndimension();
  if (dim > static_cast<int64_t>(MIOPEN_DIM_MAX) || pad > static_cast<int64_t>(MIOPEN_DIM_MAX)) {
#define _STR(X) #X
#define STR(X) _STR(X)
    throw std::runtime_error("MIOpen supports only up to " STR(MIOPEN_DIM_MAX) " dimensions");
#undef _STR
#undef STR
  }
  TORCH_CHECK(t.is_contiguous(memory_format),
      "MIOpen filters (a.k.a. weights) must be contiguous");

  int size[MIOPEN_DIM_MAX];
  int stride[MIOPEN_DIM_MAX];
  for (const auto i : c10::irange(dim)) {
    size[i] = (int) t.size(i);
  }
  for (const auto i : c10::irange(dim, pad)) {
    size[i] = (int) 1;
  }

  for (int i = pad; i >= dim; --i ) {
      stride[i] = 1;
  }
  for (int i = dim-1 ; i >=0; --i ) {
      // Pass-through
      stride[i] = t.stride(i);
  }

  dim = std::max<int64_t>(dim, pad);
  set(getDataType(t), (int) dim, size, stride);
}

}}
