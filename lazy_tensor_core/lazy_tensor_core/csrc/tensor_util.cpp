#include "lazy_tensor_core/csrc/tensor_util.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <list>
#include <numeric>
#include <thread>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/layout_manager.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace {

bool ShouldUseBF16() {
  bool use_bf16 = lazy_tensors::sys_util::GetEnvBool("LTC_USE_BF16", false);
  if (use_bf16) {
    LOG(INFO) << "Using BF16 data type for floating point values";
  }
  return use_bf16;
}

bool ShouldUseF16() {
  bool use_fp16 = lazy_tensors::sys_util::GetEnvBool("LTC_USE_FP16", false);
  if (use_fp16) {
    LOG(INFO) << "Using F16 data type for floating point values";
  }
  return use_fp16;
}

bool ShouldDowncastToBF16() {
  bool downcast_bf16 =
      lazy_tensors::sys_util::GetEnvBool("LTC_DOWNCAST_BF16", false);
  if (downcast_bf16) {
    LOG(INFO) << "Downcasting floating point values, F64->F32, F32->BF16";
  }
  return downcast_bf16;
}

bool ShouldDowncastToF16() {
  bool downcast_fp16 =
      lazy_tensors::sys_util::GetEnvBool("LTC_DOWNCAST_FP16", false);
  if (downcast_fp16) {
    LOG(INFO) << "Downcasting floating point values, F64->F32, F32->FP16";
  }
  return downcast_fp16;
}

bool ShouldUse32BitLong() {
  bool use_32bit_long =
      lazy_tensors::sys_util::GetEnvBool("LTC_USE_32BIT_LONG", false);
  if (use_32bit_long) {
    LOG(INFO) << "Using 32bit integers for kLong values";
  }
  return use_32bit_long;
}

bool UseBF16() {
  static bool use_bf16 = ShouldUseBF16();
  return use_bf16;
}

bool UseF16() {
  static bool use_fp16 = ShouldUseF16();
  return use_fp16;
}

bool DowncastBF16() {
  static bool downcast_bf16 = ShouldDowncastToBF16();
  return downcast_bf16;
}

bool DowncastF16() {
  static bool downcast_fp16 = ShouldDowncastToF16();
  return downcast_fp16;
}

bool Use32BitLong() {
  static bool use_32bit_long = ShouldUse32BitLong();
  return use_32bit_long;
}

template <typename S>
struct Caster {
  template <typename D>
  D cast(const S& value) const {
    return static_cast<D>(value);
  }
};
template <>
struct Caster<at::BFloat16> {
  template <typename D>
  D cast(const at::BFloat16& value) const {
    return static_cast<D>(static_cast<float>(value));
  }
};
template <>
struct Caster<c10::Half> {
  template <typename D>
  D cast(const c10::Half& value) const {
    return static_cast<D>(static_cast<float>(value));
  }
};
template <>
struct Caster<c10::complex<float>> {
  template <typename D>
  D cast(const c10::complex<float>& value) const {
    return static_cast<D>(value.real());
  }
};

template <>
c10::complex<float> Caster<c10::complex<float>>::cast<c10::complex<float>>(
    const c10::complex<float>& value) const {
  return c10::complex<float>(value.real(), value.imag());
}

template <>
c10::complex<double> Caster<c10::complex<float>>::cast<c10::complex<double>>(
    const c10::complex<float>& value) const {
  return c10::complex<double>(value.real(), value.imag());
}

template <>
struct Caster<c10::complex<double>> {
  template <typename D>
  D cast(const c10::complex<double>& value) const {
    return static_cast<D>(value.real());
  }
};

template <>
c10::complex<float> Caster<c10::complex<double>>::cast<c10::complex<float>>(
    const c10::complex<double>& value) const {
  return c10::complex<float>(value.real(), value.imag());
}

template <>
c10::complex<double> Caster<c10::complex<double>>::cast<c10::complex<double>>(
    const c10::complex<double>& value) const {
  return c10::complex<double>(value.real(), value.imag());
}

// Copies n bytes from source to dest, with different stride values for source
// and destination.
template <typename S, typename D>
void StridedCopy(D* dest, int64_t dest_stride, const S* source,
                 int64_t source_stride, int64_t n) {
  Caster<S> caster;
  const S* source_top = source + n * source_stride;
  for (; source < source_top; dest += dest_stride, source += source_stride) {
    *dest = caster.template cast<D>(*source);
  }
}

// Computes the offset of the value at a given index, assuming a contiguous/flat
// tensor data representation.
template <typename S>
int64_t GetFlatTensorOffset(const S& strides,
                            const std::vector<int64_t>& indices) {
  int64_t base = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    base += indices[i] * strides[i];
  }
  return base;
}

// The c10::BFloat16 does not have implicit cast operations, so using
// std::copy() for it, is not going to work.
struct CopyDirect {};
struct CopyCasted {};

template <typename T>
struct NeedCast {
  static constexpr bool value = false;
};
template <>
struct NeedCast<c10::BFloat16> {
  static constexpr bool value = true;
};
template <>
struct NeedCast<c10::Half> {
  static constexpr bool value = true;
};
template <>
struct NeedCast<c10::complex<float>> {
  static constexpr bool value = true;
};
template <>
struct NeedCast<c10::complex<double>> {
  static constexpr bool value = true;
};

template <bool CAST>
struct CopyType {
  using type = CopyDirect;
};
template <>
struct CopyType<true> {
  using type = CopyCasted;
};

template <typename D, typename S>
void CheckedMemcpy(D* dest, const S* source, int64_t n) {
  static_assert(sizeof(S) == sizeof(D), "Types size mismatch");
  std::memcpy(dest, source, n * sizeof(S));
}

template <typename D, typename S>
void CopyData(D* dest, const S* source, int64_t n, const CopyDirect&) {
  std::copy(source, source + n, dest);
}

template <typename D, typename S>
void CopyData(D* dest, const S* source, int64_t n, const CopyCasted&) {
  // Use strided copy with step 1 since it has the static_cast<> required to
  // convert from/to bfloat16.
  StridedCopy(dest, 1, source, 1, n);
}

std::vector<int64_t> GetIterationDimensions(const lazy_tensors::Shape& shape) {
  // We want to favor the most minor dimension as core iteration dimension, as
  // this walks one of the two tensors buffers in a cache friendly fashion.
  // Though, if the most minor dimension is too small, we will end up doing more
  // StridedCopy() iterations in CopyTensors().
  // So we select the most minor dimension, unless one of the other dimensions
  // is more than kMinorDimScale times the most minor one.
  static const int64_t kMinorDimScale = 8;
  std::vector<int64_t> iter_dims =
      lazy_tensors::util::ToVector<int64_t>(shape.layout().minor_to_major());
  size_t index = 0;
  int64_t scaled_dim_size = kMinorDimScale * shape.dimensions(iter_dims[index]);
  for (size_t i = 1; i < iter_dims.size(); ++i) {
    int64_t dim = iter_dims[i];
    if (shape.dimensions(dim) > scaled_dim_size) {
      index = i;
      scaled_dim_size = shape.dimensions(dim);
    }
  }
  std::swap(iter_dims[0], iter_dims[index]);
  return iter_dims;
}

struct CopyPartition {
  explicit CopyPartition(c10::ArrayRef<int64_t> dimensions)
      : base(dimensions.size()), limit(dimensions.begin(), dimensions.end()) {}

  std::vector<int64_t> base;
  std::vector<int64_t> limit;
};

std::vector<CopyPartition> CreateCopyPartitions(
    c10::ArrayRef<int64_t> dimensions, int64_t strided_copy_dimension) {
  // The minimum number of elements copy that can be assigned to a thread.
  static const int64_t kMinThreadElements = 100000;
  // Use at most 50% of the available cores.
  int64_t max_parts =
      std::max<int64_t>(std::thread::hardware_concurrency() / 2, 1);
  // Find the maximum dimension which is not the strided copy dimension.
  int64_t max_dim = -1;
  for (int64_t i = 0; i < dimensions.size(); ++i) {
    if (i != strided_copy_dimension &&
        (max_dim < 0 || dimensions[i] > dimensions[max_dim])) {
      max_dim = i;
    }
  }

  int64_t num_elements = lazy_tensors::util::Multiply<int64_t>(dimensions);
  int64_t max_dim_unit_elements = num_elements / dimensions[max_dim];
  int64_t max_dim_size = dimensions[max_dim];
  int64_t part_size =
      std::max<int64_t>(std::max<int64_t>(max_dim_size / max_parts, 1),
                        kMinThreadElements / max_dim_unit_elements);
  std::vector<CopyPartition> parts;
  int64_t csize = 0;
  while (csize < max_dim_size) {
    int64_t n = std::min<int64_t>(part_size, max_dim_size - csize);
    CopyPartition p(dimensions);
    p.base[max_dim] = csize;
    p.limit[max_dim] = csize + n;
    csize += n;
    parts.emplace_back(std::move(p));
  }
  return parts;
}

template <typename SType, typename DType>
void SlicedCopy(c10::ArrayRef<int64_t> dimensions, const SType* src_data,
                c10::ArrayRef<int64_t> src_strides, DType* dest_data,
                c10::ArrayRef<int64_t> dest_strides,
                c10::ArrayRef<int64_t> iter_dims, const CopyPartition& part) {
  std::vector<int64_t> indices(part.base);
  int64_t inner_src_stride = src_strides[iter_dims.front()];
  int64_t inner_dest_stride = dest_strides[iter_dims.front()];
  int64_t n = 0;
  while (n < indices.size()) {
    StridedCopy(dest_data + GetFlatTensorOffset(dest_strides, indices),
                inner_dest_stride,
                src_data + GetFlatTensorOffset(src_strides, indices),
                inner_src_stride, dimensions[iter_dims.front()]);
    for (n = 1; n < indices.size(); ++n) {
      int64_t dim = iter_dims[n];
      indices[dim] += 1;
      if (indices[dim] < part.limit[dim]) {
        break;
      }
      indices[dim] = part.base[dim];
    }
  }
}

template <typename SType, typename DType>
void CopyTensors(const void* src_buffer, const lazy_tensors::Shape& src_shape,
                 void* dest_buffer, size_t dest_buffer_size,
                 const lazy_tensors::Shape& dest_shape) {
  CHECK(lazy_tensors::ShapeUtil::SameDimensions(src_shape, dest_shape))
      << src_shape << " vs. " << dest_shape;

  int64_t total_elements = lazy_tensors::ShapeUtil::ElementsIn(src_shape);
  CHECK_EQ(dest_buffer_size, total_elements * sizeof(DType));

  const SType* src_data = reinterpret_cast<const SType*>(src_buffer);
  DType* dest_data = reinterpret_cast<DType*>(dest_buffer);
  if (src_shape.layout().minor_to_major() ==
      dest_shape.layout().minor_to_major()) {
    CopyData<DType, SType>(dest_data, src_data, total_elements,
                           typename CopyType < NeedCast<SType>::value ||
                               NeedCast<DType>::value > ::type());
  } else if (total_elements > 0) {
    // We issue a multi-threaded copy by slicing the bigger dimension and
    // assigning its copy to different threads. This code is only valid for
    // ranks >= 2, but the layout check above covers the case.
    std::vector<int64_t> src_strides = ComputeShapeStrides(src_shape);
    std::vector<int64_t> dest_strides = ComputeShapeStrides(dest_shape);
    std::vector<int64_t> iter_dims = GetIterationDimensions(dest_shape);
    std::vector<CopyPartition> parts =
        CreateCopyPartitions(dest_shape.dimensions(), iter_dims.front());
    auto mwait = std::make_shared<lazy_tensors::util::MultiWait>(parts.size());
    for (size_t i = 0; i < parts.size(); ++i) {
      auto copy_fn = [&, i]() {
        SlicedCopy<SType, DType>(dest_shape.dimensions(), src_data, src_strides,
                                 dest_data, dest_strides, iter_dims, parts[i]);
      };
      lazy_tensors::env::ScheduleClosure(
          lazy_tensors::util::MultiWait::Completer(mwait, std::move(copy_fn)));
    }
    mwait->Wait();
  }
}

template <typename SType, typename DType>
void TensorToBuffer(const at::Tensor& tensor,
                    const lazy_tensors::Shape& dest_shape, void* dest_buffer,
                    size_t dest_buffer_size, const Device& device) {
  at::Tensor contiguous_tensor = tensor.contiguous();
  lazy_tensors::Shape src_shape = MakeTorchTensorLayout(
      Helpers::I64List(contiguous_tensor.sizes()), /*dynamic_dimensions=*/{},
      contiguous_tensor.type().scalarType());
  CopyTensors<SType, DType>(contiguous_tensor.data_ptr<SType>(), src_shape,
                            dest_buffer, dest_buffer_size, dest_shape);
}

template <typename SType>
void TensorToBufferSType(const at::Tensor& tensor,
                         const lazy_tensors::Shape& dest_shape,
                         void* dest_buffer, size_t dest_buffer_size,
                         const Device& device) {
  switch (dest_shape.at_element_type()) {
    case c10::ScalarType::BFloat16:
      TensorToBuffer<SType, c10::BFloat16>(tensor, dest_shape, dest_buffer,
                                           dest_buffer_size, device);
      break;
    case c10::ScalarType::Half:
      TensorToBuffer<SType, c10::Half>(tensor, dest_shape, dest_buffer,
                                       dest_buffer_size, device);
      break;
    case c10::ScalarType::Float:
      TensorToBuffer<SType, float>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size, device);
      break;
    case c10::ScalarType::Double:
      TensorToBuffer<SType, double>(tensor, dest_shape, dest_buffer,
                                    dest_buffer_size, device);
      break;
    case c10::ScalarType::Bool:
      TensorToBuffer<SType, bool>(tensor, dest_shape, dest_buffer,
                                  dest_buffer_size, device);
      break;
    case c10::ScalarType::Byte:
      TensorToBuffer<SType, uint8_t>(tensor, dest_shape, dest_buffer,
                                     dest_buffer_size, device);
      break;
    case c10::ScalarType::Char:
      TensorToBuffer<SType, int8_t>(tensor, dest_shape, dest_buffer,
                                    dest_buffer_size, device);
      break;
    case c10::ScalarType::Short:
      TensorToBuffer<SType, int16_t>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    // c10 doesn't have a uint16 type
    // case lazy_tensors::PrimitiveType::U16:
    //   TensorToBuffer<SType, lazy_tensors::uint16>(
    //       tensor, dest_shape, dest_buffer, dest_buffer_size, device);
    //   break;
    case c10::ScalarType::Int:
      TensorToBuffer<SType, int32_t>(tensor, dest_shape, dest_buffer,
                                     dest_buffer_size, device);
      break;
    // c10 doesn't have a uint32 type
    // case lazy_tensors::PrimitiveType::U32:
    //   TensorToBuffer<SType, lazy_tensors::uint32>(
    //       tensor, dest_shape, dest_buffer, dest_buffer_size, device);
    //   break;
    case c10::ScalarType::Long:
      TensorToBuffer<SType, int64_t>(tensor, dest_shape, dest_buffer,
                                     dest_buffer_size, device);
      break;
    // c10 doesn't have a uint64 type
    // case lazy_tensors::PrimitiveType::U64:
    //   TensorToBuffer<SType, lazy_tensors::uint64>(
    //       tensor, dest_shape, dest_buffer, dest_buffer_size, device);
    //   break;
    case c10::ScalarType::ComplexFloat:
      TensorToBuffer<SType, c10::complex<float>>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case c10::ScalarType::ComplexDouble:
      TensorToBuffer<SType, c10::complex<double>>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    default:
      LOG(ERROR) << "Destination shape type not supported: " << dest_shape;
  }
}

lazy_tensors::ComputationClient::DataPtr TensorToDataHandle(
    const at::Tensor& tensor, const lazy_tensors::Shape& shape,
    const Device& device) {
  auto populate_fn =
      [&](const lazy_tensors::ComputationClient::TensorSource& source_tensor,
          void* dest_buffer, size_t dest_buffer_size) {
        PopulateTensorBuffer(tensor, source_tensor.shape, dest_buffer,
                             dest_buffer_size, device);
      };

  std::vector<lazy_tensors::client::TensorSource> source_tensors;
  source_tensors.emplace_back(lazy_tensors::ToShapeData(shape),
                              device.ToString(), std::move(populate_fn));

  auto handles = std::vector<lazy_tensors::ComputationClient::DataPtr>{
      MakeComputationDataFromTensor(tensor, shape, device.ToString())};
  CHECK_EQ(handles.size(), 1);
  return std::move(handles.front());
}

template <typename SType, typename DType>
at::Tensor LiteralToTensor(const lazy_tensors::Literal& literal,
                           at::ScalarType atype) {
  std::vector<int64_t> dimensions =
      lazy_tensors::util::ToVector<int64_t>(literal.shape().dimensions());
  lazy_tensors::Shape torch_shape = MakeTorchTensorLayout(
      literal.shape().dimensions(), /*dynamic_dimensions=*/{},
      literal.shape().at_element_type());
  int64_t total_elements = lazy_tensors::ShapeUtil::ElementsIn(torch_shape);

  auto literal_data = literal.data<SType>();
  at::Tensor tensor = at::empty(dimensions, at::TensorOptions(atype));
  CopyTensors<SType, DType>(literal_data.data(), literal.shape(),
                            tensor.data_ptr<DType>(),
                            total_elements * sizeof(DType), torch_shape);
  return tensor;
}

template <typename SType>
at::Tensor LiteralToTensorHelper(const lazy_tensors::Literal& literal,
                                 at::ScalarType dest_element_type) {
  switch (dest_element_type) {
    case at::ScalarType::Bool:
      return LiteralToTensor<SType, bool>(literal, dest_element_type);
    case at::ScalarType::Byte:
      return LiteralToTensor<SType, uint8_t>(literal, dest_element_type);
    case at::ScalarType::Char:
      return LiteralToTensor<SType, int8_t>(literal, dest_element_type);
    case at::ScalarType::Short:
      return LiteralToTensor<SType, int16_t>(literal, dest_element_type);
    case at::ScalarType::Int:
      return LiteralToTensor<SType, int32_t>(literal, dest_element_type);
    case at::ScalarType::Long:
      return LiteralToTensor<SType, int64_t>(literal, dest_element_type);
    case at::ScalarType::Float:
      return LiteralToTensor<SType, float>(literal, dest_element_type);
    case at::ScalarType::Double:
      return LiteralToTensor<SType, double>(literal, dest_element_type);
    case at::ScalarType::BFloat16:
      return LiteralToTensor<SType, at::BFloat16>(literal, dest_element_type);
    case at::ScalarType::Half:
      return LiteralToTensor<SType, at::Half>(literal, dest_element_type);
    case at::ScalarType::ComplexFloat:
      return LiteralToTensor<SType, c10::complex<float>>(literal,
                                                         dest_element_type);
    case at::ScalarType::ComplexDouble:
      return LiteralToTensor<SType, c10::complex<double>>(literal,
                                                          dest_element_type);
    default:
      LOG(ERROR) << "Unsupported scalar type: " << dest_element_type;
  }
}

}  // namespace

void PopulateTensorBuffer(const at::Tensor& tensor,
                          const lazy_tensors::Shape& dest_shape,
                          void* dest_buffer, size_t dest_buffer_size,
                          const Device& device) {
  switch (tensor.type().scalarType()) {
    case at::ScalarType::Double:
      TensorToBufferSType<double>(tensor, dest_shape, dest_buffer,
                                  dest_buffer_size, device);
      break;
    case at::ScalarType::Float:
      TensorToBufferSType<float>(tensor, dest_shape, dest_buffer,
                                 dest_buffer_size, device);
      break;
    case at::ScalarType::BFloat16:
      TensorToBufferSType<at::BFloat16>(tensor, dest_shape, dest_buffer,
                                        dest_buffer_size, device);
      break;
    case at::ScalarType::Half:
      TensorToBufferSType<at::Half>(tensor, dest_shape, dest_buffer,
                                    dest_buffer_size, device);
      break;
    case at::ScalarType::Bool:
      TensorToBufferSType<bool>(tensor, dest_shape, dest_buffer,
                                dest_buffer_size, device);
      break;
    case at::ScalarType::Byte:
      TensorToBufferSType<uint8_t>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size, device);
      break;
    case at::ScalarType::Char:
      TensorToBufferSType<int8_t>(tensor, dest_shape, dest_buffer,
                                  dest_buffer_size, device);
      break;
    case at::ScalarType::Short:
      TensorToBufferSType<int16_t>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size, device);
      break;
    case at::ScalarType::Int:
      TensorToBufferSType<int32_t>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size, device);
      break;
    case at::ScalarType::Long:
      TensorToBufferSType<int64_t>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size, device);
      break;
    case at::ScalarType::ComplexFloat:
      TensorToBufferSType<c10::complex<float>>(tensor, dest_shape, dest_buffer,
                                               dest_buffer_size, device);
      break;
    case at::ScalarType::ComplexDouble:
      TensorToBufferSType<c10::complex<double>>(tensor, dest_shape, dest_buffer,
                                                dest_buffer_size, device);
      break;
    default:
      LOG(ERROR) << "Tensor type not supported: " << tensor.type();
  }
}

std::vector<int64_t> ComputeShapeStrides(const lazy_tensors::Shape& shape) {
  std::vector<int64_t> strides(shape.rank());
  int64_t stride = 1;
  for (auto dim : shape.layout().minor_to_major()) {
    strides[dim] = stride;
    stride *= shape.dimensions(dim);
  }
  return strides;
}

std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
  return strides;
}

at::Tensor MakeTensorFromLiteral(const lazy_tensors::Literal& literal,
                                 at::ScalarType dest_element_type) {
  switch (literal.shape().at_element_type()) {
    case c10::ScalarType::Bool:
      return LiteralToTensorHelper<bool>(literal, dest_element_type);
    case c10::ScalarType::BFloat16:
      return LiteralToTensorHelper<c10::BFloat16>(literal, dest_element_type);
    case c10::ScalarType::Half:
      return LiteralToTensorHelper<c10::Half>(literal, dest_element_type);
    case c10::ScalarType::Float:
      return LiteralToTensorHelper<float>(literal, dest_element_type);
    case c10::ScalarType::Double:
      return LiteralToTensorHelper<double>(literal, dest_element_type);
    case c10::ScalarType::Byte:
      return LiteralToTensorHelper<uint8_t>(literal, dest_element_type);
    case c10::ScalarType::Char:
      return LiteralToTensorHelper<int8_t>(literal, dest_element_type);
    case c10::ScalarType::Short:
      return LiteralToTensorHelper<int16_t>(literal, dest_element_type);
    // case lazy_tensors::PrimitiveType::U16:
    //   return LiteralToTensorHelper<lazy_tensors::uint16>(literal,
    //                                                      dest_element_type);
    case c10::ScalarType::Int:
      return LiteralToTensorHelper<int32_t>(literal, dest_element_type);
    // case lazy_tensors::PrimitiveType::U32:
    //   return LiteralToTensorHelper<lazy_tensors::uint32>(literal,
    //                                                      dest_element_type);
    case c10::ScalarType::Long:
      return LiteralToTensorHelper<int64_t>(literal, dest_element_type);
    // case lazy_tensors::PrimitiveType::U64:
    //   return LiteralToTensorHelper<lazy_tensors::uint64>(literal,
    //                                                      dest_element_type);
    case c10::ScalarType::ComplexFloat:
      return LiteralToTensorHelper<c10::complex<float>>(literal,
                                                        dest_element_type);
    case c10::ScalarType::ComplexDouble:
      return LiteralToTensorHelper<c10::complex<double>>(literal,
                                                         dest_element_type);
    default:
      LOG(ERROR) << "Unsupported literal type: " << literal.shape();
  }
}

bool TensorCompare(const at::Tensor& t1, const at::Tensor& t2) {
  if (t1.scalar_type() != t2.scalar_type() || t1.sizes() != t2.sizes()) {
    return false;
  }
  // PyTorch currently has an issue comparing tensors which have NaN values in
  // it. The compare is not deterministic. So we do memory compare here until
  // the PyTorch equal() API is fixed.
  at::Tensor contiguous_t1 = t1.contiguous();
  at::Tensor contiguous_t2 = t2.contiguous();
  return std::memcmp(contiguous_t1.data_ptr(), contiguous_t2.data_ptr(),
                     contiguous_t1.numel() * contiguous_t1.itemsize()) == 0;
}

lazy_tensors::ComputationClient::DataPtr TensorToDataHandle(
    const at::Tensor& tensor, const Device& device) {
  return TensorToDataHandle(
      tensor, CreateComputationShapeFromTensor(tensor, &device), device);
}

std::vector<lazy_tensors::ComputationClient::DataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  CHECK_EQ(tensors.size(), devices.size());
  std::vector<lazy_tensors::ComputationClient::DataPtr> result;
  result.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    Device device(devices[i]);
    lazy_tensors::Shape shape =
        CreateComputationShapeFromTensor(tensors[i], &device);
    result.push_back(
        MakeComputationDataFromTensor(tensors[i], shape, devices[i]));
  }
  return result;
}

lazy_tensors::Literal GetTensorLiteral(const at::Tensor& tensor,
                                       const lazy_tensors::Shape* shape,
                                       const Device* device) {
  Device ltc_device = GetDeviceOrCurrent(device);
  lazy_tensors::Shape computed_shape;
  if (shape == nullptr) {
    auto dimensions = Helpers::I64List(tensor.sizes());
    computed_shape = MakeTorchTensorLayout(
        dimensions, /*dynamic_dimensions=*/{}, tensor.type().scalarType());
    shape = &computed_shape;
  }
  lazy_tensors::Literal literal(*shape);
  PopulateTensorBuffer(tensor, *shape, literal.untyped_data(),
                       literal.size_bytes(), ltc_device);
  return literal;
}

std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<lazy_tensors::ComputationClient::DataPtr> data_handles,
    at::ScalarType dest_element_type) {
  std::vector<at::Tensor> tensors;
  for (const auto& handle : data_handles) {
    tensors.push_back(
        lazy_tensors::MakeTensorFromComputationData(handle, dest_element_type));
  }
  return tensors;
}

torch::lazy::hash_t TensorHash(const at::Tensor& tensor) {
  at::Tensor ctensor = tensor.contiguous();
  int64_t size = ctensor.numel() * ctensor.element_size();
  switch (ctensor.scalar_type()) {
    case at::ScalarType::Bool:
      return torch::lazy::DataHash(ctensor.data_ptr<bool>(), size);
    case at::ScalarType::Byte:
      return torch::lazy::DataHash(ctensor.data_ptr<uint8_t>(), size);
    case at::ScalarType::Char:
      return torch::lazy::DataHash(ctensor.data_ptr<int8_t>(), size);
    case at::ScalarType::Short:
      return torch::lazy::DataHash(ctensor.data_ptr<int16_t>(), size);
    case at::ScalarType::Int:
      return torch::lazy::DataHash(ctensor.data_ptr<int32_t>(), size);
    case at::ScalarType::Long:
      return torch::lazy::DataHash(ctensor.data_ptr<int64_t>(), size);
    case at::ScalarType::Float:
      return torch::lazy::DataHash(ctensor.data_ptr<float>(), size);
    case at::ScalarType::Double:
      return torch::lazy::DataHash(ctensor.data_ptr<double>(), size);
    case at::ScalarType::BFloat16:
      return torch::lazy::DataHash(ctensor.data_ptr<at::BFloat16>(), size);
    case at::ScalarType::Half:
      return torch::lazy::DataHash(ctensor.data_ptr<at::Half>(), size);
    case at::ScalarType::ComplexFloat:
      return torch::lazy::DataHash(ctensor.data_ptr<c10::complex<float>>(),
                                   size);
    case at::ScalarType::ComplexDouble:
      return torch::lazy::DataHash(ctensor.data_ptr<c10::complex<double>>(),
                                   size);
    default:
      LOG(ERROR) << "Unsupported scalar type: " << ctensor.scalar_type();
  }
}

std::vector<lazy_tensors::Shape> GetComponentShapes(
    const lazy_tensors::Shape& shape) {
  std::vector<lazy_tensors::Shape> component_shapes;
  if (shape.IsTuple()) {
    for (const lazy_tensors::Shape& component_shape : shape.tuple_shapes()) {
      CHECK(!component_shape.IsTuple()) << shape;
      component_shapes.push_back(component_shape);
    }
  } else {
    component_shapes.push_back(shape);
  }
  return component_shapes;
}

lazy_tensors::Shape MakeShapeWithDeviceLayout(const lazy_tensors::Shape& shape,
                                              DeviceType device_type) {
  lazy_tensors::Shape device_shape(shape);
  lazy_tensors::ShapeUtil::ForEachMutableSubshape(
      &device_shape,
      [&](lazy_tensors::Shape* subshape, const lazy_tensors::ShapeIndex&) {
        if (subshape->IsArray()) {
          *subshape = MakeArrayShapeFromDimensions(
              subshape->dimensions(), subshape->dynamic_dimensions(),
              subshape->at_element_type(), device_type);
        }
      });
  return device_shape;
}

lazy_tensors::Shape CreateComputationShapeFromTensor(const at::Tensor& tensor,
                                                     const Device* device) {
  Device ltc_device = GetDeviceOrCurrent(device);
  return MakeArrayShapeFromDimensions(Helpers::I64List(tensor.sizes()),
                                      /*dynamic_dimensions=*/{},
                                      tensor.type().scalarType(),
                                      ltc_device.hw_type);
}

at::ScalarType TensorTypeFromLtcType(lazy_tensors::PrimitiveType ltc_type) {
  switch (ltc_type) {
    case lazy_tensors::PrimitiveType::BF16:
      return UseBF16() || DowncastBF16() ? at::ScalarType::Float
                                         : at::ScalarType::BFloat16;
    case lazy_tensors::PrimitiveType::F16:
      return UseF16() || DowncastF16() ? at::ScalarType::Float
                                       : at::ScalarType::Half;
    case lazy_tensors::PrimitiveType::F32:
      return DowncastBF16() || DowncastF16() ? at::ScalarType::Double
                                             : at::ScalarType::Float;
    case lazy_tensors::PrimitiveType::F64:
      return at::ScalarType::Double;
    case lazy_tensors::PrimitiveType::PRED:
      return at::ScalarType::Bool;
    case lazy_tensors::PrimitiveType::U8:
      return at::ScalarType::Byte;
    case lazy_tensors::PrimitiveType::S8:
      return at::ScalarType::Char;
    case lazy_tensors::PrimitiveType::S16:
    case lazy_tensors::PrimitiveType::U16:
      return at::ScalarType::Short;
    case lazy_tensors::PrimitiveType::S32:
    case lazy_tensors::PrimitiveType::U32:
      return at::ScalarType::Int;
    case lazy_tensors::PrimitiveType::S64:
    case lazy_tensors::PrimitiveType::U64:
      return at::ScalarType::Long;
    case lazy_tensors::PrimitiveType::C64:
      return at::ScalarType::ComplexFloat;
    case lazy_tensors::PrimitiveType::C128:
      return at::ScalarType::ComplexDouble;
    default:
      LOG(ERROR) << "Type not supported: " << ltc_type;
  }
}

lazy_tensors::PrimitiveType TensorTypeToLtcType(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Double:
      return lazy_tensors::PrimitiveType::F64;
    case at::ScalarType::Float:
      return lazy_tensors::PrimitiveType::F32;
    case at::ScalarType::BFloat16:
      return lazy_tensors::PrimitiveType::BF16;
    case at::ScalarType::Half:
      return lazy_tensors::PrimitiveType::F16;
    case at::ScalarType::Bool:
      return lazy_tensors::PrimitiveType::PRED;
    case at::ScalarType::Byte:
      return lazy_tensors::PrimitiveType::U8;
    case at::ScalarType::Char:
      return lazy_tensors::PrimitiveType::S8;
    case at::ScalarType::Short:
      return lazy_tensors::PrimitiveType::S16;
    case at::ScalarType::Int:
      return lazy_tensors::PrimitiveType::S32;
    case at::ScalarType::Long:
      return lazy_tensors::PrimitiveType::S64;
    case at::ScalarType::ComplexFloat:
      return lazy_tensors::PrimitiveType::C64;
    case at::ScalarType::ComplexDouble:
      return lazy_tensors::PrimitiveType::C128;
    default:
      LOG(ERROR) << "Type not supported: " << scalar_type;
  }
}

lazy_tensors::PrimitiveType GetDevicePrimitiveType(
    lazy_tensors::PrimitiveType type, const Device* device) {
  Device ltc_device = GetDeviceOrCurrent(device);
  switch (type) {
    case lazy_tensors::PrimitiveType::F64:
      if (UseF16()) {
        return lazy_tensors::PrimitiveType::F16;
      }
      if (UseBF16()) {
        return lazy_tensors::PrimitiveType::BF16;
      }
      if (DowncastBF16() || DowncastF16()) {
        return lazy_tensors::PrimitiveType::F32;
      }
      return ltc_device.hw_type != DeviceType::TPU
                 ? lazy_tensors::PrimitiveType::F64
                 : lazy_tensors::PrimitiveType::F32;
    case lazy_tensors::PrimitiveType::F32:
      if (UseF16() || DowncastF16()) {
        return lazy_tensors::PrimitiveType::F16;
      }
      return UseBF16() || DowncastBF16() ? lazy_tensors::PrimitiveType::BF16
                                         : lazy_tensors::PrimitiveType::F32;
    case lazy_tensors::PrimitiveType::U16:
      return ltc_device.hw_type != DeviceType::TPU
                 ? lazy_tensors::PrimitiveType::U16
                 : lazy_tensors::PrimitiveType::U32;
    case lazy_tensors::PrimitiveType::S16:
      return ltc_device.hw_type != DeviceType::TPU
                 ? lazy_tensors::PrimitiveType::S16
                 : lazy_tensors::PrimitiveType::S32;
    case lazy_tensors::PrimitiveType::S64:
      return Use32BitLong() ? lazy_tensors::PrimitiveType::S32
                            : lazy_tensors::PrimitiveType::S64;
    case lazy_tensors::PrimitiveType::U64:
      return Use32BitLong() ? lazy_tensors::PrimitiveType::U32
                            : lazy_tensors::PrimitiveType::U64;
    case lazy_tensors::PrimitiveType::C128:
      return ltc_device.hw_type != DeviceType::TPU
                 ? lazy_tensors::PrimitiveType::C128
                 : lazy_tensors::PrimitiveType::C64;
    default:
      return type;
  }
}

lazy_tensors::PrimitiveType MakeLtcPrimitiveType(at::ScalarType scalar_type,
                                                 const Device* device) {
  switch (scalar_type) {
    case at::ScalarType::Double:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::F64, device);
    case at::ScalarType::Float:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::F32, device);
    case at::ScalarType::BFloat16:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::BF16, device);
    case at::ScalarType::Half:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::F16, device);
    case at::ScalarType::Bool:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::PRED, device);
    case at::ScalarType::Byte:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::U8, device);
    case at::ScalarType::Char:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::S8, device);
    case at::ScalarType::Short:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::S16, device);
    case at::ScalarType::Int:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::S32, device);
    case at::ScalarType::Long:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::S64, device);
    case at::ScalarType::ComplexFloat:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::C64, device);
    case at::ScalarType::ComplexDouble:
      return GetDevicePrimitiveType(lazy_tensors::PrimitiveType::C128, device);
    default:
      LOG(ERROR) << "Type not supported: " << scalar_type;
  }
}

bool RequiresRawTypeCasting(at::ScalarType scalar_type, const Device* device) {
  switch (scalar_type) {
    case at::ScalarType::Byte:
    case at::ScalarType::Char:
    case at::ScalarType::Short:
      return MakeLtcPrimitiveType(scalar_type, device) !=
             TensorTypeToLtcType(scalar_type);
    default:
      return false;
  }
}

c10::ScalarType GetShapeDimensionType(const Device* device) {
  Device ltc_device = GetDeviceOrCurrent(device);
  return ltc_device.hw_type == DeviceType::CPU ? c10::ScalarType::Long
                                               : c10::ScalarType::Int;
}

bool IsSpecialScalar(const at::Scalar& value) {
  static bool no_scalars =
      lazy_tensors::sys_util::GetEnvBool("NO_SPECIAL_SCALARS", false);
  if (!no_scalars && (value.isIntegral() || value.isFloatingPoint())) {
    double scalar_value = value.toDouble();
    return scalar_value == 0.0 || std::fabs(scalar_value) == 1.0;
  }
  return false;
}

}  // namespace torch_lazy_tensors
