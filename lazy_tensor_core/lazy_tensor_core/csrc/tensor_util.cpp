#include "lazy_tensor_core/csrc/tensor_util.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <list>
#include <numeric>
#include <thread>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/layout_manager.h"
// #include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/ltc_logging.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/core/lib/bfloat16/bfloat16.h"
#include "lazy_tensors/literal_util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace {

bool ShouldUseBF16() {
  bool use_bf16 = lazy_tensors::sys_util::GetEnvBool("LTC_USE_BF16", false);
  if (use_bf16) {
    LTC_LOG(INFO) << "Using BF16 data type for floating point values";
  }
  return use_bf16;
}

bool ShouldUseF16() {
  bool use_fp16 = lazy_tensors::sys_util::GetEnvBool("LTC_USE_FP16", false);
  if (use_fp16) {
    LTC_LOG(INFO) << "Using F16 data type for floating point values";
  }
  return use_fp16;
}

bool ShouldDowncastToBF16() {
  bool downcast_bf16 =
      lazy_tensors::sys_util::GetEnvBool("LTC_DOWNCAST_BF16", false);
  if (downcast_bf16) {
    LTC_LOG(INFO) << "Downcasting floating point values, F64->F32, F32->BF16";
  }
  return downcast_bf16;
}

bool ShouldDowncastToF16() {
  bool downcast_fp16 =
      lazy_tensors::sys_util::GetEnvBool("LTC_DOWNCAST_FP16", false);
  if (downcast_fp16) {
    LTC_LOG(INFO) << "Downcasting floating point values, F64->F32, F32->FP16";
  }
  return downcast_fp16;
}

bool ShouldUse32BitLong() {
  bool use_32bit_long =
      lazy_tensors::sys_util::GetEnvBool("LTC_USE_32BIT_LONG", false);
  if (use_32bit_long) {
    LTC_LOG(INFO) << "Using 32bit integers for kLong values";
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

lazy_tensors::PrimitiveType LtcTypeFromTensorType(at::ScalarType scalar_type,
                                                  const Device& device) {
  switch (scalar_type) {
    case at::ScalarType::Double:
      return device.hw_type != DeviceType::TPU
                 ? lazy_tensors::PrimitiveType::F64
                 : lazy_tensors::PrimitiveType::F32;
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
      LTC_ERROR() << "Type not supported: " << scalar_type;
  }
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
struct Caster<lazy_tensors::bfloat16> {
  template <typename D>
  D cast(const lazy_tensors::bfloat16& value) const {
    return static_cast<D>(static_cast<float>(value));
  }
};
template <>
struct Caster<lazy_tensors::half> {
  template <typename D>
  D cast(const lazy_tensors::half& value) const {
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
std::complex<float> Caster<c10::complex<float>>::cast<std::complex<float>>(
    const c10::complex<float>& value) const {
  return std::complex<float>(value.real(), value.imag());
}

template <>
std::complex<double> Caster<c10::complex<float>>::cast<std::complex<double>>(
    const c10::complex<float>& value) const {
  return std::complex<double>(value.real(), value.imag());
}

template <>
struct Caster<c10::complex<double>> {
  template <typename D>
  D cast(const c10::complex<double>& value) const {
    return static_cast<D>(value.real());
  }
};

template <>
std::complex<float> Caster<c10::complex<double>>::cast<std::complex<float>>(
    const c10::complex<double>& value) const {
  return std::complex<float>(value.real(), value.imag());
}

template <>
std::complex<double> Caster<c10::complex<double>>::cast<std::complex<double>>(
    const c10::complex<double>& value) const {
  return std::complex<double>(value.real(), value.imag());
}

template <>
struct Caster<std::complex<float>> {
  template <typename D>
  D cast(const std::complex<float>& value) const {
    return static_cast<D>(value.real());
  }
};

template <>
c10::complex<float> Caster<std::complex<float>>::cast<c10::complex<float>>(
    const std::complex<float>& value) const {
  return c10::complex<float>(value.real(), value.imag());
}
template <>
c10::complex<double> Caster<std::complex<float>>::cast<c10::complex<double>>(
    const std::complex<float>& value) const {
  return c10::complex<double>(value.real(), value.imag());
}

template <>
struct Caster<std::complex<double>> {
  template <typename D>
  D cast(const std::complex<double>& value) const {
    return static_cast<D>(value.real());
  }
};

// Copies n bytes from source to dest, with different stride values for source
// and destination.
template <typename S, typename D>
void StridedCopy(D* dest, lazy_tensors::int64 dest_stride, const S* source,
                 lazy_tensors::int64 source_stride, lazy_tensors::int64 n) {
  Caster<S> caster;
  const S* source_top = source + n * source_stride;
  for (; source < source_top; dest += dest_stride, source += source_stride) {
    *dest = caster.template cast<D>(*source);
  }
}

// Computes the offset of the value at a given index, assuming a contiguous/flat
// tensor data representation.
template <typename S>
lazy_tensors::int64 GetFlatTensorOffset(
    const S& strides, const std::vector<lazy_tensors::int64>& indices) {
  lazy_tensors::int64 base = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    base += indices[i] * strides[i];
  }
  return base;
}

// The lazy_tensors::bfloat16 does not have implicit cast operations, so using
// std::copy() for it, is not going to work.
struct CopyDirect {};
struct CopyCasted {};

template <typename T>
struct NeedCast {
  static constexpr bool value = false;
};
template <>
struct NeedCast<lazy_tensors::bfloat16> {
  static constexpr bool value = true;
};
template <>
struct NeedCast<at::BFloat16> {
  static constexpr bool value = true;
};
template <>
struct NeedCast<lazy_tensors::half> {
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
template <>
struct NeedCast<std::complex<float>> {
  static constexpr bool value = true;
};
template <>
struct NeedCast<std::complex<double>> {
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
void CheckedMemcpy(D* dest, const S* source, lazy_tensors::int64 n) {
  static_assert(sizeof(S) == sizeof(D), "Types size mismatch");
  std::memcpy(dest, source, n * sizeof(S));
}

template <typename D, typename S>
void CopyData(D* dest, const S* source, lazy_tensors::int64 n,
              const CopyDirect&) {
  std::copy(source, source + n, dest);
}

template <typename D, typename S>
void CopyData(D* dest, const S* source, lazy_tensors::int64 n,
              const CopyCasted&) {
  // Use strided copy with step 1 since it has the static_cast<> required to
  // convert from/to bfloat16.
  StridedCopy(dest, 1, source, 1, n);
}

template <>
void CopyData<at::BFloat16, lazy_tensors::bfloat16>(
    at::BFloat16* dest, const lazy_tensors::bfloat16* source,
    lazy_tensors::int64 n, const CopyCasted&) {
  CheckedMemcpy<at::BFloat16, lazy_tensors::bfloat16>(dest, source, n);
}
template <>
void CopyData<lazy_tensors::bfloat16, at::BFloat16>(
    lazy_tensors::bfloat16* dest, const at::BFloat16* source,
    lazy_tensors::int64 n, const CopyCasted&) {
  CheckedMemcpy<lazy_tensors::bfloat16, at::BFloat16>(dest, source, n);
}

std::vector<lazy_tensors::int64> GetIterationDimensions(
    const lazy_tensors::Shape& shape) {
  // We want to favor the most minor dimension as core iteration dimension, as
  // this walks one of the two tensors buffers in a cache friendly fashion.
  // Though, if the most minor dimension is too small, we will end up doing more
  // StridedCopy() iterations in CopyTensors().
  // So we select the most minor dimension, unless one of the other dimensions
  // is more than kMinorDimScale times the most minor one.
  static const lazy_tensors::int64 kMinorDimScale = 8;
  std::vector<lazy_tensors::int64> iter_dims =
      lazy_tensors::util::ToVector<lazy_tensors::int64>(
          shape.layout().minor_to_major());
  size_t index = 0;
  lazy_tensors::int64 scaled_dim_size =
      kMinorDimScale * shape.dimensions(iter_dims[index]);
  for (size_t i = 1; i < iter_dims.size(); ++i) {
    lazy_tensors::int64 dim = iter_dims[i];
    if (shape.dimensions(dim) > scaled_dim_size) {
      index = i;
      scaled_dim_size = shape.dimensions(dim);
    }
  }
  std::swap(iter_dims[0], iter_dims[index]);
  return iter_dims;
}

struct CopyPartition {
  explicit CopyPartition(
      lazy_tensors::Span<const lazy_tensors::int64> dimensions)
      : base(dimensions.size()), limit(dimensions.begin(), dimensions.end()) {}

  std::vector<lazy_tensors::int64> base;
  std::vector<lazy_tensors::int64> limit;
};

std::vector<CopyPartition> CreateCopyPartitions(
    lazy_tensors::Span<const lazy_tensors::int64> dimensions,
    lazy_tensors::int64 strided_copy_dimension) {
  // The minimum number of elements copy that can be assigned to a thread.
  static const lazy_tensors::int64 kMinThreadElements = 100000;
  // Use at most 50% of the available cores.
  lazy_tensors::int64 max_parts =
      std::max<lazy_tensors::int64>(std::thread::hardware_concurrency() / 2, 1);
  // Find the maximum dimension which is not the strided copy dimension.
  lazy_tensors::int64 max_dim = -1;
  for (lazy_tensors::int64 i = 0; i < dimensions.size(); ++i) {
    if (i != strided_copy_dimension &&
        (max_dim < 0 || dimensions[i] > dimensions[max_dim])) {
      max_dim = i;
    }
  }

  lazy_tensors::int64 num_elements =
      lazy_tensors::util::Multiply<lazy_tensors::int64>(dimensions);
  lazy_tensors::int64 max_dim_unit_elements =
      num_elements / dimensions[max_dim];
  lazy_tensors::int64 max_dim_size = dimensions[max_dim];
  lazy_tensors::int64 part_size = std::max<lazy_tensors::int64>(
      std::max<lazy_tensors::int64>(max_dim_size / max_parts, 1),
      kMinThreadElements / max_dim_unit_elements);
  std::vector<CopyPartition> parts;
  lazy_tensors::int64 csize = 0;
  while (csize < max_dim_size) {
    lazy_tensors::int64 n =
        std::min<lazy_tensors::int64>(part_size, max_dim_size - csize);
    CopyPartition p(dimensions);
    p.base[max_dim] = csize;
    p.limit[max_dim] = csize + n;
    csize += n;
    parts.emplace_back(std::move(p));
  }
  return parts;
}

template <typename SType, typename DType>
void SlicedCopy(lazy_tensors::Span<const lazy_tensors::int64> dimensions,
                const SType* src_data,
                lazy_tensors::Span<const lazy_tensors::int64> src_strides,
                DType* dest_data,
                lazy_tensors::Span<const lazy_tensors::int64> dest_strides,
                lazy_tensors::Span<const lazy_tensors::int64> iter_dims,
                const CopyPartition& part) {
  std::vector<lazy_tensors::int64> indices(part.base);
  lazy_tensors::int64 inner_src_stride = src_strides[iter_dims.front()];
  lazy_tensors::int64 inner_dest_stride = dest_strides[iter_dims.front()];
  lazy_tensors::int64 n = 0;
  while (n < indices.size()) {
    StridedCopy(dest_data + GetFlatTensorOffset(dest_strides, indices),
                inner_dest_stride,
                src_data + GetFlatTensorOffset(src_strides, indices),
                inner_src_stride, dimensions[iter_dims.front()]);
    for (n = 1; n < indices.size(); ++n) {
      lazy_tensors::int64 dim = iter_dims[n];
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
  LTC_CHECK(lazy_tensors::ShapeUtil::SameDimensions(src_shape, dest_shape))
      << src_shape << " vs. " << dest_shape;

  lazy_tensors::int64 total_elements =
      lazy_tensors::ShapeUtil::ElementsIn(src_shape);
  LTC_CHECK_EQ(dest_buffer_size, total_elements * sizeof(DType));

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
    std::vector<lazy_tensors::int64> src_strides =
        ComputeShapeStrides(src_shape);
    std::vector<lazy_tensors::int64> dest_strides =
        ComputeShapeStrides(dest_shape);
    std::vector<lazy_tensors::int64> iter_dims =
        GetIterationDimensions(dest_shape);
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
      LtcTypeFromTensorType(contiguous_tensor.type().scalarType(), device));
  CopyTensors<SType, DType>(contiguous_tensor.data_ptr<SType>(), src_shape,
                            dest_buffer, dest_buffer_size, dest_shape);
}

template <typename SType>
void TensorToBufferSType(const at::Tensor& tensor,
                         const lazy_tensors::Shape& dest_shape,
                         void* dest_buffer, size_t dest_buffer_size,
                         const Device& device) {
  switch (dest_shape.element_type()) {
    case lazy_tensors::PrimitiveType::BF16:
      TensorToBuffer<SType, lazy_tensors::bfloat16>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::F16:
      TensorToBuffer<SType, lazy_tensors::half>(tensor, dest_shape, dest_buffer,
                                                dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::F32:
      TensorToBuffer<SType, float>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::F64:
      TensorToBuffer<SType, double>(tensor, dest_shape, dest_buffer,
                                    dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::PRED:
      TensorToBuffer<SType, bool>(tensor, dest_shape, dest_buffer,
                                  dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::U8:
      TensorToBuffer<SType, lazy_tensors::uint8>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::S8:
      TensorToBuffer<SType, lazy_tensors::int8>(tensor, dest_shape, dest_buffer,
                                                dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::S16:
      TensorToBuffer<SType, lazy_tensors::int16>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::U16:
      TensorToBuffer<SType, lazy_tensors::uint16>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::S32:
      TensorToBuffer<SType, lazy_tensors::int32>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::U32:
      TensorToBuffer<SType, lazy_tensors::uint32>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::S64:
      TensorToBuffer<SType, lazy_tensors::int64>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::U64:
      TensorToBuffer<SType, lazy_tensors::uint64>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::C64:
      TensorToBuffer<SType, lazy_tensors::complex64>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    case lazy_tensors::PrimitiveType::C128:
      TensorToBuffer<SType, lazy_tensors::complex128>(
          tensor, dest_shape, dest_buffer, dest_buffer_size, device);
      break;
    default:
      LTC_ERROR() << "Destination shape type not supported: " << dest_shape;
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
  LTC_CHECK_EQ(handles.size(), 1);
  return std::move(handles.front());
}

template <typename SType, typename DType>
at::Tensor LiteralToTensor(const lazy_tensors::Literal& literal,
                           at::ScalarType atype) {
  std::vector<int64_t> dimensions =
      lazy_tensors::util::ToVector<int64_t>(literal.shape().dimensions());
  lazy_tensors::Shape torch_shape = MakeTorchTensorLayout(
      literal.shape().dimensions(), /*dynamic_dimensions=*/{},
      literal.shape().element_type());
  lazy_tensors::int64 total_elements =
      lazy_tensors::ShapeUtil::ElementsIn(torch_shape);

  const auto literal_data = literal.data<SType>();
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
      LTC_ERROR() << "Unsupported scalar type: " << dest_element_type;
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
      LTC_ERROR() << "Tensor type not supported: " << tensor.type();
  }
}

std::vector<lazy_tensors::int64> ComputeShapeStrides(
    const lazy_tensors::Shape& shape) {
  std::vector<lazy_tensors::int64> strides(shape.rank());
  lazy_tensors::int64 stride = 1;
  for (auto dim : shape.layout().minor_to_major()) {
    strides[dim] = stride;
    stride *= shape.dimensions(dim);
  }
  return strides;
}

std::vector<lazy_tensors::int64> ComputeArrayStrides(
    lazy_tensors::Span<const lazy_tensors::int64> sizes) {
  std::vector<lazy_tensors::int64> strides(sizes.size(), 1);
  for (lazy_tensors::int64 i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
  return strides;
}

at::Tensor MakeTensorFromLiteral(const lazy_tensors::Literal& literal,
                                 at::ScalarType dest_element_type) {
  switch (literal.shape().element_type()) {
    case lazy_tensors::PrimitiveType::PRED:
      return LiteralToTensorHelper<bool>(literal, dest_element_type);
    case lazy_tensors::PrimitiveType::BF16:
      return LiteralToTensorHelper<lazy_tensors::bfloat16>(literal,
                                                           dest_element_type);
    case lazy_tensors::PrimitiveType::F16:
      return LiteralToTensorHelper<lazy_tensors::half>(literal,
                                                       dest_element_type);
    case lazy_tensors::PrimitiveType::F32:
      return LiteralToTensorHelper<float>(literal, dest_element_type);
    case lazy_tensors::PrimitiveType::F64:
      return LiteralToTensorHelper<double>(literal, dest_element_type);
    case lazy_tensors::PrimitiveType::U8:
      return LiteralToTensorHelper<lazy_tensors::uint8>(literal,
                                                        dest_element_type);
    case lazy_tensors::PrimitiveType::S8:
      return LiteralToTensorHelper<lazy_tensors::int8>(literal,
                                                       dest_element_type);
    case lazy_tensors::PrimitiveType::S16:
      return LiteralToTensorHelper<lazy_tensors::int16>(literal,
                                                        dest_element_type);
    case lazy_tensors::PrimitiveType::U16:
      return LiteralToTensorHelper<lazy_tensors::uint16>(literal,
                                                         dest_element_type);
    case lazy_tensors::PrimitiveType::S32:
      return LiteralToTensorHelper<lazy_tensors::int32>(literal,
                                                        dest_element_type);
    case lazy_tensors::PrimitiveType::U32:
      return LiteralToTensorHelper<lazy_tensors::uint32>(literal,
                                                         dest_element_type);
    case lazy_tensors::PrimitiveType::S64:
      return LiteralToTensorHelper<lazy_tensors::int64>(literal,
                                                        dest_element_type);
    case lazy_tensors::PrimitiveType::U64:
      return LiteralToTensorHelper<lazy_tensors::uint64>(literal,
                                                         dest_element_type);
    case lazy_tensors::PrimitiveType::C64:
      return LiteralToTensorHelper<lazy_tensors::complex64>(literal,
                                                            dest_element_type);
    case lazy_tensors::PrimitiveType::C128:
      return LiteralToTensorHelper<lazy_tensors::complex128>(literal,
                                                             dest_element_type);
    default:
      LTC_ERROR() << "Unsupported literal type: " << literal.shape();
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
  LTC_CHECK_EQ(tensors.size(), devices.size());
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
        dimensions, /*dynamic_dimensions=*/{},
        LtcTypeFromTensorType(tensor.type().scalarType(), ltc_device));
    shape = &computed_shape;
  }
  lazy_tensors::Literal literal(*shape);
  PopulateTensorBuffer(tensor, *shape, literal.untyped_data(),
                       literal.size_bytes(), ltc_device);
  return literal;
}

std::vector<at::Tensor> DataHandlesToTensors(
    lazy_tensors::Span<const lazy_tensors::ComputationClient::DataPtr>
        data_handles,
    at::ScalarType dest_element_type) {
  std::vector<at::Tensor> tensors;
  for (const auto& handle : data_handles) {
    tensors.push_back(
        lazy_tensors::MakeTensorFromComputationData(handle, dest_element_type));
  }
  return tensors;
}

lazy_tensors::hash_t TensorHash(const at::Tensor& tensor) {
  at::Tensor ctensor = tensor.contiguous();
  int64_t size = ctensor.numel() * ctensor.element_size();
  switch (ctensor.scalar_type()) {
    case at::ScalarType::Bool:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<bool>(), size);
    case at::ScalarType::Byte:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<uint8_t>(), size);
    case at::ScalarType::Char:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<int8_t>(), size);
    case at::ScalarType::Short:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<int16_t>(), size);
    case at::ScalarType::Int:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<int32_t>(), size);
    case at::ScalarType::Long:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<int64_t>(), size);
    case at::ScalarType::Float:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<float>(), size);
    case at::ScalarType::Double:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<double>(), size);
    case at::ScalarType::BFloat16:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<at::BFloat16>(),
                                          size);
    case at::ScalarType::Half:
      return lazy_tensors::util::DataHash(ctensor.data_ptr<at::Half>(), size);
    case at::ScalarType::ComplexFloat:
      return lazy_tensors::util::DataHash(
          ctensor.data_ptr<c10::complex<float>>(), size);
    case at::ScalarType::ComplexDouble:
      return lazy_tensors::util::DataHash(
          ctensor.data_ptr<c10::complex<double>>(), size);
    default:
      LTC_ERROR() << "Unsupported scalar type: " << ctensor.scalar_type();
  }
}

std::vector<lazy_tensors::Shape> GetComponentShapes(
    const lazy_tensors::Shape& shape) {
  std::vector<lazy_tensors::Shape> component_shapes;
  if (shape.IsTuple()) {
    for (const lazy_tensors::Shape& component_shape : shape.tuple_shapes()) {
      LTC_CHECK(!component_shape.IsTuple()) << shape;
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
              subshape->element_type(), device_type);
        }
      });
  return device_shape;
}

lazy_tensors::Shape CreateComputationShapeFromTensor(const at::Tensor& tensor,
                                                     const Device* device) {
  Device ltc_device = GetDeviceOrCurrent(device);
  return MakeArrayShapeFromDimensions(
      Helpers::I64List(tensor.sizes()),
      /*dynamic_dimensions=*/{},
      MakeLtcPrimitiveType(tensor.type().scalarType(), &ltc_device),
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
      LTC_ERROR() << "Type not supported: " << ltc_type;
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
      LTC_ERROR() << "Type not supported: " << scalar_type;
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
      LTC_ERROR() << "Type not supported: " << scalar_type;
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

lazy_tensors::PrimitiveType GetShapeDimensionType(const Device* device) {
  Device ltc_device = GetDeviceOrCurrent(device);
  return ltc_device.hw_type == DeviceType::CPU
             ? lazy_tensors::PrimitiveType::S64
             : lazy_tensors::PrimitiveType::S32;
}

}  // namespace torch_lazy_tensors
