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

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/helpers.h"
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

}  // namespace

std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t i = sizes.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * sizes[i - 1];
  }
  return strides;
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

compiler::DataPtr TensorToDataHandle(const at::Tensor& tensor,
                                     const Device& device) {
  return torch_lazy_tensors::compiler::getBackendRegistrar()
      ->MakeComputationDataFromTensor(
          tensor, lazy_tensors::Shape(tensor.scalar_type(), tensor.sizes()),
          device.ToString());
}

std::vector<compiler::DataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  CHECK_EQ(tensors.size(), devices.size());
  std::vector<compiler::DataPtr> result;
  result.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    Device device(devices[i]);
    lazy_tensors::Shape shape(tensors[i].scalar_type(), tensors[i].sizes());
    result.push_back(
        torch_lazy_tensors::compiler::getBackendRegistrar()
            ->MakeComputationDataFromTensor(tensors[i], shape, devices[i]));
  }
  return result;
}

std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<compiler::DataPtr> data_handles,
    at::ScalarType dest_element_type) {
  std::vector<at::Tensor> tensors;
  for (const auto& handle : data_handles) {
    tensors.push_back(
        torch_lazy_tensors::compiler::getBackendRegistrar()
            ->MakeTensorFromComputationData(handle, dest_element_type));
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
