#include <c10/util/irange.h>

// Extract size and strides
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::string TensorArgAbstract::toString() const {
  std::stringstream ss;
  auto rank = getRank();
  ss << "tensor dtype: " << getDataType() << " sizes: (";
  for (auto i = 0; i < rank; i++) {
    ss << getSize(i) << ", ";
  }
  ss << ") stride: (";
  for (auto i = 0; i < rank; i++) {
    ss << getStride(i) << ", ";
  }
  ss << ") pointer: " << getPointer();
  return ss.str();
}

namespace {

template <typename T, typename nvfuser_index_t>
std::unique_ptr<TensorArgAbstract> getTensorArg(int nDims) {
  switch (nDims) {
    case (0):
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 0, nvfuser_index_t>,
          nvfuser_index_t>>();
    case (1):
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 1, nvfuser_index_t>,
          nvfuser_index_t>>();
    case (2):
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 2, nvfuser_index_t>,
          nvfuser_index_t>>();
    case (3):
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 3, nvfuser_index_t>,
          nvfuser_index_t>>();
    case (4):
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 4, nvfuser_index_t>,
          nvfuser_index_t>>();
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    case (5):
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 5, nvfuser_index_t>,
          nvfuser_index_t>>();
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    case (6):
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 6, nvfuser_index_t>,
          nvfuser_index_t>>();
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    case (7):
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 7, nvfuser_index_t>,
          nvfuser_index_t>>();
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    case (8):
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      return std::make_unique<TensorArg<
          TensorArgCodegen<T, 8, nvfuser_index_t>,
          nvfuser_index_t>>();
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to generate a tensor to run a generated kernel with ",
          nDims,
          " dimensions, however only 0 to 8 dimensional tensor are supported.");
  }
  return nullptr;
}

template <typename INDEX_MODE>
std::unique_ptr<TensorArgAbstract> getTensorArg(
    c10::ScalarType dtype,
    int nDims) {
  switch (dtype) {
    case c10::ScalarType::Double:
      return getTensorArg<double, INDEX_MODE>(nDims);
    case c10::ScalarType::Float:
      return getTensorArg<float, INDEX_MODE>(nDims);
    case c10::ScalarType::Half:
      return getTensorArg<at::Half, INDEX_MODE>(nDims);
    case c10::ScalarType::BFloat16:
      return getTensorArg<at::BFloat16, INDEX_MODE>(nDims);
    case c10::ScalarType::Bool:
      return getTensorArg<bool, INDEX_MODE>(nDims);
    case c10::ScalarType::Long:
      return getTensorArg<int64_t, INDEX_MODE>(nDims);
    case c10::ScalarType::Int:
      return getTensorArg<int32_t, INDEX_MODE>(nDims);
    case c10::ScalarType::ComplexFloat:
      return getTensorArg<c10::complex<float>, INDEX_MODE>(nDims);
    case c10::ScalarType::ComplexDouble:
      return getTensorArg<c10::complex<double>, INDEX_MODE>(nDims);
    default:
      TORCH_CHECK(
          false,
          "Dtype: ",
          dtype,
          " not currently supported in code generated kernels.");
  }
}

std::unique_ptr<TensorArgAbstract> getTensorArg(
    c10::ScalarType dtype,
    int nDims,
    KernelIndexMode index_mode) {
  switch (index_mode) {
    case KernelIndexMode::INT32:
      return getTensorArg<int>(dtype, nDims);
    case KernelIndexMode::INT64:
      return getTensorArg<int64_t>(dtype, nDims);
    default:
      break;
  }

  TORCH_INTERNAL_ASSERT(false, "unknown index mode");
  return nullptr;
}

} // namespace

KernelArgumentHolder KernelArgumentHolder::createKernelArgumentHolder(
    const c10::ArrayRef<c10::IValue>& inputs) {
  if (inputs.empty()) {
    // default to int32 on device 0
    KernelArgumentHolder args(KernelIndexMode::INT32);
    args.setDeviceIndex(0);
    return args;
  }
  auto device_index = getCommonDeviceCUDA(inputs);
  auto index_mode = collectIndexMode(inputs);

  KernelArgumentHolder args(index_mode);
  args.setDeviceIndex(device_index);
  args.push(inputs);

  return args;
}

// Push a tensor to the arguments
void KernelArgumentHolder::push(const at::Tensor& tensor) {
  changed_ = true;
  if (is_cpu_scalar(tensor)) {
    switch (tensor.scalar_type()) {
      case c10::ScalarType::ComplexDouble:
        arguments_.push_back(std::make_unique<CpuScalarTensorArg<
                                 CpuScalarTensorCodegen<c10::complex<double>>>>(
            tensor.data_ptr<c10::complex<double>>()[0]));
        break;
      case c10::ScalarType::ComplexFloat:
        arguments_.push_back(std::make_unique<CpuScalarTensorArg<
                                 CpuScalarTensorCodegen<c10::complex<float>>>>(
            tensor.data_ptr<c10::complex<float>>()[0]));
        break;
      case c10::ScalarType::Double:
        arguments_.push_back(
            std::make_unique<
                CpuScalarTensorArg<CpuScalarTensorCodegen<double>>>(
                tensor.data_ptr<double>()[0]));
        break;
      case c10::ScalarType::Float:
        arguments_.push_back(
            std::make_unique<CpuScalarTensorArg<CpuScalarTensorCodegen<float>>>(
                tensor.data_ptr<float>()[0]));
        break;
      case c10::ScalarType::Half:
        arguments_.push_back(
            std::make_unique<
                CpuScalarTensorArg<CpuScalarTensorCodegen<at::Half>>>(
                tensor.data_ptr<at::Half>()[0]));
        break;
      case c10::ScalarType::BFloat16:
        arguments_.push_back(
            std::make_unique<
                CpuScalarTensorArg<CpuScalarTensorCodegen<at::BFloat16>>>(
                tensor.data_ptr<at::BFloat16>()[0]));
        break;
      case c10::ScalarType::Bool:
        arguments_.push_back(
            std::make_unique<CpuScalarTensorArg<CpuScalarTensorCodegen<bool>>>(
                tensor.data_ptr<bool>()[0]));
        break;
      case c10::ScalarType::Long:
        arguments_.push_back(
            std::make_unique<
                CpuScalarTensorArg<CpuScalarTensorCodegen<int64_t>>>(
                tensor.data_ptr<int64_t>()[0]));
        break;
      case c10::ScalarType::Int:
        arguments_.push_back(
            std::make_unique<
                CpuScalarTensorArg<CpuScalarTensorCodegen<int32_t>>>(
                tensor.data_ptr<int32_t>()[0]));
        break;
      default:
        TORCH_CHECK(
            false,
            "Dtype: ",
            tensor.scalar_type(),
            " not currently supported in code generated kernels.");
    }
  } else {
    int nDims = tensor.ndimension();

    c10::ScalarType dtype = tensor.scalar_type();
    std::unique_ptr<TensorArgAbstract> tensor_arg =
        getTensorArg(dtype, nDims, index_mode_);
    tensor_arg->setTensor(tensor);
    tensor_arg->setPointer(tensor.data_ptr());
    tensor_arg->setDataType(aten_to_data_type(dtype));
    for (const auto i : c10::irange(nDims)) {
      tensor_arg->setSize(i, tensor.sizes()[i]);
      tensor_arg->setStride(i, tensor.strides()[i]);
    }
    arguments_.push_back(std::move(tensor_arg));
  }
}

// Push a scalar or integer to the arguments
void KernelArgumentHolder::push(const IValue& val) {
  changed_ = true;
  TORCH_INTERNAL_ASSERT(
      val.isScalar(),
      "Tried to push an arg to run in a fused kernel, expected a scalar but got, ",
      val);
  auto scalar_val = val.toScalar();
  switch (scalar_val.type()) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case c10::ScalarType::ComplexDouble:
      arguments_.push_back(
          std::make_unique<ComplexDoubleArg>(scalar_val.toComplexDouble()));
      return;
    case c10::ScalarType::Double:
      arguments_.push_back(std::make_unique<DoubleArg>(scalar_val.toDouble()));
      return;
    case c10::ScalarType::Long:
      arguments_.push_back(std::make_unique<LongArg>(scalar_val.toLong()));
      return;
    case c10::ScalarType::Bool:
      arguments_.push_back(std::make_unique<BoolArg>(scalar_val.toBool()));
      return;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          " Tried to create argument to send to a fused kernel, but got an unexpected type.");
  }
  TORCH_INTERNAL_ASSERT(
      false,
      " Tried to create argument to send to a fused kernel, but got a non-scalar type.");
}

void KernelArgumentHolder::push(int64_t val) {
  arguments_.push_back(std::make_unique<LongArg>(val));
}

void KernelArgumentHolder::push(const at::PhiloxCudaState& val) {
  arguments_.push_back(std::make_unique<PhiloxCudaStateArg>(val));
}

// Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
// in the buffer
void** KernelArgumentHolder::getBuffer() {
  if (changed_) {
    void_ptrs_ = std::vector<void*>(arguments_.size(), nullptr);
    for (const auto i : c10::irange(arguments_.size())) {
      void_ptrs_[i] = static_cast<void*>(arguments_[i]->arg());
    }
    changed_ = false;
  }
  return void_ptrs_.data();
}

void KernelArgumentHolder::push(const c10::ArrayRef<c10::IValue>& args) {
  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (const auto& arg : args) {
    if (arg.isTensor()) {
      push(arg.toTensor());
    } else {
      push(arg);
    }
  }
}

void KernelArgumentHolder::push(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    push(tensor);
  }
}

void KernelArgumentHolder::push(const ArgAbstract* arg) {
  changed_ = true;
  arguments_.emplace_back(arg->copy_unique_ptr());
}

void KernelArgumentHolder::swap(int i, const ArgAbstract* arg) {
  changed_ = true;
  auto holder = arg->copy_unique_ptr();
  arguments_[i].swap(holder);
}

void KernelArgumentHolder::appendPhiloxRNGSeed(uint64_t rand_offset) {
  at::PhiloxCudaState philox_engine_inputs;
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    philox_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(
            rand_offset);
  }
  push(philox_engine_inputs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
