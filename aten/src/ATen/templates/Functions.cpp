// ${generated_comment}

#include <array>

#include <ATen/Functions.h>
#include <ATen/Utils.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/adaption.h>

${static_dispatch_extra_headers}

namespace at {

Tensor var(const Tensor& self, int dim) {
  return at::var(self, IntArrayRef{dim});
}

std::tuple<Tensor, Tensor> var_mean(const Tensor& self, int dim) {
  return at::var_mean(self, IntArrayRef{dim});
}

Tensor std(const Tensor& self, int dim) {
  return at::std(self, IntArrayRef{dim});
}

std::tuple<Tensor, Tensor> std_mean(const Tensor& self, int dim) {
  return at::std_mean(self, IntArrayRef{dim});
}

at::Tensor conv1d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    std::initializer_list<int64_t> padding_,
    IntArrayRef dilation,
    int64_t groups) {
  auto padding = IntArrayRef(padding_);
  return at::conv1d(input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    std::initializer_list<int64_t> padding_,
    IntArrayRef dilation,
    int64_t groups) {
  auto padding = IntArrayRef(padding_);
  return at::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

at::Tensor conv3d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    std::initializer_list<int64_t> padding_,
    IntArrayRef dilation,
    int64_t groups) {
  auto padding = IntArrayRef(padding_);
  return at::conv3d(input, weight, bias, stride, padding, dilation, groups);
}

namespace detail {

void noopDelete(void*) {}

} // namespace detail

Tensor TensorMaker::make_tensor() {
  AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
  tracer::impl::NoTracerDispatchMode tracer_guard{};

  check_size_nonnegative(sizes_);

  TORCH_CHECK_VALUE(
      !deleter_ || !ctx_,
      "The deleter and context arguments are mutually exclusive.");

  if (device_ == nullopt) {
    device_ = globalContext().getDeviceFromPtr(data_, opts_.device().type());
  }

  if (opts_.device().has_index()) {
    // clang-format off
    TORCH_CHECK_VALUE(
        opts_.device() == *device_,
        "Specified device ", opts_.device(), " does not match device of data ", *device_);
    // clang-format on
  }

  std::size_t size_bytes = computeStorageSize();

  DataPtr data_ptr{};
  if (deleter_) {
    data_ptr = makeDataPtrFromDeleter();
  } else {
    data_ptr = makeDataPtrFromContext();
  }

  Storage storage{Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr)};

  Tensor tensor = detail::make_tensor<TensorImpl>(
      std::move(storage), opts_.computeDispatchKey(), opts_.dtype());

  if (sizes_.size() != 1 || sizes_[0] != 0) {
    TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

    if (strides_) {
      tensor_impl->set_sizes_and_strides(sizes_, *strides_);
    } else {
      tensor_impl->set_sizes_contiguous(sizes_);
    }
  }

  return tensor;
}

std::size_t TensorMaker::computeStorageSize() const noexcept {
  std::size_t itemsize = opts_.dtype().itemsize();

  if (strides_) {
    return detail::computeStorageNbytes(sizes_, *strides_, itemsize);
  }

  std::size_t size = 1;
  for (std::int64_t s : sizes_) {
    size *= static_cast<std::size_t>(s);
  }
  return size * itemsize;
}

inline DataPtr TensorMaker::makeDataPtrFromDeleter() const {
  return InefficientStdFunctionContext::makeDataPtr(data_, deleter_, *device_);
}

inline DataPtr TensorMaker::makeDataPtrFromContext() noexcept {
  return DataPtr{data_, ctx_.release(), ctx_.get_deleter(), *device_};
}

IntArrayRef TensorMaker::makeTempSizes() const noexcept {
  static std::int64_t zeros[5] = {0, 0, 0, 0, 0};
  if (opts_.has_memory_format()) {
    MemoryFormat format = *opts_.memory_format_opt();
    if (format == MemoryFormat::ChannelsLast) {
      return IntArrayRef(zeros, 4);
    }
    if (format == MemoryFormat::ChannelsLast3d) {
      return IntArrayRef(zeros, 5);
    }
  }
  return IntArrayRef(zeros, 1);
}

${function_definitions}

} // namespace at
