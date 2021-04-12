// ${generated_comment}

#include <ATen/Functions.h>

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

std::vector<Tensor> to(const std::vector<Tensor> tensors, Device device) {
    std::vector<Tensor> output_tensors;
    for (const auto& t : tensors) {
        output_tensors.push_back(t.to(device));
    }
    return output_tensors;
}

std::vector<c10::optional<at::Tensor>> to_cpu(const std::vector<c10::optional<at::Tensor>>& tensors) {
    std::vector<c10::optional<at::Tensor>> opt_cpu_tensors(tensors.size());
    std::vector<bool> copy_indices(tensors.size());
    std::vector<at::Tensor> valid_tensors;
    for (auto i = 0; i < tensors.size(); ++i) {
        if (tensors[i].has_value()) {
            valid_tensors.push_back(*tensors[i]);
            copy_indices[i] = true;
        } else {
            opt_cpu_tensors[i] = tensors[i];
        }
    }
    auto cpu_tensors = at::to_cpu(valid_tensors); // redispatch!

    int idx = 0;
    for (auto i = 0; i < tensors.size(); ++i) {
        if (copy_indices[i]) {
            opt_cpu_tensors[i] = c10::optional<at::Tensor>(cpu_tensors[idx++]);
        }
    }
    return opt_cpu_tensors;
}

namespace detail {

void noopDelete(void*)
{}

}  // namespace detail

Tensor TensorMaker::make_tensor() {
  AutoNonVariableTypeMode guard{}; // TODO: Remove.
  tracer::impl::NoTracerDispatchMode tracer_guard{};

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

  at::Tensor tensor = makeEmptyTensor();

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

  tensor_impl->set_storage_keep_dtype(std::move(storage));

  if (strides_) {
    tensor_impl->set_sizes_and_strides(sizes_, *strides_);
  } else {
    tensor_impl->set_sizes_contiguous(sizes_);
  }

  return tensor;
}

std::size_t TensorMaker::computeStorageSize() const noexcept {
  std::size_t itemsize = opts_.dtype().itemsize();

  if (strides_) {
    return detail::computeStorageNbytes(sizes_, *strides_, itemsize);
  }

  std::size_t size = 1;
  for (std::size_t s : sizes_) {
    size *= s;
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
  static int64_t zeros[5] = {0, 0, 0, 0, 0};
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

inline Tensor TensorMaker::makeEmptyTensor() const {
  return empty(makeTempSizes(), opts_);
}

${function_definitions}

} // namespace at
