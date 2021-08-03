#include <array>

#include <ATen/Functions.h>
#include <ATen/Utils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

namespace at {

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

// From build/ATen/RegisterCPU.cpp
TORCH_API at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  return self.add(other, alpha);
}

namespace native {
TORCH_API at::Tensor add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  return self.add(other, alpha);
}
} // namespace native


// out version
struct structured_add_out_out final : public at::native::structured_add_out {
    structured_add_out_out(Tensor& out0) : outputs_{ std::ref(out0) } {}

    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options, DimnameList names) override {

        const auto& out = outputs_[output_idx].get();
        TORCH_CHECK(options.dtype() == out.dtype(),
            "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
        TORCH_CHECK(options.device() == out.device(),
            "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
        bool resized = at::native::resize_output(outputs_[output_idx], sizes);
        // Only restride if a resize occurred; otherwise we ignore the (advisory)
        // strides from the meta function and directly use the output tensor's
        // preexisting strides
        if (resized) {
            if (!strides.empty()) {
                TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
                at::native::as_strided_(outputs_[output_idx], sizes, strides);
            } else if (options.memory_format_opt().has_value()) {
                outputs_[output_idx].get().unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
            }
        }

        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_add_out::set_output(output_idx, sizes, strides, options, names);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor & wrapper_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  structured_add_out_out op(out);
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.outputs_[0]);
  return out;
}

TORCH_API at::Tensor & add_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  return wrapper_add_out_out(self, other, alpha, out);
}

namespace native {

TORCH_API at::Tensor & add_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  return wrapper_add_out_out(self, other, alpha, out);
}
} // namespace native

} // namespace at
