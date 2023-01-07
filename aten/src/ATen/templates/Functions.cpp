#include <array>

#include <ATen/Functions.h>
#include <ATen/Utils.h>

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

  TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  if (strides_) {
    tensor_impl->set_sizes_and_strides(sizes_, *strides_);
  } else {
    tensor_impl->set_sizes_contiguous(sizes_);
  }
  if (storage_offset_) {
    tensor_impl->set_storage_offset(*storage_offset_);
  }

   return tensor;
 }

 std::size_t TensorMaker::computeStorageSize() const noexcept {
   std::size_t itemsize = opts_.dtype().itemsize();

   if (strides_) {
     auto storage_size = detail::computeStorageNbytes(sizes_, *strides_, itemsize);
     if (storage_offset_) {
       storage_size += storage_offset_.value();
     }
     return storage_size;
   }

   std::size_t size = 1;
   for (std::int64_t s : sizes_) {
     size *= static_cast<std::size_t>(s);
   }
   auto storage_size = size * itemsize;
   if (storage_offset_) {
     storage_size += storage_offset_.value();
   }
   return storage_size;
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

} // namespace at
