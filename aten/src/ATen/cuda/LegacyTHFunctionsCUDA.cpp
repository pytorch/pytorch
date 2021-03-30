#include <ATen/LegacyTHFunctionsCUDA.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/ExpandUtils.h>
#include <THC/THC.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/THCUNN.h>
#undef THNN_
#undef THCIndexTensor_
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {
namespace legacy {
namespace cuda {

namespace {
  ScalarType infer_scalar_type(const Tensor & t) {
    return t.scalar_type();
  }
  ScalarType infer_scalar_type(const TensorList & tl) {
    TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
    return tl[0].scalar_type();
  }

  TensorOptions options(ScalarType s) {
    return TensorOptions().dtype(s)
                          .device(DeviceType::CUDA)
                          .layout(kStrided);
  }

  Allocator* allocator() {
    return at::cuda::getCUDADeviceAllocator();
  }
}
Tensor & _th_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Bool: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaBoolTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto index_ = checked_dense_tensor_unwrap(index, "index", 2, "_th_put_", false, DeviceType::CUDA, ScalarType::Long);
            auto source_ = checked_dense_tensor_unwrap(source, "source", 3, "_th_put_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_put(globalContext().getTHCState(), self_, index_, source_, accumulate);
            break;
        }
        default:
            AT_ERROR("_th_put_ not supported on CUDAType for ", dispatch_scalar_type);
    }
    return self;
}
std::tuple<Tensor &,Tensor &> _th_mode_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Char: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Double: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Float: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Int: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Long: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Short: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Half: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_mode_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        default:
            AT_ERROR("_th_mode_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> _th_mode(const Tensor & self, int64_t dim, bool keepdim) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto values_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto values = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(values_));
    auto indices_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(ScalarType::Long)).release();
    auto indices = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(indices_));
    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_mode", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_mode(globalContext().getTHCState(), values_, indices_, self_, dim, keepdim);
            break;
        }
        default:
            AT_ERROR("_th_mode not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor &,Tensor &> _th_sort_out_stable(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    TORCH_INTERNAL_ASSERT(stable.has_value(), "sort_out(): c10::optional<bool> for stable has to have value.");
    TORCH_CHECK(!stable.value(), "stable=True is not implemented on CUDA yet.");

    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Char: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Double: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Float: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Int: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Long: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Short: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Half: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_sort_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        default:
            AT_ERROR("_th_sort_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor &,Tensor &> _th_sort_out(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
  return at::native::legacy::cuda::_th_sort_out_stable(self, /*stable=*/false, dim, descending, values, indices);
}
std::tuple<Tensor,Tensor> _th_sort_stable(const Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
    // DeviceGuard omitted

    TORCH_INTERNAL_ASSERT(stable.has_value(), "sort_out(): c10::optional<bool> for stable has to have value.");
    TORCH_CHECK(!stable.value(), "stable=True is not implemented on CUDA yet.");

    auto dispatch_scalar_type = infer_scalar_type(self);
    auto values_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto values = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(values_));
    auto indices_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(ScalarType::Long)).release();
    auto indices = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(indices_));
    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_sort", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_sort(globalContext().getTHCState(), values_, indices_, self_, dim, descending);
            break;
        }
        default:
            AT_ERROR("_th_sort not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(values, indices);
}
std::tuple<Tensor,Tensor> _th_sort(const Tensor & self, int64_t dim, bool descending) {
  return _th_sort_stable(self, /*stable=*/false, dim, descending);
}
std::tuple<Tensor &,Tensor &> _th_topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Char: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Double: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Float: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Int: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Long: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Short: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Half: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::BFloat16: {
            auto values_ = checked_dense_tensor_unwrap(values, "values", 0, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto indices_ = checked_dense_tensor_unwrap(indices, "indices", 0, "_th_topk_out", false, DeviceType::CUDA, ScalarType::Long);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaBFloat16Tensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        default:
            AT_ERROR("_th_topk_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(values, indices);
}
std::tuple<Tensor,Tensor> _th_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto values_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto values = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(values_));
    auto indices_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(ScalarType::Long)).release();
    auto indices = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(indices_));
    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_topk", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaBFloat16Tensor_topk(globalContext().getTHCState(), values_, indices_, self_, k, dim, largest, sorted);
            break;
        }
        default:
            AT_ERROR("_th_topk not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(values, indices);
}
Tensor & _th_renorm_out(const Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm, Tensor & result) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_renorm_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toDouble();
            auto maxnorm_ = maxnorm.toDouble();
            THCudaDoubleTensor_renorm(globalContext().getTHCState(), result_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_renorm_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toFloat();
            auto maxnorm_ = maxnorm.toFloat();
            THCudaTensor_renorm(globalContext().getTHCState(), result_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Half: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_renorm_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toHalf();
            auto maxnorm_ = maxnorm.toHalf();
            THCudaHalfTensor_renorm(globalContext().getTHCState(), result_, self_, p_, dim, maxnorm_);
            break;
        }
        default:
            AT_ERROR("_th_renorm_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_renorm(const Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toDouble();
            auto maxnorm_ = maxnorm.toDouble();
            THCudaDoubleTensor_renorm(globalContext().getTHCState(), result_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toFloat();
            auto maxnorm_ = maxnorm.toFloat();
            THCudaTensor_renorm(globalContext().getTHCState(), result_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toHalf();
            auto maxnorm_ = maxnorm.toHalf();
            THCudaHalfTensor_renorm(globalContext().getTHCState(), result_, self_, p_, dim, maxnorm_);
            break;
        }
        default:
            AT_ERROR("_th_renorm not supported on CUDAType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor & _th_renorm_(Tensor & self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toDouble();
            auto maxnorm_ = maxnorm.toDouble();
            THCudaDoubleTensor_renorm(globalContext().getTHCState(), self_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toFloat();
            auto maxnorm_ = maxnorm.toFloat();
            THCudaTensor_renorm(globalContext().getTHCState(), self_, self_, p_, dim, maxnorm_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_renorm_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto p_ = p.toHalf();
            auto maxnorm_ = maxnorm.toHalf();
            THCudaHalfTensor_renorm(globalContext().getTHCState(), self_, self_, p_, dim, maxnorm_);
            break;
        }
        default:
            AT_ERROR("_th_renorm_ not supported on CUDAType for ", dispatch_scalar_type);
    }
    return self;
}
Tensor & _th_cross_kernel_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Char: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Int: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Long: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Short: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Half: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        default:
            AT_ERROR("_th_cross_kernel_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_cross_kernel(const Tensor & self, const Tensor & other, int64_t dim) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            auto other_ = checked_dense_tensor_unwrap(other, "other", 2, "_th_cross_kernel", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_crossKernel(globalContext().getTHCState(), result_, self_, other_, dim);
            break;
        }
        default:
            AT_ERROR("_th_cross_kernel not supported on CUDAType for ", dispatch_scalar_type);
    }
    return result;
}

std::tuple<Tensor &,Tensor &> _th_gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
            break;
        }
        case ScalarType::Float: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
            break;
        }
        default:
            AT_ERROR("_th_gels_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto res1_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res1 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res1_));
    auto res2_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res2 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res2_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_gels(globalContext().getTHCState(), res1_, res2_, self_, A_);
            break;
        }
        default:
            AT_ERROR("_th_gels not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(res1, res2);
}
std::tuple<Tensor &,Tensor &> _th_geqrf_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_geqrf_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_geqrf_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_geqrf(globalContext().getTHCState(), res1_, res2_, self_);
            break;
        }
        case ScalarType::Float: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_geqrf_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_geqrf_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_geqrf(globalContext().getTHCState(), res1_, res2_, self_);
            break;
        }
        default:
            AT_ERROR("_th_geqrf_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(res1, res2);
}
std::tuple<Tensor,Tensor> _th_geqrf(const Tensor & self) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto res1_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res1 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res1_));
    auto res2_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res2 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res2_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_geqrf(globalContext().getTHCState(), res1_, res2_, self_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_geqrf", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_geqrf(globalContext().getTHCState(), res1_, res2_, self_);
            break;
        }
        default:
            AT_ERROR("_th_geqrf not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(res1, res2);
}
Tensor & _th_copy_ignoring_overlaps_(Tensor & self, const Tensor & src) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Byte: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaByteTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        case ScalarType::Char: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaCharTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaDoubleTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        case ScalarType::Int: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaIntTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        case ScalarType::Long: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaLongTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        case ScalarType::Short: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaShortTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto src_ = checked_dense_tensor_unwrap(src, "src", 2, "_th_copy_ignoring_overlaps_", false, DeviceType::CUDA, dispatch_scalar_type);
            THCudaHalfTensor_copyIgnoringOverlaps(globalContext().getTHCState(), self_, src_);
            break;
        }
        default:
            AT_ERROR("_th_copy_ignoring_overlaps_ not supported on CUDAType for ", dispatch_scalar_type);
    }
    return self;
}
Tensor & _thnn_multi_margin_loss_forward_out(const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_multi_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        default:
            AT_ERROR("_thnn_multi_margin_loss_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor _thnn_multi_margin_loss_forward(const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const c10::optional<Tensor>& weight_opt, int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multi_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 5, "_thnn_multi_margin_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfMultiMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        default:
            AT_ERROR("_thnn_multi_margin_loss_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor & _thnn_multi_margin_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor & grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_multi_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        default:
            AT_ERROR("_thnn_multi_margin_loss_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor _thnn_multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Scalar& p, const Scalar& margin, const c10::optional<Tensor>& weight_opt, int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multi_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto p_ = p.toDouble();
            auto margin_ = margin.toDouble();
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 6, "_thnn_multi_margin_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfMultiMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, p_, weight_ ? weight_ : NULL, margin_);
            break;
        }
        default:
            AT_ERROR("_thnn_multi_margin_loss_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
std::tuple<Tensor &,Tensor &> _thnn_multilabel_margin_loss_forward_out(const Tensor & self, const Tensor & target, int64_t reduction, Tensor & output, Tensor & is_target) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 3, "_thnn_multilabel_margin_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16MultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        default:
            AT_ERROR("_thnn_multilabel_margin_loss_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(output, is_target);
}
std::tuple<Tensor,Tensor> _thnn_multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    auto is_target_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto is_target = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(is_target_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            THNN_CudaDoubleMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            THNN_CudaMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            THNN_CudaHalfMultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_multilabel_margin_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            THNN_CudaBFloat16MultiLabelMarginCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, is_target_, reduction);
            break;
        }
        default:
            AT_ERROR("_thnn_multilabel_margin_loss_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(output, is_target);
}
Tensor & _thnn_multilabel_margin_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target, Tensor & grad_input) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 5, "_thnn_multilabel_margin_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16MultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        default:
            AT_ERROR("_thnn_multilabel_margin_loss_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor _thnn_multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfMultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto is_target_ = checked_dense_tensor_unwrap(is_target, "is_target", 5, "_thnn_multilabel_margin_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16MultiLabelMarginCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, is_target_, reduction);
            break;
        }
        default:
            AT_ERROR("_thnn_multilabel_margin_loss_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
std::tuple<Tensor &,Tensor &> _thnn_nll_loss_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16ClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> _thnn_nll_loss_forward(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    auto total_weight_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto total_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(total_weight_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16ClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & _thnn_nll_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor & total_weight, Tensor & grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16ClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor _thnn_nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16ClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
std::tuple<Tensor &,Tensor &> _thnn_nll_loss2d_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 5, "_thnn_nll_loss2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16SpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss2d_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(output, total_weight);
}
std::tuple<Tensor,Tensor> _thnn_nll_loss2d_forward(const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    auto total_weight_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto total_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(total_weight_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfSpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 2, "_thnn_nll_loss2d_forward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_nll_loss2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16SpatialClassNLLCriterion_updateOutput(globalContext().getTHCState(), self_, target_, output_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss2d_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(output, total_weight);
}
Tensor & _thnn_nll_loss2d_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor & total_weight, Tensor & grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_nll_loss2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16SpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss2d_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor _thnn_nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor>& weight_opt, int64_t reduction, int64_t ignore_index, const Tensor & total_weight) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfSpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto target_ = checked_dense_tensor_unwrap(target, "target", 3, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, ScalarType::Long);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 4, "_thnn_nll_loss2d_backward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto total_weight_ = checked_dense_tensor_unwrap(total_weight, "total_weight", 7, "_thnn_nll_loss2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16SpatialClassNLLCriterion_updateGradInput(globalContext().getTHCState(), self_, target_, grad_output_, grad_input_, reduction, weight_ ? weight_ : NULL, total_weight_, ignore_index);
            break;
        }
        default:
            AT_ERROR("_thnn_nll_loss2d_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor & _thnn_glu_forward_out(const Tensor & self, int64_t dim, Tensor & output) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 2, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 2, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 2, "_thnn_glu_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
            break;
        }
        default:
            AT_ERROR("_thnn_glu_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor _thnn_glu_forward(const Tensor & self, int64_t dim) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_glu_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfGatedLinear_updateOutput(globalContext().getTHCState(), self_, output_, dim);
            break;
        }
        default:
            AT_ERROR("_thnn_glu_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor & _thnn_glu_backward_out(const Tensor & grad_output, const Tensor & self, int64_t dim, Tensor & grad_input) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_glu_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
            break;
        }
        default:
            AT_ERROR("_thnn_glu_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor _thnn_glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_glu_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfGatedLinear_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, dim);
            break;
        }
        default:
            AT_ERROR("_thnn_glu_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
std::tuple<Tensor &,Tensor &> _thnn_log_sigmoid_forward_out(const Tensor & self, Tensor & output, Tensor & buffer) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 1, "_thnn_log_sigmoid_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
            break;
        }
        default:
            AT_ERROR("_thnn_log_sigmoid_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(output, buffer);
}
std::tuple<Tensor,Tensor> _thnn_log_sigmoid_forward(const Tensor & self) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    auto buffer_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto buffer = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(buffer_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_log_sigmoid_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfLogSigmoid_updateOutput(globalContext().getTHCState(), self_, output_, buffer_);
            break;
        }
        default:
            AT_ERROR("_thnn_log_sigmoid_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(output, buffer);
}
Tensor & _thnn_log_sigmoid_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & buffer, Tensor & grad_input) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 3, "_thnn_log_sigmoid_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
            break;
        }
        default:
            AT_ERROR("_thnn_log_sigmoid_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor _thnn_log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto buffer_ = checked_dense_tensor_unwrap(buffer, "buffer", 3, "_thnn_log_sigmoid_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfLogSigmoid_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_, buffer_);
            break;
        }
        default:
            AT_ERROR("_thnn_log_sigmoid_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return grad_input;
}
Tensor & _thnn_rrelu_with_noise_forward_out(const Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training, c10::optional<at::Generator> generator, Tensor & output) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleRReLU_updateOutput(globalContext().getTHCState(), self_, output_, noise_, lower_, upper_, training, false, generator);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaRReLU_updateOutput(globalContext().getTHCState(), self_, output_, noise_, lower_, upper_, training, false, generator);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_rrelu_with_noise_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfRReLU_updateOutput(globalContext().getTHCState(), self_, output_, noise_, lower_, upper_, training, false, generator);
            break;
        }
        default:
            AT_ERROR("_thnn_rrelu_with_noise_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor _thnn_rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training, c10::optional<at::Generator> generator) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            THNN_CudaDoubleRReLU_updateOutput(globalContext().getTHCState(), self_, output_, noise_, lower_, upper_, training, false, generator);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            THNN_CudaRReLU_updateOutput(globalContext().getTHCState(), self_, output_, noise_, lower_, upper_, training, false, generator);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            THNN_CudaHalfRReLU_updateOutput(globalContext().getTHCState(), self_, output_, noise_, lower_, upper_, training, false, generator);
            break;
        }
        default:
            AT_ERROR("_thnn_rrelu_with_noise_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor & _thnn_rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, const Scalar& lower, const Scalar& upper, bool training, c10::optional<at::Generator> generator) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            THNN_CudaDoubleRReLU_updateOutput(globalContext().getTHCState(), self_, self_, noise_, lower_, upper_, training, true, generator);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            THNN_CudaRReLU_updateOutput(globalContext().getTHCState(), self_, self_, noise_, lower_, upper_, training, true, generator);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_rrelu_with_noise_forward_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto noise_ = checked_dense_tensor_unwrap(noise, "noise", 2, "_thnn_rrelu_with_noise_forward_", false, DeviceType::CUDA, dispatch_scalar_type);
            auto lower_ = lower.toDouble();
            auto upper_ = upper.toDouble();
            THNN_CudaHalfRReLU_updateOutput(globalContext().getTHCState(), self_, self_, noise_, lower_, upper_, training, true, generator);
            break;
        }
        default:
            AT_ERROR("_thnn_rrelu_with_noise_forward_ not supported on CUDAType for ", dispatch_scalar_type);
    }
    return self;
}
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv2d_forward_out(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, Tensor & output, Tensor & columns, Tensor & ones) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 6, "_thnn_conv2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16SpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        default:
            AT_ERROR("_thnn_conv2d_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &, Tensor &>(output, columns, ones);
}
std::tuple<Tensor,Tensor,Tensor> _thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    auto columns_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto columns = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(columns_));
    auto ones_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto ones = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(ones_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            THNN_CudaDoubleSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            THNN_CudaSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            THNN_CudaHalfSpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            THNN_CudaBFloat16SpatialConvolutionMM_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            break;
        }
        default:
            AT_ERROR("_thnn_conv2d_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor, Tensor>(output, columns, ones);
}
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & columns, const Tensor & ones) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaDoubleSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaHalfSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_bias_ = checked_dense_tensor_unwrap(grad_bias, "grad_bias", 8, "_thnn_conv2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaBFloat16SpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaBFloat16SpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        default:
            AT_ERROR("_thnn_conv2d_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &, Tensor &>(grad_input, grad_weight, grad_bias);
}
std::tuple<Tensor,Tensor,Tensor> _thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = output_mask[0] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_input_));
    auto grad_weight_ = output_mask[1] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
    auto grad_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_weight_));
    auto grad_bias_ = output_mask[2] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
    auto grad_bias = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_bias_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_bias_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaDoubleSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaDoubleSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaHalfSpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaHalfSpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto columns_ = checked_dense_tensor_unwrap(columns, "columns", 7, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto ones_ = checked_dense_tensor_unwrap(ones, "ones", 8, "_thnn_conv2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaBFloat16SpatialConvolutionMM_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0]);
            if (grad_weight_ || grad_bias_) THNN_CudaBFloat16SpatialConvolutionMM_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, grad_bias_ ? grad_bias_ : NULL, columns_, ones_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], 1);
            break;
        }
        default:
            AT_ERROR("_thnn_conv2d_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}
Tensor & _thnn_conv_depthwise2d_forward_out(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor & output) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaHalfSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto output_ = checked_dense_tensor_unwrap(output, "output", 7, "_thnn_conv_depthwise2d_forward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            THNN_CudaBFloat16SpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        default:
            AT_ERROR("_thnn_conv_depthwise2d_forward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
Tensor _thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const c10::optional<Tensor>& bias_opt, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto output_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto output = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(output_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            THNN_CudaDoubleSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            THNN_CudaSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Half: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            THNN_CudaHalfSpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::BFloat16: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 2, "_thnn_conv_depthwise2d_forward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 3);
            auto bias_ = checked_dense_tensor_unwrap(bias, "bias", 4, "_thnn_conv_depthwise2d_forward", true, DeviceType::CUDA, dispatch_scalar_type);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            THNN_CudaBFloat16SpatialDepthwiseConvolution_updateOutput(globalContext().getTHCState(), self_, output_, weight_, bias_ ? bias_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        default:
            AT_ERROR("_thnn_conv_depthwise2d_forward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return output;
}
std::tuple<Tensor &,Tensor &> _thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaHalfSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaHalfSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward_out", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            auto grad_input_ = checked_dense_tensor_unwrap(grad_input, "grad_input", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            auto grad_weight_ = checked_dense_tensor_unwrap(grad_weight, "grad_weight", 7, "_thnn_conv_depthwise2d_backward_out", true, DeviceType::CUDA, dispatch_scalar_type);
            if (grad_input_) THNN_CudaBFloat16SpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaBFloat16SpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        default:
            AT_ERROR("_thnn_conv_depthwise2d_backward_out not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor &, Tensor &>(grad_input, grad_weight);
}
std::tuple<Tensor,Tensor> _thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask) {
    const OptionalDeviceGuard device_guard(device_of(self));
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto grad_input_ = output_mask[0] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
    auto grad_input = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_input_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_input_));
    auto grad_weight_ = output_mask[1] ? c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CUDA, scalarTypeToTypeMeta(dispatch_scalar_type)).release() : nullptr;
    auto grad_weight = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(grad_weight_ == nullptr ? (TensorImpl*)UndefinedTensorImpl::singleton() : (TensorImpl*)grad_weight_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            if (grad_input_) THNN_CudaDoubleSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaDoubleSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Float: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            if (grad_input_) THNN_CudaSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::Half: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            if (grad_input_) THNN_CudaHalfSpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaHalfSpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        case ScalarType::BFloat16: {
            auto grad_output_ = checked_dense_tensor_unwrap(grad_output, "grad_output", 1, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 2, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto weight_ = checked_dense_tensor_unwrap(weight, "weight", 3, "_thnn_conv_depthwise2d_backward", false, DeviceType::CUDA, dispatch_scalar_type);
            auto kernel_size_ = check_intlist<2>(kernel_size, "kernel_size", 4);
            auto stride_ = check_intlist<2>(stride, "stride", 5);
            auto padding_ = check_intlist<2>(padding, "padding", 6);
            auto dilation_ = check_intlist<2>(dilation, "dilation", 7);
            if (grad_input_) THNN_CudaBFloat16SpatialDepthwiseConvolution_updateGradInput(globalContext().getTHCState(), self_, grad_output_, grad_input_ ? grad_input_ : NULL, weight_, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            if (grad_weight_) THNN_CudaBFloat16SpatialDepthwiseConvolution_accGradParameters(globalContext().getTHCState(), self_, grad_output_, grad_weight_ ? grad_weight_ : NULL, kernel_size_[1], kernel_size_[0], stride_[1], stride_[0], padding_[1], padding_[0], dilation_[1], dilation_[0]);
            break;
        }
        default:
            AT_ERROR("_thnn_conv_depthwise2d_backward not supported on CUDAType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(grad_input, grad_weight);
}

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
