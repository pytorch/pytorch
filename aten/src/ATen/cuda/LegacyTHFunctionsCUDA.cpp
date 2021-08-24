#include <ATen/LegacyTHFunctionsCUDA.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/ExpandUtils.h>
#include <THC/THC.h>
#include <THC/THCTensor.hpp>
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

std::tuple<Tensor &,Tensor &> _th_gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) {
    TORCH_WARN_ONCE(
      "torch.lstsq is deprecated in favor of torch.linalg.lstsq and will be removed in a future PyTorch release.\n",
      "torch.linalg.lstsq has reversed arguments and does not return the QR decomposition in "
      "the returned tuple (although it returns other information about the problem).\n",
      "To get the qr decomposition consider using torch.linalg.qr.\n",
      "The returned solution in torch.lstsq stored the residuals of the solution in the ",
      "last m - n columns of the returned value whenever m > n. In torch.linalg.lstsq, the ",
      "residuals in the field 'residuals' of the returned named tuple.\n",
      "The unpacking of the solution, as in\n",
      "X, _ = torch.lstsq(B, A).solution[:A.size(1)]\n",
      "should be replaced with\n",
      "X = torch.linalg.lstsq(A, B).solution"
    );
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
    TORCH_WARN_ONCE(
      "torch.lstsq is deprecated in favor of torch.linalg.lstsq and will be removed in a future PyTorch release.\n",
      "torch.linalg.lstsq has reversed arguments and does not return the QR decomposition in "
      "the returned tuple (although it returns other information about the problem).\n",
      "To get the qr decomposition consider using torch.linalg.qr.\n",
      "The returned solution in torch.lstsq stored the residuals of the solution in the ",
      "last m - n columns of the returned value whenever m > n. In torch.linalg.lstsq, the ",
      "residuals in the field 'residuals' of the returned named tuple.\n",
      "The unpacking of the solution, as in\n",
      "X, _ = torch.lstsq(B, A).solution[:A.size(1)]\n",
      "should be replaced with\n",
      "X = torch.linalg.lstsq(A, B).solution"
    );
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

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
