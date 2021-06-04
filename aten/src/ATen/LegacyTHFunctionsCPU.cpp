#include <ATen/LegacyTHFunctionsCPU.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ExpandUtils.h>
#include <TH/TH.h>
#include <TH/THTensor.hpp>


namespace at {
namespace native {
namespace legacy {
namespace cpu {

namespace {
  ScalarType infer_scalar_type(const Tensor & t) {
    return t.scalar_type();
  }
  // NOLINTNEXTLINE(clang-diagnostic-unused-function)
  ScalarType infer_scalar_type(const TensorList & tl) {
    TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
    return tl[0].scalar_type();
  }

  TensorOptions options(ScalarType s) {
    return TensorOptions().dtype(s)
                          .device(DeviceType::CPU)
                          .layout(kStrided);
  }

  Allocator* allocator() {
    return getCPUAllocator();
  }
}

Tensor & _th_histc_out(const Tensor & self, int64_t bins, const Scalar& min, const Scalar& max, Tensor & result) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);

    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toDouble();
            auto max_ = max.toDouble();
            THDoubleTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        case ScalarType::Float: {
            auto result_ = checked_dense_tensor_unwrap(result, "result", 0, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toFloat();
            auto max_ = max.toFloat();
            THFloatTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        default:
            AT_ERROR("_th_histc_out not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
}
Tensor _th_histc(const Tensor & self, int64_t bins, const Scalar& min, const Scalar& max) {
    // DeviceGuard omitted
    auto dispatch_scalar_type = infer_scalar_type(self);
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto result = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(result_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toDouble();
            auto max_ = max.toDouble();
            THDoubleTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_histc", false, DeviceType::CPU, dispatch_scalar_type);
            auto min_ = min.toFloat();
            auto max_ = max.toFloat();
            THFloatTensor_histc(result_, self_, bins, min_, max_);
            break;
        }
        default:
            AT_ERROR("_th_histc not supported on CPUType for ", dispatch_scalar_type);
    }
    return result;
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
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        case ScalarType::Float: {
            auto res1_ = checked_dense_tensor_unwrap(res1, "res1", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto res2_ = checked_dense_tensor_unwrap(res2, "res2", 0, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels_out", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        default:
            AT_ERROR("_th_gels_out not supported on CPUType for ", dispatch_scalar_type);
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
    auto res1_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res1 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res1_));
    auto res2_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(c10::Storage(c10::Storage::use_byte_size_t(), 0, allocator(), true),DispatchKey::CPU, scalarTypeToTypeMeta(dispatch_scalar_type)).release();
    auto res2 = Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(res2_));
    switch (dispatch_scalar_type) {
        case ScalarType::Double: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            THDoubleTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        case ScalarType::Float: {
            auto self_ = checked_dense_tensor_unwrap(self, "self", 1, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            auto A_ = checked_dense_tensor_unwrap(A, "A", 2, "_th_gels", false, DeviceType::CPU, dispatch_scalar_type);
            THFloatTensor_gels(res1_, res2_, self_, A_);
            break;
        }
        default:
            AT_ERROR("_th_gels not supported on CPUType for ", dispatch_scalar_type);
    }
    return std::tuple<Tensor, Tensor>(res1, res2);
}

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
