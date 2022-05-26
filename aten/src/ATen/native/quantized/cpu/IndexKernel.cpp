#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/quantized/affine_quantizer_base.h>
#include <c10/core/Scalar.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {

namespace {
template <typename scalar_t, typename mask_t>
void cpu_masked_fill_kernel_quantized_cpu(TensorIterator& iter, float value, double scale, int zero_point) {
  auto is_mask_bool = std::is_same<mask_t, bool>::value;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* mask = data[1];
    for (const auto i : c10::irange(n)) {
      mask_t mask_value = *(mask_t*)(mask + strides[1] * i);
      if (!is_mask_bool) {
        TORCH_CHECK(mask_value == 0 || mask_value == 1, "Mask tensor can take 0 and 1 values only");
      }
      if (mask_value) {
        *(scalar_t*)(dst + strides[0] * i) = quantize_val<scalar_t>(scale, zero_point, value);
      }
    }
  };
  iter.for_each(loop);
}
}

void masked_fill_kernel_quantized_cpu(TensorIterator& iter, const Scalar& value, double scale, int zero_point) {
  AT_DISPATCH_QINT_TYPES(iter.dtype(), "masked_fill", [&] {
    float scalar_val = value.to<float>();
    auto mask_dtype = iter.input_dtype(0);
    if (mask_dtype == ScalarType::Bool) {
      cpu_masked_fill_kernel_quantized_cpu<scalar_t, bool>(iter, scalar_val, scale, zero_point);
    } else {
      cpu_masked_fill_kernel_quantized_cpu<scalar_t, unsigned char>(iter, scalar_val, scale, zero_point);
    }
  });
}

} // native
} // at
