#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cpu/IndexKernelUtils.h>
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

// currently, we do not support accumulate=True for quantized tensors. We throw an exception in _index_put_impl_quantized_cpu_
// However, we currently do have an implementation for accumulate=True in the below code, but I think accumulation for quantized tensors can be defined
// different ways since quantize(x + y) = quantize(x) + quantize(y) does not always hold. The code in this function implements the RHS, i.e., it quantizes the
// fp values and then accumluates the quantized values. The LHS first accumulates the fp values and then quantizes the accumulated fp value.
// TODO: decide on which one we should be supporting
void index_put_kernel_quantized_cpu(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate, double scale, int zero_point) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_QINT_TYPES(iter.dtype(), "index_put", [&] {
    // See Note [Enabling Deterministic Operations]
    // Parallel cpu_index_kernel with accumulation is nondeterministic, so we
    // must enable serial execution if deterministic algorithms are enabled.
    const bool is_deterministic = at::globalContext().deterministicAlgorithms();
    if (accumulate) {
      // TODO: can we do this atomically on ints like the fp variant did with cpu_atomic_add_float?
      at::native::cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [scale, zero_point](char* dst, char* src, int64_t offset) {
        // TODO: define operator+= that can add qdtypes
        *(underlying_t*)(dst + offset) += quantize_val<scalar_t>(scale, zero_point, *(float*)src).val_;
      }, /*serial_execution=*/true);
    } else {
      at::native::cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [scale, zero_point](char* dst, char* src, int64_t offset) {
        *(scalar_t*)(dst + offset) = quantize_val<scalar_t>(scale, zero_point, *(float*)src);
      }, /*serial_execution=*/is_deterministic);
    }
  });
}

} // native
} // at
