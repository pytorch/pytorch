#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>
#include <c10/core/Scalar.h>

namespace at { namespace native {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  __device__ __forceinline__ scalar_t operator() () const {
    return value;
  }
  private:
    scalar_t value;
};

const char fill_name[] = "fill_kernel";
void fill_kernel_cuda(TensorIterator& iter, const Scalar& value) {
  auto dtype = iter.dtype();
  if(at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
  static const auto fill_string = jiterator_stringify(
      template<typename scalar_t>
      struct FillFunctor {
        FillFunctor(scalar_t v): value(v) {}
        __device__ __foceinline__ scalar_t operator() () const {
          return value;
        }
        private:
          scalar_t value;
      };
  ); // fill_string
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "fill_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ fill_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, fill_string);
  });
#else
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "fill_cuda", [&]() {
      gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
  });
#endif
  } else {
  AT_DISPATCH_ALL_TYPES_AND3(kBool, kHalf, kBFloat16, dtype, "fill_cuda", [&]() {
    gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
  });
  }
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_cuda);

} // namespace native
} // namespace at
