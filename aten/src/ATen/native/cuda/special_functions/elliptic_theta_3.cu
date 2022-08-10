namespace at {
namespace native {
namespace {
const auto elliptic_theta_3_string = jiterator_stringify(
  template<typename T>
  T elliptic_theta_3(T x, T n) {
    return x;
  } // T elliptic_theta_3(T x, T n)
); // elliptic_theta_3_string

const char elliptic_theta_3_name[] = "elliptic_theta_3";

void elliptic_theta_3_cuda_kernel(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "elliptic_theta_3_cuda_kernel", [&]() {
    opmath_jitted_gpu_kernel_with_scalars<elliptic_theta_3_name, scalar_t, scalar_t>(iterator, elliptic_theta_3_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "elliptic_theta_3_cuda_kernel", [&]() {
    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      return special_functions::elliptic_theta_3<scalar_t, true>(x, y);
    });
  });
#endif
} // void elliptic_theta_3_cuda_kernel(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(elliptic_theta_3_stub, &elliptic_theta_3_cuda_kernel);
} // namespace native
} // namespace at
