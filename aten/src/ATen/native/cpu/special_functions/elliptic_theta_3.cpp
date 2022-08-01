namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
static void elliptic_theta_3_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "elliptic_theta_3_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return special_functions::elliptic_theta_3(x);
    });
  });
} // void elliptic_theta_3_cpu_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY
REGISTER_DISPATCH(special_elliptic_theta_3_stub, &CPU_CAPABILITY::elliptic_theta_3_cpu_kernel);
} // namespace native
} // namespace at
