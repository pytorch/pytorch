#include <ATen/native/optimizers.h>

#include <cmath>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at {
namespace native {

namespace {

template <typename T>
void SparseAdamStepKernelImpl(
    T alpha,
    T beta1,
    T beta2,
    T eps,
    int64_t step,
    TensorIterator* it) {
  TORCH_CHECK(it->is_contiguous());
  const T* grad_data = static_cast<T*>(it->data_ptr(3));
  const T* moment1_data = static_cast<T*>(it->data_ptr(4));
  const T* moment2_data = static_cast<T*>(it->data_ptr(5));
  T* adam_step_data = static_cast<T*>(it->data_ptr(0));
  T* moment1_step_data = static_cast<T*>(it->data_ptr(1));
  T* moment2_step_data = static_cast<T*>(it->data_ptr(2));
  const int64_t N = it->numel();
  constexpr int64_t K = vec256::Vec256<T>::size();
  const int64_t n = N / K * K;
  const T bias_correction1 = T(1) - std::pow(beta1, static_cast<T>(step));
  const T bias_correction2 = T(1) - std::pow(beta2, static_cast<T>(step));
  const T step_size = alpha * std::sqrt(bias_correction2) / bias_correction1;
  const vec256::Vec256<T> b1_vec(T(1) - beta1);
  const vec256::Vec256<T> b2_vec(T(1) - beta2);
  const vec256::Vec256<T> eps_vec(eps);
  const vec256::Vec256<T> step_size_vec(-step_size);
  for (int64_t i = 0; i < n; i += K) {
    const vec256::Vec256<T> g_vec = vec256::Vec256<T>::loadu(grad_data + i);
    const vec256::Vec256<T> m1_vec = vec256::Vec256<T>::loadu(moment1_data + i);
    const vec256::Vec256<T> m2_vec = vec256::Vec256<T>::loadu(moment2_data + i);
    const vec256::Vec256<T> m_vec = (g_vec - m1_vec) * b1_vec;
    const vec256::Vec256<T> v_vec = (g_vec * g_vec - m2_vec) * b2_vec;
    const vec256::Vec256<T> s_vec =
        (m_vec + m1_vec) / ((v_vec + m2_vec).sqrt() + eps_vec) * step_size_vec;
    s_vec.store(adam_step_data + i);
    m_vec.store(moment1_step_data + i);
    v_vec.store(moment2_step_data + i);
  }
  for (int64_t i = n; i < N; ++i) {
    const T m = (grad_data[i] - moment1_data[i]) * (T(1) - beta1);
    const T v =
        (grad_data[i] * grad_data[i] - moment2_data[i]) * (T(1) - beta2);
    adam_step_data[i] = -step_size * (m + moment1_data[i]) /
        (std::sqrt(v + moment2_data[i]) + eps);
    moment1_step_data[i] = m;
    moment2_step_data[i] = v;
  }
}

void SparseAdamStepKernel(
    double alpha,
    double beta1,
    double beta2,
    double eps,
    int64_t step,
    TensorIterator* it) {
  AT_DISPATCH_FLOATING_TYPES(it->dtype(), "SparseAdamStepKernel", [&]() {
    SparseAdamStepKernelImpl<scalar_t>(
        static_cast<scalar_t>(alpha),
        static_cast<scalar_t>(beta1),
        static_cast<scalar_t>(beta2),
        static_cast<scalar_t>(eps),
        step,
        it);
  });
}

} // namespace

REGISTER_DISPATCH(sparse_adam_step_stub, &SparseAdamStepKernel);

} // namespace native
} // namespace at
