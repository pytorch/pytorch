#include "ATen/native/cuda/Normalization.cuh"

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda(const Tensor& self, const Tensor& weight, const Tensor& bias,
                                                   const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "batch_norm", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_cuda_template<scalar_t, int32_t>(self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      } else {
        return batch_norm_cuda_template<scalar_t, int64_t>(self, weight, bias, running_mean, running_var, train, momentum, epsilon);
      }
    });
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(const Tensor& grad_out, const Tensor& self, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var,
                                                            const Tensor& save_mean, const Tensor& save_invstd, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "batch_norm_backward", [&] {
      if (cuda::detail::canUse32BitIndexMath(self)) {
        return batch_norm_backward_cuda_template<scalar_t, int32_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      } else {
        return batch_norm_backward_cuda_template<scalar_t, int64_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      }
    });
}

} } // namespace at::native
