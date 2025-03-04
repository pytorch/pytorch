#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/library_stable.h>

using RAIIATH = torch::aot_inductor::RAIIAtenTensorHandle;

void inline sgd_math(
  float* param_ptr,
  float* grad_ptr,
  float* out_ptr,
  const double weight_decay,
  const double lr,
  const bool maximize,
  int64_t size
){
  int64_t d = 0;
  for (; d < size; d++) {
    float grad_val = grad_ptr[d];
    if (grad_scale_ptr) {
      grad_val = grad_ptr[d] / float(*grad_scale_ptr);
      grad_ptr[d] = grad_val;
    }
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.0){
      grad_val += param_ptr[d] * float(weight_decay);
    }
    out_ptr[d] -= grad_val * float(lr);
  }
}


RAIIATH sgd_out_of_place(
    const RAIIATH param,
    const RAIIATH grad,
    const double weight_decay,
    const double lr,
    const bool maximize) {
  
  int64_t param_dim;
  aoti_torch_get_dim(param.get(), &param_dim);

  int64_t param_sizes[param_dim];
  int64_t param_strides[param_dim];
  aoti_torch_get_sizes(param.get(), &param_sizes);
  aoti_torch_get_strides(param.get(), &param_strides);

  int32_t param_dtype;
  aoti_torch_dtype_int32(param.get(), &param_dtype);
  
  int32_t param_device_type;
  int32_t param_device_index;
  aoti_torch_get_device_type(param.get(), &param_device_type);
  aoti_torch_get_device_index(param.get(), &param_device_index);

  AtenTensorHandle out;
  aoti_torch_empty_strided(param_dim, param_sizes, param_strides, param_dtype, param_device_type, param_device_index, &out);

  void* param_ptr;
  aoti_torch_data_ptr(param.get(), &param_ptr);
  void* grad_ptr;
  aoti_torch_data_ptr(grad.get(), &grad_ptr);
  void* out_ptr;
  aoti_torch_data_ptr(out, &out_ptr);

  auto param_fp_ptr = reinterpret_cast<float*>(param_ptr);
  auto grad_fp_ptr = reinterpret_cast<float*>(grad_ptr);
  auto out_fp_ptr = reinterpret_cast<float*>(out_ptr);
  
  sgd_math(
    param_fp_ptr,
    grad_fp_ptr,
    out_fp_ptr,
    weight_decay,
    lr,
    maximize,
    param.numel()
  );

  return RAIIATH(out);
}


void voidyvoid_boxed_ATH_sgd_out_of_place(void **stack, int64_t num_args, int64_t num_outputs) {
  RAIIATH param(reinterpret_cast<AtenTensorHandle>(stack[0]));
  RAIIATH grad(reinterpret_cast<AtenTensorHandle>(stack[1]));
  double weight_decay = reinterpret_cast<double>(stack[2]);
  double lr = reinterpret_cast<double>(stack[3]);
  bool maximize = reinterpret_cast<bool>(stack[4]);

  RAIIATH raiiath_res = sgd_out_of_place(
    std::move(param),
    std::move(grad),
    weight_decay,
    lr,
    maximize);
  
  void *out = reinterpret_cast<void*>(raiiath_res.release());
  stack[num_args] = out;
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
    m.impl("libtorch_agnostic::sgd_out_of_place", &voidyvoid_boxed_ATH_sgd_out_of_place);
}
