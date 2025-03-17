#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/stable/library.h>

#include <optional>

using RAIIATH = torch::aot_inductor::RAIIAtenTensorHandle;

void inline sgd_math(
  float* param_ptr,
  float* grad_ptr,
  float* out_ptr,
  const float weight_decay,
  const double lr,
  const bool maximize,
  int64_t size
){
  int64_t d = 0;
  for (; d < size; d++) {
    float grad_val = grad_ptr[d];
    if (maximize) grad_val = -grad_val;
    if (weight_decay != 0.0){
      grad_val += param_ptr[d] * weight_decay;
    }
    out_ptr[d] = param_ptr[d] - grad_val * float(lr);
  }
}


RAIIATH sgd_out_of_place(
    const RAIIATH param,
    const RAIIATH grad,
    const float weight_decay,
    const double lr,
    const bool maximize) {

  int64_t param_dim;
  aoti_torch_get_dim(param.get(), &param_dim);

  int64_t *param_sizes;
  int64_t *param_strides;
  aoti_torch_get_sizes(param.get(), &param_sizes);
  aoti_torch_get_strides(param.get(), &param_strides);

  int32_t param_dtype;
  aoti_torch_get_dtype(param.get(), &param_dtype);

  int32_t param_device_type;
  int32_t param_device_index;
  aoti_torch_get_device_type(param.get(), &param_device_type);
  aoti_torch_get_device_index(param.get(), &param_device_index);

  AtenTensorHandle out;
  aoti_torch_empty_strided(param_dim, param_sizes, param_strides, param_dtype, param_device_type, param_device_index, &out);

  void* param_ptr;
  aoti_torch_get_data_ptr(param.get(), &param_ptr);
  void* grad_ptr;
  aoti_torch_get_data_ptr(grad.get(), &grad_ptr);
  void* out_ptr;
  aoti_torch_get_data_ptr(out, &out_ptr);

  auto param_fp_ptr = reinterpret_cast<float*>(param_ptr);
  auto grad_fp_ptr = reinterpret_cast<float*>(grad_ptr);
  auto out_fp_ptr = reinterpret_cast<float*>(out_ptr);

  int64_t param_numel;
  aoti_torch_get_numel(param.get(), &param_numel);

  sgd_math(
    param_fp_ptr,
    grad_fp_ptr,
    out_fp_ptr,
    weight_decay,
    lr,
    maximize,
    param_numel
  );

  return RAIIATH(out);
}


void boxed_sgd_out_of_place(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  RAIIATH param(to<AtenTensorHandle>(stack[0]));
  RAIIATH grad(to<AtenTensorHandle>(stack[1]));
  auto weight_decay = to<double>(stack[2]);
  auto lr = to<double>(stack[3]);
  auto maximize = to<bool>(stack[4]);

  RAIIATH raiiath_res = sgd_out_of_place(
    std::move(param),
    std::move(grad),
    float(weight_decay),
    lr,
    maximize);

  stack[0] = from(raiiath_res.release());
}

STABLE_TORCH_LIBRARY(libtorch_agnostic, m) {
  m.def("sgd_out_of_place(Tensor param, Tensor grad, float weight_decay, float lr, bool maximize) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
  m.impl("sgd_out_of_place", &boxed_sgd_out_of_place);
}

RAIIATH identity(RAIIATH t) {
  return std::move(t);
}

void boxed_identity(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  RAIIATH t(to<AtenTensorHandle>(stack[0]));
  RAIIATH raiiath_res = identity(std::move(t));
  stack[0] = from(raiiath_res.release());
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("identity(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CUDA, m) {
  m.impl("identity", &boxed_identity);
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CPU, m) {
  m.impl("identity", &boxed_identity);
}

RAIIATH my_abs(RAIIATH t) {
  const auto num_args = 1;
  StableIValue stack[num_args];
  stack[0] = from(t.release());
  aoti_torch_call_dispatcher("aten::abs", "", stack);
  return RAIIATH(to<AtenTensorHandle>(stack[0]));
}

void boxed_my_abs(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  RAIIATH t(to<AtenTensorHandle>(stack[0]));
  RAIIATH raiiath_res = my_abs(std::move(t));
  stack[0] = from(raiiath_res.release());
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_abs(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_abs", &boxed_my_abs);
}

RAIIATH my_ones_like(RAIIATH t, StableIValue device) {
  const auto num_args = 6;
  StableIValue stack[num_args];

  int32_t t_dtype;
  aoti_torch_get_dtype(t.get(), &t_dtype);
  auto mf = aoti_torch_memory_format_contiguous_format();

  stack[0] = from(t.release());
  stack[1] = from(std::optional(t_dtype));    // dtype
  stack[2] = from(std::nullopt);              // layout
  stack[3] = from(std::optional(device));     // device
  stack[4] = from(std::optional(false));      // pin_memory
  stack[5] = from(std::optional(mf));         // memory_format

  aoti_torch_call_dispatcher("aten::ones_like", "", stack);

  return RAIIATH(to<AtenTensorHandle>(stack[0]));
}

void boxed_my_ones_like(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  RAIIATH t(to<AtenTensorHandle>(stack[0]));
  StableIValue device = stack[1];

  RAIIATH raiiath_res = my_ones_like(std::move(t), device);
  stack[0] = from(raiiath_res.release());
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("my_ones_like(Tensor t, Device d) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("my_ones_like", &boxed_my_ones_like);
}

std::tuple<RAIIATH, RAIIATH, bool> exp_neg_is_leaf(RAIIATH t1, RAIIATH t2, RAIIATH t3) {
  StableIValue stack1[1];
  stack1[0] = from(t1.release());
  aoti_torch_call_dispatcher("aten::exp", "", stack1);

  StableIValue stack2[1];
  stack2[0] = from(t2.release());
  aoti_torch_call_dispatcher("aten::neg", "", stack2);

  StableIValue stack3[1];
  stack3[0] = from(t3.release());
  aoti_torch_call_dispatcher("aten::is_leaf", "", stack3);

  return std::make_tuple(
    RAIIATH(to<AtenTensorHandle>(stack1[0])),
    RAIIATH(to<AtenTensorHandle>(stack2[0])),
    to<bool>(stack3[0]));
}

void boxed_exp_neg_is_leaf(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  RAIIATH t1(to<AtenTensorHandle>(stack[0]));
  RAIIATH t2(to<AtenTensorHandle>(stack[1]));
  RAIIATH t3(to<AtenTensorHandle>(stack[2]));
  auto tuple = exp_neg_is_leaf(std::move(t1), std::move(t2), std::move(t3));
  stack[0] = from(std::get<0>(tuple).release());
  stack[1] = from(std::get<1>(tuple).release());
  stack[2] = from(std::get<2>(tuple));
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic, m) {
  m.def("exp_neg_is_leaf(Tensor t1, Tensor t2, Tensor t3) -> (Tensor, Tensor, bool)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic, CompositeExplicitAutograd, m) {
  m.impl("exp_neg_is_leaf", &boxed_exp_neg_is_leaf);
}
