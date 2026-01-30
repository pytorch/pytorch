#include "kernel.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/core/ScalarType.h>

#ifdef LAE_USE_CUDA
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#endif

#include <optional>

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

using torch::stable::Tensor;

Tensor sgd_out_of_place(
    const Tensor param,
    const Tensor grad,
    const double weight_decay,
    const double lr,
    const bool maximize) {
  STD_TORCH_CHECK(param.dim() == 1, "param must be 1D");

  // these test the get_device() and get_device_index() methods
  // while ascertaining that we are still on CPU
  STD_TORCH_CHECK(param.get_device() == -1, "CPU device index = -1");
  STD_TORCH_CHECK(param.get_device_index() == -1, "CPU device index = -1");

  // testing Tensor strides + stride
  STD_TORCH_CHECK(param.strides()[0] == param.stride(0));

  auto out = new_empty(param, param.sizes());

  sgd_math(
    reinterpret_cast<float*>(param.data_ptr()),
    reinterpret_cast<float*>(grad.data_ptr()),
    reinterpret_cast<float*>(out.data_ptr()),
    float(weight_decay),
    lr,
    maximize,
    param.numel()
  );

  return out;
}

void boxed_sgd_out_of_place(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor res = sgd_out_of_place(
    torch::stable::detail::to<Tensor>(stack[0]),
    torch::stable::detail::to<Tensor>(stack[1]),
    float(torch::stable::detail::to<double>(stack[2])),
    torch::stable::detail::to<double>(stack[3]),
    torch::stable::detail::to<bool>(stack[4]));

  stack[0] = from(res);
}

STABLE_TORCH_LIBRARY(libtorch_agn_2_9, m) {
  m.def("sgd_out_of_place(Tensor param, Tensor grad, float weight_decay, float lr, bool maximize) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CPU, m) {
  m.impl("sgd_out_of_place", &boxed_sgd_out_of_place);
}

Tensor identity(Tensor t) {
  return t;
}


STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("identity(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CUDA, m) {
  m.impl("identity", TORCH_BOX(&identity));
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CPU, m) {
  m.impl("identity", TORCH_BOX(&identity));
}

Tensor my_abs(Tensor t) {
  const auto num_args = 1;
  StableIValue stack[num_args];
  stack[0] = torch::stable::detail::from(t);
  aoti_torch_call_dispatcher("aten::abs", "", stack);
  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_abs(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_abs", TORCH_BOX(&my_abs));
}

Tensor my_ones_like(Tensor t, StableIValue device) {
  const auto num_args = 6;
  StableIValue stack[num_args];

  auto mf = aoti_torch_memory_format_contiguous_format();

  stack[0] = torch::stable::detail::from(t);
  stack[1] = torch::stable::detail::from(std::optional(t.scalar_type()));    // dtype
  stack[2] = torch::stable::detail::from(std::nullopt);              // layout
  stack[3] = torch::stable::detail::from(std::optional(device));     // device
  stack[4] = torch::stable::detail::from(std::optional(false));      // pin_memory
  stack[5] = torch::stable::detail::from(std::optional(mf));         // memory_format

  aoti_torch_call_dispatcher("aten::ones_like", "", stack);

  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_ones_like(Tensor t, Device d) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_ones_like", TORCH_BOX(&my_ones_like));
}

std::tuple<Tensor, Tensor, bool> exp_neg_is_leaf(Tensor t1, Tensor t2, Tensor t3) {
  StableIValue stack_exp[1];
  stack_exp[0] = torch::stable::detail::from(t1);
  aoti_torch_call_dispatcher("aten::exp", "", stack_exp);

  StableIValue stack_neg[1];
  stack_neg[0] = torch::stable::detail::from(t2);
  aoti_torch_call_dispatcher("aten::neg", "", stack_neg);

  StableIValue stack_is_leaf[1];
  stack_is_leaf[0] = torch::stable::detail::from(t3);
  aoti_torch_call_dispatcher("aten::is_leaf", "", stack_is_leaf);

  return std::make_tuple(
    torch::stable::detail::to<Tensor>(stack_exp[0]),
    torch::stable::detail::to<Tensor>(stack_neg[0]),
    torch::stable::detail::to<bool>(stack_is_leaf[0]));
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("exp_neg_is_leaf(Tensor t1, Tensor t2, Tensor t3) -> (Tensor, Tensor, bool)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("exp_neg_is_leaf", TORCH_BOX(&exp_neg_is_leaf));
}

Tensor neg_exp(Tensor t) {
  StableIValue stack[1];
  stack[0] = torch::stable::detail::from(t);
  aoti_torch_call_dispatcher("aten::exp", "", stack);
  aoti_torch_call_dispatcher("aten::neg", "", stack);
  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("neg_exp(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("neg_exp", TORCH_BOX(&neg_exp));
}

Tensor divide_neg_exp(Tensor t) {
  StableIValue stack_neg[1];
  stack_neg[0] = torch::stable::detail::from(t);

  StableIValue stack_exp[1];
  stack_exp[0] = torch::stable::detail::from(t);
  aoti_torch_call_dispatcher("aten::exp", "", stack_exp);
  aoti_torch_call_dispatcher("aten::neg", "", stack_neg);

  StableIValue stack_div[2];
  stack_div[0] = stack_neg[0];
  stack_div[1] = stack_exp[0];
  aoti_torch_call_dispatcher("aten::divide", "Tensor", stack_div);
  return torch::stable::detail::to<Tensor>(stack_div[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("divide_neg_exp(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("divide_neg_exp", TORCH_BOX(&divide_neg_exp));
}

bool is_contiguous(Tensor t) {
  return t.is_contiguous();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("is_contiguous(Tensor t) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("is_contiguous", TORCH_BOX(&is_contiguous));
}

Tensor my_transpose(Tensor t, int64_t dim0, int64_t dim1) {
  return transpose(t, dim0, dim1);
}

// This is used to test const torch::stable::Tensor& with TORCH_BOX
Tensor my_empty_like(const Tensor& t) {
  return empty_like(t);
}

// This is used to test torch::stable::Tensor& with TORCH_BOX
bool my_is_cpu(Tensor& t) {
  return t.is_cpu();
}

Tensor fill_infinity(Tensor t) {
  auto value = std::numeric_limits<float>::infinity();
  return fill_(t, value);
}

Tensor my_pad(Tensor t) {
  std::string mode = "constant";
  double value = 0.0;
  return pad(t, {1, 2, 2, 1}, mode, value);
}

Tensor my_narrow(Tensor t, int64_t dim, int64_t start, int64_t length) {
  return narrow(t, dim, start, length);
}

Tensor my_new_empty_dtype_variant(Tensor t) {
  // Still using a std::vector below even though people can just pass in an
  // initializer list (which will be implicitly converted to an HeaderOnlyArrayRef)
  // directly.
  // This is to test that passing in a std::vector works for BC. (It gets
  // implicitly converted to HeaderOnlyArrayRef too!)
  std::vector<int64_t> sizes = {2, 5};
  auto dtype = std::make_optional(torch::headeronly::ScalarType::BFloat16);
  return new_empty(t, sizes, dtype);
}

Tensor my_new_zeros_dtype_variant(Tensor t) {
  auto dtype = std::make_optional(at::ScalarType::Float);
  return new_zeros(t, {2, 5}, dtype);
}

Tensor my_copy_(Tensor dst, Tensor src, bool non_blocking) {
  return copy_(dst, src, non_blocking);
}

Tensor my_clone(Tensor t) {
  return clone(t);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_transpose(Tensor t, int dim0, int dim1) -> Tensor");
  m.def("my_empty_like(Tensor t) -> Tensor");
  m.def("fill_infinity(Tensor(a!) t) -> Tensor(a!)");
  m.def("my_pad(Tensor t) -> Tensor");
  m.def("my_narrow(Tensor t, int dim, int start, int length) -> Tensor");
  m.def("my_new_empty_dtype_variant(Tensor t) -> Tensor");
  m.def("my_new_zeros_dtype_variant(Tensor t) -> Tensor");
  m.def("my_copy_(Tensor dst, Tensor src, bool non_blocking) -> Tensor");
  m.def("my_clone(Tensor t) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_transpose", TORCH_BOX(&my_transpose));
  m.impl("my_empty_like", TORCH_BOX(&my_empty_like));
  m.impl("fill_infinity", TORCH_BOX(&fill_infinity));
  m.impl("my_is_cpu", TORCH_BOX(&my_is_cpu));
  m.impl("my_new_empty_dtype_variant", TORCH_BOX(&my_new_empty_dtype_variant));
  m.impl("my_new_zeros_dtype_variant", TORCH_BOX(&my_new_zeros_dtype_variant));
  m.impl("my_copy_", TORCH_BOX(&my_copy_));
  m.impl("my_clone", TORCH_BOX(&my_clone));
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeImplicitAutograd, m) {
  m.impl("my_pad", TORCH_BOX(&my_pad));
  m.impl("my_narrow", TORCH_BOX(&my_narrow));
}

Tensor my_zero_(Tensor t) {
  return zero_(t);
}

Tensor my_amax(Tensor t) {
  return amax(t, 0, false);
}

Tensor my_amax_vec(Tensor t) {
  return amax(t, {0,1}, false);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_zero_(Tensor(a!) t) -> Tensor(a!)");
  m.def("my_amax(Tensor a) -> Tensor");
  m.def("my_amax_vec(Tensor a) -> Tensor");
  m.def("my_is_cpu(Tensor t) -> bool");
  m.def("test_default_constructor(bool undefined) -> bool");
}

bool test_default_constructor(bool defined) {
  Tensor out;
  if (defined) {
    AtenTensorHandle defined_ath;
    int64_t sizes[] = {2, 3};
    int64_t strides[] = {3, 1};
    aoti_torch_empty_strided(
        2,
        sizes,
        strides,
        aoti_torch_dtype_float32(),
        aoti_torch_device_type_cpu(),
        0,
        &defined_ath);
    out = Tensor(defined_ath);
  }
  return out.defined();
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_zero_", TORCH_BOX(&my_zero_));
  m.impl("my_amax", TORCH_BOX(&my_amax));
  m.impl("my_amax_vec", TORCH_BOX(&my_amax_vec));
  m.impl("test_default_constructor", TORCH_BOX(&test_default_constructor));
}

Tensor mv_tensor_accessor_cpu(Tensor m, Tensor v) {
  STD_TORCH_CHECK(m.dim() == 2, "m must be 2D");
  STD_TORCH_CHECK(v.dim() == 1, "v must be 1D");
  STD_TORCH_CHECK(m.size(1) == v.size(0), "m.shape[1] == v.shape[0] must hold");
  STD_TORCH_CHECK(m.scalar_type() == v.scalar_type(), "m and v must have the same dtype");
  STD_TORCH_CHECK(m.device() == v.device(), "m and v must be on the same device");
  Tensor res = new_empty(m, {m.size(0)});
  THO_DISPATCH_V2(m.scalar_type(), "mv_tensor_accessor_cpu",
                  AT_WRAP(([&]() {
                    auto resa = Accessor_cpu<scalar_t, 1>(reinterpret_cast<scalar_t*>(res.data_ptr()), res.sizes().data(), res.strides().data());
                    auto ma = Accessor_cpu<scalar_t, 2>(reinterpret_cast<scalar_t*>(m.data_ptr()), m.sizes().data(), m.strides().data());
                    auto va = Accessor_cpu<scalar_t, 1>(reinterpret_cast<scalar_t*>(v.data_ptr()), v.sizes().data(), v.strides().data());
                    mv_tensor_accessor_kernel<Accessor_cpu, scalar_t>(resa, ma, va);
                  })),
                  AT_FLOATING_TYPES);
  return res;
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("mv_tensor_accessor(Tensor m, Tensor v) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CPU, m) {
  m.impl("mv_tensor_accessor", TORCH_BOX(&mv_tensor_accessor_cpu));
}

// Test functions for torch::stable::accelerator APIs

#ifdef LAE_USE_CUDA
int64_t test_device_guard(int64_t device_index) {
  using torch::stable::accelerator::DeviceGuard;

  STD_TORCH_CHECK(
      device_index >= std::numeric_limits<int32_t>::min() &&
          device_index <= std::numeric_limits<int32_t>::max(),
      "Device index is out of range of DeviceIndex (int32_t).");

  DeviceGuard guard(device_index);
  int currentDevice;
  cudaError_t err = cudaGetDevice(&currentDevice);
  STD_TORCH_CHECK(err == cudaSuccess);
  return currentDevice;
}

int64_t test_device_guard_set_index() {
  using torch::stable::accelerator::DeviceGuard;

  DeviceGuard guard(1);
  guard.set_index(0);
  int currentDevice;
  cudaError_t err = cudaGetDevice(&currentDevice);
  STD_TORCH_CHECK(err == cudaSuccess);
  return currentDevice;
}

int64_t test_stream(int32_t device_index) {
  STD_TORCH_CHECK(
      device_index >= std::numeric_limits<int32_t>::min() &&
          device_index <= std::numeric_limits<int32_t>::max(),
      "Device index is out of range of DeviceIndex (int32_t).");

  return torch::stable::accelerator::getCurrentStream(device_index).id();
}

int64_t test_get_current_device_index() {
  return torch::stable::accelerator::getCurrentDeviceIndex();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("test_device_guard(int device_index) -> int");
  m.def("test_device_guard_set_index() -> int");
  m.def("test_stream(int device_index) -> int");
  m.def("test_get_current_device_index() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("test_device_guard", TORCH_BOX(&test_device_guard));
  m.impl("test_device_guard_set_index", TORCH_BOX(&test_device_guard_set_index));
  m.impl("test_stream", TORCH_BOX(&test_stream));
  m.impl("test_get_current_device_index", TORCH_BOX(&test_get_current_device_index));
}

#endif // LAE_USE_CUDA

Tensor my_flatten(Tensor t, int64_t start_dim, int64_t end_dim) {
  return flatten(t, start_dim, end_dim);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_flatten(Tensor t, int start_dim=0, int end_dim=-1) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_flatten", TORCH_BOX(&my_flatten));
}

// Test function for const std::optional<Tensor>& with TORCH_BOX
// Returns the tensor if present, otherwise returns a zeros tensor of specified size
Tensor my_optional_tensor_ref(
    const std::optional<Tensor>& maybe_tensor,
    int64_t default_size) {
  if (maybe_tensor.has_value()) {
    return maybe_tensor.value();
  }
  // Create a zeros tensor as default
  AtenTensorHandle zeros_ath;
  int64_t sizes[] = {default_size};
  int64_t strides[] = {1};
  aoti_torch_empty_strided(
      1,
      sizes,
      strides,
      aoti_torch_dtype_float32(),
      aoti_torch_device_type_cpu(),
      0,
      &zeros_ath);
  Tensor zeros_tensor(zeros_ath);
  return zero_(zeros_tensor);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_optional_tensor_ref(Tensor? maybe_tensor, int default_size) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_optional_tensor_ref", TORCH_BOX(&my_optional_tensor_ref));
}

int64_t my_storage_offset(Tensor t) {
  return t.storage_offset();
}

int64_t my_element_size(Tensor t) {
  return static_cast<int64_t>(t.element_size());
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_storage_offset(Tensor t) -> int");
  m.def("my_element_size(Tensor t) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_storage_offset", TORCH_BOX(&my_storage_offset));
  m.impl("my_element_size", TORCH_BOX(&my_element_size));
}

Tensor my_unsqueeze(const Tensor& t, int64_t dim) {
  return torch::stable::unsqueeze(t, dim);
}

Tensor my_squeeze(const Tensor& t, int64_t dim) {
  return torch::stable::squeeze(t, dim);
}

Tensor my_select(const Tensor& t, int64_t dim, int64_t index) {
  return torch::stable::select(t, dim, index);
}

Tensor my_matmul(const Tensor& self, const Tensor& other) {
  return torch::stable::matmul(self, other);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_9, m) {
  m.def("my_unsqueeze(Tensor t, int dim) -> Tensor");
  m.def("my_squeeze(Tensor t, int dim) -> Tensor");
  m.def("my_select(Tensor t, int dim, int index) -> Tensor");
  m.def("my_matmul(Tensor self, Tensor other) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_9, CompositeExplicitAutograd, m) {
  m.impl("my_unsqueeze", TORCH_BOX(&my_unsqueeze));
  m.impl("my_squeeze", TORCH_BOX(&my_squeeze));
  m.impl("my_select", TORCH_BOX(&my_select));
  m.impl("my_matmul", TORCH_BOX(&my_matmul));
}
