// Shared identity / abs / is_cpu kernels for the frozen BC fixture.
//
// TORCH_TARGET_VERSION is set by setup.py (overridable via the env var of the
// same name). Values match the other libtorch_agn_* extensions:
//   0x0209000000000000 (2.9)  -> manual boxed kernels (to<>/from<>)
//   0x020d000000000000 (2.13) -> TORCH_BOX (+ csrc/v213 sources)

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#ifndef TORCH_TARGET_VERSION
#error "TORCH_TARGET_VERSION must be defined (see libtorch_agn_frozen setup.py)"
#endif

using torch::stable::Tensor;

Tensor identity(Tensor t) {
  return t;
}

Tensor my_abs(Tensor t) {
  const auto num_args = 1;
  StableIValue stack[num_args];
  stack[0] = from(t);
  aoti_torch_call_dispatcher("aten::abs", "", stack);
  return to<Tensor>(stack[0]);
}

bool my_is_cpu(Tensor t) {
  return t.is_cpu();
}

#if TORCH_TARGET_VERSION >= 0x020d000000000000

STABLE_TORCH_LIBRARY(STABLE_LIB_NAME, m) {
  m.def("identity(Tensor t) -> Tensor");
  m.def("my_abs(Tensor t) -> Tensor");
  m.def("my_is_cpu(Tensor t) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CPU, m) {
  m.impl("identity", TORCH_BOX(&identity));
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CUDA, m) {
  m.impl("identity", TORCH_BOX(&identity));
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_abs", TORCH_BOX(&my_abs));
  m.impl("my_is_cpu", TORCH_BOX(&my_is_cpu));
}

#else // TORCH_TARGET_VERSION < 2.13

void boxed_identity(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor res = identity(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

void boxed_my_abs(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor tensor_res = my_abs(to<Tensor>(stack[0]));
  stack[0] = from(tensor_res);
}

void boxed_my_is_cpu(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  auto res = my_is_cpu(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

// Hardcoded library name: STABLE_LIB_NAME / TORCH_BOX are tip conveniences;
// the 2.9 header surface registers with a literal identifier.
STABLE_TORCH_LIBRARY(libtorch_agn_frozen, m) {
  m.def("identity(Tensor t) -> Tensor");
  m.def("my_abs(Tensor t) -> Tensor");
  m.def("my_is_cpu(Tensor t) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_frozen, CPU, m) {
  m.impl("identity", &boxed_identity);
}

// Same boxed identity works on CUDA tensors (matches the 2.9-era extension).
STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_frozen, CUDA, m) {
  m.impl("identity", &boxed_identity);
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_frozen, CompositeExplicitAutograd, m) {
  m.impl("my_abs", &boxed_my_abs);
  m.impl("my_is_cpu", &boxed_my_is_cpu);
}

#endif // TORCH_TARGET_VERSION >= 2.13
