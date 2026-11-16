#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/generator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <vector>

using torch::stable::Generator;
using torch::stable::Tensor;

// Calls aten::rand.generator through torch_call_dispatcher, threading a
// Generator passed from Python and deriving the output device from the
// generator itself (which exercises Generator::device()).
Tensor my_rand_with_generator(std::vector<int64_t> size, Generator generator) {
  torch::stable::Device device = generator.device();

  // aten::rand.generator(SymInt[] size, *, Generator? generator,
  //   ScalarType? dtype=None, Layout? layout=None, Device? device=None,
  //   bool? pin_memory=None) -> Tensor
  StableIValue stack[6];
  stack[0] = torch::stable::detail::from(size);
  stack[1] = torch::stable::detail::from(std::optional<Generator>(generator));
  stack[2] = torch::stable::detail::from(std::nullopt); // dtype
  stack[3] = torch::stable::detail::from(std::nullopt); // layout
  stack[4] =
      torch::stable::detail::from(std::optional<torch::stable::Device>(device));
  stack[5] = torch::stable::detail::from(std::nullopt); // pin_memory
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::rand", "generator", stack, TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_rand_with_generator(int[] size, Generator generator) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_rand_with_generator", TORCH_BOX(&my_rand_with_generator));
}
