#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/operator.h>

using namespace at;

namespace {

// This test file gives an example implementation of vmap, using the
// dispatcher as the mechanism for converting calls to non-batched operations
// into batched operations.

// We support nesting VMapMode (in the same way that vmap in JAX is
// compositional)
thread_local int64_t vmap_level = 0;

constexpr auto VMapModeKey = DispatchKey::TESTING_ONLY_GenericModeTensorId;
struct VMapMode {
  // TODO: move constructor on DispatchKeyGuard is busted, please fix
  VMapMode() {
    if (vmap_level == 0) {
      c10::impl::tls_set_dispatch_key_included(VMapModeKey, true);
    }
    vmap_level++;
  }
  ~VMapMode() {
    TORCH_INTERNAL_ASSERT(vmap_level > 0);
    vmap_level--;
    if (vmap_level == 0) {
      c10::impl::tls_set_dispatch_key_included(VMapModeKey, false);
    }
  }
};
struct UnVMapMode {
  UnVMapMode() {
    TORCH_INTERNAL_ASSERT(vmap_level > 0);
    vmap_level--;
    if (vmap_level == 0) {
      c10::impl::tls_set_dispatch_key_included(VMapModeKey, false);
    }
  }
  ~UnVMapMode() {
    if (vmap_level == 0) {
      c10::impl::tls_set_dispatch_key_included(VMapModeKey, true);
    }
    vmap_level++;
  }
};

TEST(VMapTest, TestVMap) {
  // Many operations in PyTorch are already batched on the first dimension.  By
  // default, directly fallthrough and rely on the implicit batching.
  auto registry = c10::Dispatcher::singleton().registerBackendFallbackKernel(
      VMapModeKey,
      KernelFunction::makeFallthrough()
  );
  auto registry2 = torch::RegisterOperators()
    // Some operations need to be transformed to their batched versions
    .op(torch::RegisterOperators::options()
        .schema("aten::mv(Tensor self, Tensor vec) -> Tensor")
        .kernel(VMapModeKey, [] (const Tensor& a, const Tensor& b) -> Tensor {
          UnVMapMode g;
          return at::matmul(a, b.unsqueeze(-1)).squeeze(-1);
        }))
    ;

  auto mat = at::randn({10, 5, 20});
  auto vec = at::randn({10, 20});

  // at::mv is not batched
  ASSERT_ANY_THROW(at::mv(mat, vec));

  // But with vmap mode, you can write "normal" code and it will batch
  {
    VMapMode g;

    auto result = at::mv(mat, vec);

    ASSERT_EQ(result.size(0), 10);
    ASSERT_EQ(result.size(1), 5);
  }

  // You can nest vmap mode
  {
    auto mat2 = at::randn({3, 3, 5, 20}); // two batch dimensions
    auto vec2 = at::randn({3, 3, 20});

    VMapMode g;
    VMapMode g2;

    auto result = at::mv(mat2, vec2);

    ASSERT_EQ(result.size(0), 3);
    ASSERT_EQ(result.size(1), 3);
    ASSERT_EQ(result.size(2), 5);

    // NB: This example cheats a little, because matmul will batch to an arbitrary
    // number of dimensions, so you don't actually need the second VMapMode
  }

  // One major limitation of this proof of concept is that it doesn't
  // distinguish between tensors which are batched, and tensors which shouldn't
  // be batched (and instead should just be broadcasted on the batch dimension);
  // we assume ALL tensors are batched tensors.  We could further adjust the
  // implementation to handle this, probably by having VMapMode keep track (with
  // weak references) of what tensors are batched.  You could also record
  // batching directly on the tensor itself, but you would give up
  // compositionality in this case.
}

}
