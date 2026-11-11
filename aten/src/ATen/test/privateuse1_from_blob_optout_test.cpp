#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

using namespace at;

namespace {

// A PrivateUse1 backend that registers hooks for unrelated reasons (as most
// backends do -- generators, pinned memory, etc.) but does NOT opt into the
// new from_blob hook. This must be a completely separate test binary from
// privateuse1_from_blob_test.cpp: PrivateUse1HooksInterface registration is
// a global one-shot per process, so the two fake implementations (opted-in
// vs. opted-out) can never coexist in the same test binary.
class MinimalHooksNoFromBlobOverride : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override {
    return true;
  }

  // Deliberately does NOT override hasCustomFromBlob()/fromBlobPrivateUse1():
  // this is the "existing backend, unmodified" scenario.
};

bool g_registered = [] {
  at::RegisterPrivateUse1HooksInterface(new MinimalHooksNoFromBlobOverride());
  return true;
}();

} // namespace

TEST(PrivateUse1FromBlobOptOut, FallsThroughToDefaultConstructionPath) {
  ASSERT_TRUE(g_registered);
  ASSERT_TRUE(at::isPrivateUse1HooksRegistered());
  ASSERT_FALSE(at::detail::getPrivateUse1Hooks().hasCustomFromBlob());

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  // Since hasCustomFromBlob() is false, TensorMaker::make_tensor() must take
  // the unchanged default branch even though PrivateUse1 hooks ARE
  // registered -- proving existing backends that don't opt in are
  // completely unaffected by this change.
  at::Device device(at::DeviceType::PrivateUse1, 0);
  Tensor tensor = at::from_blob(
      data.data(),
      {2, 2},
      /*deleter=*/[](void*) {},
      at::TensorOptions().dtype(at::kFloat).device(device),
      /*target_device=*/device);

  // typeid check: this must be exactly the generic TensorImpl, not any
  // subclass -- the default path was never touched by this change.
  EXPECT_EQ(typeid(*tensor.unsafeGetTensorImpl()), typeid(c10::TensorImpl));
  EXPECT_EQ(tensor.sizes(), (at::IntArrayRef{2, 2}));
  EXPECT_EQ(tensor.device().type(), at::DeviceType::PrivateUse1);
}
