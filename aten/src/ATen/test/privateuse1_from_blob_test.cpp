#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <atomic>

using namespace at;

namespace {

// A minimal TensorImpl subclass, standing in for a real backend's (e.g.
// NPU's) custom TensorImpl. Its only purpose is to prove that
// fromBlobPrivateUse1() can hand back a tensor whose impl is genuinely NOT
// the generic c10::TensorImpl.
class TestPrivateUse1TensorImpl : public c10::TensorImpl {
 public:
  TestPrivateUse1TensorImpl(
      c10::Storage&& storage,
      c10::DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type)
      : c10::TensorImpl(std::move(storage), dispatch_key, data_type) {}
};

std::atomic<int> g_hook_call_count{0};

class TestFromBlobHooks : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override {
    return true;
  }

  bool hasCustomFromBlob() const override {
    return true;
  }

  at::TensorBase fromBlobPrivateUse1(
      c10::DataPtr&& data_ptr,
      std::size_t size_bytes,
      at::IntArrayRef sizes,
      at::OptionalIntArrayRef strides,
      std::optional<int64_t> storage_offset,
      const at::TensorOptions& options,
      bool resizeable,
      c10::Allocator* allocator) const override {
    g_hook_call_count++;

    c10::Storage storage{
        c10::Storage::use_byte_size_t{},
        size_bytes,
        std::move(data_ptr),
        /*allocator=*/allocator,
        /*resizable=*/resizeable};

    at::TensorBase tensor = at::detail::make_tensor_base<
        TestPrivateUse1TensorImpl>(
        std::move(storage), options.computeDispatchKey(), options.dtype());

    auto* impl = tensor.unsafeGetTensorImpl();
    if (strides) {
      impl->set_sizes_and_strides(sizes, *strides);
    } else {
      impl->set_sizes_contiguous(sizes);
    }
    if (storage_offset) {
      impl->set_storage_offset(*storage_offset);
    }
    impl->set_requires_grad(options.requires_grad());
    return tensor;
  }
};

// Registered exactly once for this test binary. PrivateUse1HooksInterface
// registration is a global one-shot (throws if called twice), so this hook
// implementation must not be shared with any other test binary target.
bool g_registered = [] {
  at::RegisterPrivateUse1HooksInterface(new TestFromBlobHooks());
  return true;
}();

} // namespace

TEST(PrivateUse1FromBlob, DefaultDeviceIsUnaffected) {
  ASSERT_TRUE(g_registered);
  int before = g_hook_call_count.load();

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor cpu_tensor = at::from_blob(
      data.data(), {2, 3}, at::TensorOptions().dtype(at::kFloat));

  // The hook must not fire for non-PrivateUse1 devices, and the resulting
  // tensor must be a plain TensorImpl, exactly as before this change.
  EXPECT_EQ(g_hook_call_count.load(), before);
  EXPECT_EQ(
      dynamic_cast<TestPrivateUse1TensorImpl*>(
          cpu_tensor.unsafeGetTensorImpl()),
      nullptr);
  EXPECT_TRUE(cpu_tensor.is_cpu());
  EXPECT_EQ(cpu_tensor.sizes(), (at::IntArrayRef{2, 3}));
}

TEST(PrivateUse1FromBlob, PrivateUse1RoutesThroughCustomHook) {
  int before = g_hook_call_count.load();

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  // A deliberately non-default, non-contiguous view: sizes {2, 3}, strides
  // {4, 1} (i.e. row pitch of 4 over a buffer of 8 elements), storage_offset
  // 1 -- exercising the exact data plumbing (sizes/strides/storage_offset)
  // that a real DLPack-import round trip relies on.
  at::Device device(at::DeviceType::PrivateUse1, 0);
  Tensor tensor = at::from_blob(
      data.data(),
      {2, 3},
      {4, 1},
      /*storage_offset=*/1,
      /*deleter=*/[](void*) {},
      at::TensorOptions().dtype(at::kFloat).device(device),
      /*target_device=*/device);

  EXPECT_EQ(g_hook_call_count.load(), before + 1);
  EXPECT_NE(
      dynamic_cast<TestPrivateUse1TensorImpl*>(tensor.unsafeGetTensorImpl()),
      nullptr);
  EXPECT_EQ(tensor.sizes(), (at::IntArrayRef{2, 3}));
  EXPECT_EQ(tensor.strides(), (at::IntArrayRef{4, 1}));
  EXPECT_EQ(tensor.storage_offset(), 1);
  EXPECT_EQ(tensor.scalar_type(), at::kFloat);
  EXPECT_EQ(tensor.device().type(), at::DeviceType::PrivateUse1);

  // storage_offset=1 means element [0,0] of the view is data[1] == 2.0f,
  // while the underlying storage still starts at data[0] == 1.0f.
  EXPECT_EQ(*static_cast<const float*>(tensor.storage().data()), 1.0f);
  EXPECT_EQ(tensor.const_data_ptr<float>()[0], 2.0f);
}
