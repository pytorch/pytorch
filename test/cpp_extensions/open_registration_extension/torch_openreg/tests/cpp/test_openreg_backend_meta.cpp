#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>

#include "runtime/OpenRegSerialization.h"

namespace c10::openreg {
namespace {

OpenRegBackendMeta* getMeta(const at::Tensor& t) {
  return dynamic_cast<OpenRegBackendMeta*>(
      t.unsafeGetTensorImpl()->get_backend_meta());
}

} // namespace

TEST(OpenRegBackendMetaTest, ForBlobSetterAttachesMeta) {
  std::array<int32_t, 3> storage{1, 2, 3};
  auto options = at::TensorOptions().dtype(at::kInt);

  // Without the setter, no backend meta is attached (zero-cost default).
  auto plain =
      at::for_blob(storage.data(), {3}).options(options).make_tensor();
  EXPECT_EQ(plain.unsafeGetTensorImpl()->get_backend_meta(), nullptr);

  // The setter folds the exact OpenRegBackendMeta in at construction.
  auto meta = c10::make_intrusive<OpenRegBackendMeta>(1, 29);
  auto t = at::for_blob(storage.data(), {3})
               .options(options)
               .backend_meta(meta)
               .make_tensor();
  EXPECT_EQ(t.unsafeGetTensorImpl()->get_backend_meta(), meta.get());
  ASSERT_NE(getMeta(t), nullptr);
  EXPECT_EQ(getMeta(t)->version_number_, 1);
  EXPECT_EQ(getMeta(t)->format_number_, 29);
}

TEST(OpenRegBackendMetaTest, SetterMatchesPostHoc) {
  std::array<int32_t, 3> storage{1, 2, 3};
  auto options = at::TensorOptions().dtype(at::kInt);

  // Post-hoc idiom (what for_deserialization uses): build, then set.
  auto postHocMeta = c10::make_intrusive<OpenRegBackendMeta>(1, 29);
  auto postHoc =
      at::for_blob(storage.data(), {3}).options(options).make_tensor();
  postHoc.unsafeGetTensorImpl()->set_backend_meta(postHocMeta);

  // Setter idiom: fold a distinct meta in at construction.
  auto setterMeta = c10::make_intrusive<OpenRegBackendMeta>(2, 30);
  auto viaSetter = at::for_blob(storage.data(), {3})
                       .options(options)
                       .backend_meta(setterMeta)
                       .make_tensor();

  // Both attach exactly the object they were handed, in the same slot.
  EXPECT_EQ(
      postHoc.unsafeGetTensorImpl()->get_backend_meta(), postHocMeta.get());
  EXPECT_EQ(
      viaSetter.unsafeGetTensorImpl()->get_backend_meta(), setterMeta.get());

  // The setter preserves its own distinct payload, not the post-hoc one.
  ASSERT_NE(getMeta(viaSetter), nullptr);
  EXPECT_EQ(getMeta(viaSetter)->version_number_, 2);
  EXPECT_EQ(getMeta(viaSetter)->format_number_, 30);
}

} // namespace c10::openreg
