#include <c10/core/impl/cow/try_ensure.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/cow/context.h>
#include <c10/core/impl/cow/deleter.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

namespace c10::impl {
namespace {

MATCHER(is_copy_on_write, "") {
  const c10::StorageImpl& storage = std::ref(arg);
  return storage.data_ptr().get_deleter() == cow::delete_context;
}

TEST(try_ensure_test, no_context) {
  StorageImpl original_storage(
      {}, /*size_bytes=*/7, GetCPUAllocator(), /*resizable=*/false);
  ASSERT_THAT(original_storage, testing::Not(is_copy_on_write()));

  intrusive_ptr<StorageImpl> new_storage = cow::try_ensure(original_storage);
  ASSERT_THAT(new_storage, testing::NotNull());

  // The original storage was modified in-place to now hold a copy on
  // write context.
  ASSERT_THAT(original_storage, is_copy_on_write());

  // The result is a different storage impl.
  ASSERT_THAT(&*new_storage, testing::Ne(&original_storage));
  // But it is also copy-on-write.
  ASSERT_THAT(*new_storage, is_copy_on_write());
  // But they share the same data!
  ASSERT_THAT(new_storage->data(), testing::Eq(original_storage.data()));
}

class OpaqueContext {};

TEST(try_ensure_test, different_context) {
  StorageImpl storage(
      {},
      /*size_bytes=*/5,
      at::DataPtr(
          /*data=*/new std::byte[5],
          /*ctx=*/new OpaqueContext,
          +[](void* opaque_ctx) {
            delete static_cast<OpaqueContext*>(opaque_ctx);
          },
          Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  // We can't handle an arbitrary context.
  ASSERT_THAT(cow::try_ensure(storage), testing::IsNull());
}

TEST(try_ensure_test, already_copy_on_write) {
  std::unique_ptr<void, DeleterFnPtr> data(
      new std::byte[5],
      +[](void* bytes) { delete[] static_cast<std::byte*>(bytes); });
  void* data_ptr = data.get();
  StorageImpl original_storage(
      {},
      /*size_bytes=*/5,
      at::DataPtr(
          /*data=*/data_ptr,
          /*ctx=*/new cow::Context(std::move(data)),
          cow::delete_context,
          Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  ASSERT_THAT(original_storage, is_copy_on_write());

  intrusive_ptr<StorageImpl> new_storage = cow::try_ensure(original_storage);
  ASSERT_THAT(new_storage, testing::NotNull());

  // The result is a different storage.
  ASSERT_THAT(&*new_storage, testing::Ne(&original_storage));
  // But it is also copy-on-write.
  ASSERT_THAT(*new_storage, is_copy_on_write());
  // But they share the same data!
  ASSERT_THAT(new_storage->data(), testing::Eq(original_storage.data()));
}

} // namespace
} // namespace c10::impl
