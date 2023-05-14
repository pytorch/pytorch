#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/cow/context.h>
#include <c10/core/impl/cow/deleter.h>
#include <c10/core/impl/cow/try_ensure.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string_view>

namespace c10::impl {
namespace {

MATCHER(is_copy_on_write, "") {
  const c10::StorageImpl& storage_impl = std::ref(arg);
  return storage_impl.data_ptr().get_deleter() == cow::delete_context;
}

TEST(materialize_test, not_copy_on_write_context) {
  StorageImpl storage(
      {}, /*size_bytes=*/7, GetCPUAllocator(), /*resizable=*/false);
  ASSERT_THAT(storage, testing::Not(is_copy_on_write()));

  void const* original_data = storage.data();

  // Nothing to materialize.
  ASSERT_THAT(storage.mutable_data(), testing::Eq(original_data));
}

TEST(materialize_test, copy_on_write_single_reference) {
  // A copy-on-write storage with only a single reference can just
  // drop the copy-on-write context upon materialization.
  std::unique_ptr<void, DeleterFnPtr> data(
      new std::byte[5],
      +[](void* bytes) { delete[] static_cast<std::byte*>(bytes); });
  void* data_ptr = data.get();
  StorageImpl storage(
      {},
      /*size_bytes=*/5,
      at::DataPtr(
          /*data=*/data_ptr,
          /*ctx=*/new cow::Context(std::move(data)),
          cow::delete_context,
          Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  ASSERT_THAT(storage, is_copy_on_write());

  ASSERT_THAT(storage.data(), testing::Eq(data_ptr));

  void const* original_data = storage.data();

  // Materializes storage. Only reference, so no new allocation.
  ASSERT_THAT(storage.mutable_data(), testing::Eq(original_data));
  // But it is no longer copy-on-write.
  ASSERT_THAT(storage, testing::Not(is_copy_on_write()));
}

TEST(materialize_test, copy_on_write) {
  StorageImpl original_storage(
      {}, /*size_bytes=*/7, GetCPUAllocator(), /*resizable=*/false);
  std::memcpy(original_storage.mutable_data(), "abcd", 5);
  void const* original_data = original_storage.data();

  auto new_storage = cow::try_ensure(original_storage);
  ASSERT_THAT(new_storage, testing::NotNull());

  auto context =
      new_storage->data_ptr().cast_context<cow::Context>(cow::delete_context);
  ASSERT_THAT(context, testing::NotNull());

  // Materialized storage has new copy of data.
  ASSERT_THAT(new_storage->mutable_data(), testing::Ne(original_data));

  // But the original storage still has the original copy.
  ASSERT_THAT(original_storage.data(), testing::Eq(original_data));

  // But their data is the same.
  ASSERT_THAT(
      static_cast<char const*>(new_storage->data()),
      testing::StrEq(static_cast<char const*>(original_storage.data())));
}

} // namespace
} // namespace c10::impl
