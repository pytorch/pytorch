#include <c10/core/impl/cow/try_ensure.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/Storage.h>
#include <c10/core/impl/cow/context.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace c10::impl {
namespace {

MATCHER(is_copy_on_write, "") {
  return arg.data_ptr().get_deleter() == cow::Context::delete_instance;
}

TEST(try_ensure_test, basic) {
  Storage storage(/*use_byte_size_t=*/{}, /*size_bytes=*/7, GetCPUAllocator());
  ASSERT_THAT(
      *storage.getWeakStorageImpl().lock(), testing::Not(is_copy_on_write()));
  void* original_ctx = storage.data_ptr().get_context();

  // Create a new view family on the storage.
  intrusive_ptr<StorageImpl> view_family_2 = cow::try_ensure(storage);
  // Storage is now copy-on-write.
  ASSERT_THAT(*storage.getWeakStorageImpl().lock(), is_copy_on_write());
  // And it has a new context.
  void* new_ctx = storage.data_ptr().get_context();
  ASSERT_THAT(new_ctx, testing::Ne(original_ctx));

  // The view family is also a new copy-on-write storage.
  // ASSERT_THAT(new_impl, testing::Not(testing::IsNull()));
  ASSERT_THAT(view_family_2, testing::Pointee(is_copy_on_write()));
  // The new view family is that shares the same context.
  ASSERT_THAT(view_family_2->data_ptr().get_context(), testing::Eq(new_ctx));

  // Do it again. The storage is already copy-on-write, so it won't
  // change.
  intrusive_ptr<StorageImpl> view_family_3 = cow::try_ensure(storage);
  ASSERT_THAT(storage.data_ptr().get_context(), testing::Eq(new_ctx));
  // And the new family has the same context, of course.
  ASSERT_THAT(view_family_3->data_ptr().get_context(), testing::Eq(new_ctx));
}

TEST(try_ensure_test, does_not_wrap_custom_context) {
  Storage storage(
      /*use_byte_size_t=*/{},
      /*size_bytes=*/7,
      DataPtr(new std::byte[7], std::malloc(1), std::free, DeviceType::CPU),
      GetCPUAllocator());
  ASSERT_THAT(
      *storage.getWeakStorageImpl().lock(), testing::Not(is_copy_on_write()));
  void* original_ctx = storage.data_ptr().get_context();

  // Fails to wrap.
  ASSERT_THAT(cow::try_ensure(storage), testing::IsNull());
  // Storage is unmodified.
  ASSERT_THAT(storage.data_ptr().get_context(), testing::Eq(original_ctx));
}

} // namespace
} // namespace c10::impl
