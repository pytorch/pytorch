#include <c10/core/impl/COW.h>
#include <c10/core/impl/COWDeleter.h>

#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

// NOLINTBEGIN(clang-analyzer-cplusplus*)
namespace c10::impl {
namespace {

class DeleteTracker {
 public:
  explicit DeleteTracker(int& delete_count) : delete_count_(delete_count) {}
  DeleteTracker(const DeleteTracker&) = delete;
  DeleteTracker(DeleteTracker&&) = delete;
  DeleteTracker& operator=(const DeleteTracker&) = delete;
  DeleteTracker& operator=(DeleteTracker&&) = delete;
  ~DeleteTracker() {
    ++delete_count_;
  }

 private:
  int& delete_count_;
};

class ContextTest : public testing::Test {
 protected:
  auto delete_count() const -> int {
    return delete_count_;
  }
  auto new_delete_tracker() -> std::unique_ptr<void, DeleterFnPtr> {
    return {new DeleteTracker(delete_count_), +[](void* ptr) {
              delete static_cast<DeleteTracker*>(ptr);
            }};
  }

 private:
  int delete_count_ = 0;
};

TEST_F(ContextTest, Basic) {
  auto& context = *new cow::COWDeleterContext(new_delete_tracker());
  ASSERT_THAT(delete_count(), testing::Eq(0));

  context.increment_refcount();

  {
    // This is in a sub-scope because this call to decrement_refcount
    // is expected to give us a shared lock.
    auto result = context.decrement_refcount();
    ASSERT_THAT(
        std::holds_alternative<cow::COWDeleterContext::NotLastReference>(
            result),
        testing::IsTrue());
    ASSERT_THAT(delete_count(), testing::Eq(0));
  }

  {
    auto result = context.decrement_refcount();
    ASSERT_THAT(
        std::holds_alternative<cow::COWDeleterContext::LastReference>(result),
        testing::IsTrue());
    // Result holds the DeleteTracker.
    ASSERT_THAT(delete_count(), testing::Eq(0));
  }

  // When result is deleted, the DeleteTracker is also deleted.
  ASSERT_THAT(delete_count(), testing::Eq(1));
}

TEST_F(ContextTest, cow_deleter) {
  // This is effectively the same thing as decrement_refcount() above.
  auto& context = *new cow::COWDeleterContext(new_delete_tracker());
  ASSERT_THAT(delete_count(), testing::Eq(0));

  cow::cow_deleter(&context);
  ASSERT_THAT(delete_count(), testing::Eq(1));
}

MATCHER(is_copy_on_write, "") {
  const c10::StorageImpl& storage = std::ref(arg);
  return cow::is_cow_data_ptr(storage.data_ptr());
}

TEST(lazy_clone_storage_test, no_context) {
  StorageImpl original_storage(
      {}, /*size_bytes=*/7, GetDefaultCPUAllocator(), /*resizable=*/false);
  ASSERT_THAT(original_storage, testing::Not(is_copy_on_write()));
  ASSERT_TRUE(cow::has_simple_data_ptr(original_storage));

  intrusive_ptr<StorageImpl> new_storage =
      cow::lazy_clone_storage(original_storage);
  ASSERT_THAT(new_storage.get(), testing::NotNull());

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

struct MyDeleterContext {
  MyDeleterContext(void* bytes) : bytes(bytes) {}

  MyDeleterContext(const MyDeleterContext&) = delete;
  MyDeleterContext(MyDeleterContext&&) = delete;
  MyDeleterContext& operator=(const MyDeleterContext&) = delete;
  MyDeleterContext& operator=(MyDeleterContext&&) = delete;
  ~MyDeleterContext() {
    delete[] static_cast<std::byte*>(bytes);
  }

  void* bytes;
};

void my_deleter(void* ctx) {
  delete static_cast<MyDeleterContext*>(ctx);
}

TEST(lazy_clone_storage_test, different_context) {
  void* bytes = new std::byte[5];
  StorageImpl storage(
      {},
      /*size_bytes=*/5,
      at::DataPtr(
          /*data=*/bytes,
          /*ctx=*/new MyDeleterContext(bytes),
          /*ctx_deleter=*/my_deleter,
          /*device=*/Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  // We can't handle an arbitrary context.
  ASSERT_THAT(cow::lazy_clone_storage(storage), testing::IsNull());
}

TEST(lazy_clone_storage_test, already_copy_on_write) {
  std::unique_ptr<void, DeleterFnPtr> data(
      new std::byte[5],
      +[](void* bytes) { delete[] static_cast<std::byte*>(bytes); });
  void* data_ptr = data.get();
  StorageImpl original_storage(
      {},
      /*size_bytes=*/5,
      at::DataPtr(
          /*data=*/data_ptr,
          /*ctx=*/new cow::COWDeleterContext(std::move(data)),
          cow::cow_deleter,
          Device(Device::Type::CPU)),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  ASSERT_THAT(original_storage, is_copy_on_write());

  intrusive_ptr<StorageImpl> new_storage =
      cow::lazy_clone_storage(original_storage);
  ASSERT_THAT(new_storage.get(), testing::NotNull());

  // The result is a different storage.
  ASSERT_THAT(&*new_storage, testing::Ne(&original_storage));
  // But it is also copy-on-write.
  ASSERT_THAT(*new_storage, is_copy_on_write());
  // But they share the same data!
  ASSERT_THAT(new_storage->data(), testing::Eq(original_storage.data()));
}

TEST(materialize_test, not_copy_on_write_context) {
  StorageImpl storage(
      {}, /*size_bytes=*/6, GetCPUAllocator(), /*resizable=*/false);
  ASSERT_THAT(storage, testing::Not(is_copy_on_write()));

  void const* original_data = storage.data();

  // Nothing to materialize.
  ASSERT_THAT(storage.mutable_data(), testing::Eq(original_data));
}

TEST(materialize_test, copy_on_write_single_reference) {
  // A copy-on-write storage with only a single reference can just
  // drop the copy-on-write context upon materialization.
  std::unique_ptr<void, DeleterFnPtr> data(
      new std::byte[4],
      +[](void* bytes) { delete[] static_cast<std::byte*>(bytes); });
  void* data_ptr = data.get();
  StorageImpl storage(
      {},
      /*size_bytes=*/4,
      at::DataPtr(
          /*data=*/data_ptr,
          /*ctx=*/new cow::COWDeleterContext(std::move(data)),
          cow::cow_deleter,
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

bool buffers_are_equal(const void* a, const void* b, size_t nbytes) {
  const char* a_ = static_cast<const char*>(a);
  const char* b_ = static_cast<const char*>(b);

  for (size_t idx = 0; idx < nbytes; idx++) {
    if (a_[idx] != b_[idx]) {
      return false;
    }
  }
  return true;
}

TEST(materialize_test, copy_on_write) {
  StorageImpl original_storage(
      {}, /*size_bytes=*/4, GetCPUAllocator(), /*resizable=*/false);
  std::memcpy(original_storage.mutable_data(), "abcd", 4);
  void const* original_data = original_storage.data();

  auto new_storage = cow::lazy_clone_storage(original_storage);
  ASSERT_THAT(new_storage, testing::NotNull());

  auto context = new_storage->data_ptr().cast_context<cow::COWDeleterContext>(
      cow::cow_deleter);
  ASSERT_THAT(context, testing::NotNull());

  // Materialized storage has new copy of data.
  ASSERT_THAT(new_storage->mutable_data(), testing::Ne(original_data));

  // But the original storage still has the original copy.
  ASSERT_THAT(original_storage.data(), testing::Eq(original_data));

  // And their data is the same
  ASSERT_TRUE(new_storage->nbytes() == original_storage.nbytes());
  ASSERT_TRUE(buffers_are_equal(
      new_storage->data(), original_storage.data(), new_storage->nbytes()));
}

} // namespace
} // namespace c10::impl
// NOLINTEND(clang-analyzer-cplusplus*)
