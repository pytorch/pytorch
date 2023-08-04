#include <gtest/gtest.h>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/StorageUtils.h>

using namespace ::testing;

TEST(StorageUtilsTest, shm_storage_refcount) {
  auto t1 = std::make_unique<at::Tensor>(
      at::full({5, 5}, 7, at::dtype(at::kLong).device(at::kCPU)));
  auto t2 = std::make_unique<at::Tensor>(t1->slice(0, 0, 3));

  auto verificationTensor = t1->clone();
  ASSERT_EQ(t1->storage().use_count(), 2);
  ASSERT_EQ(t2->storage().use_count(), 2);
  ASSERT_EQ(verificationTensor.storage().use_count(), 1);

  at::share_memory_(*t1);
  ASSERT_EQ(t1->storage().allocator(), nullptr)
      << "Expect original storage allocator to be detached";
  ASSERT_NE(verificationTensor.storage().allocator(), nullptr);
  ASSERT_EQ(t1->storage().use_count(), 2) << "Expect refcount to be the same";
  ASSERT_EQ(t2->storage().use_count(), 2);

  ASSERT_TRUE(t1->equal(verificationTensor));
  auto weakStoragePtr = t1->storage().getWeakStorageImpl();
  // weak + 1 (if any strong ref exists due to how intrusive_ptr refcount works)
  ASSERT_EQ(weakStoragePtr.weak_use_count(), 2);
  t1.reset();
  t2.reset();
  ASSERT_TRUE(weakStoragePtr.expired());
}
