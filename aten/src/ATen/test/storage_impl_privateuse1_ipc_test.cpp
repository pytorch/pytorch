#include <gtest/gtest.h>

#include <c10/core/StorageImpl.h>

namespace {

c10::intrusive_ptr<c10::StorageImpl> make_privateuse1_storage() {
  return c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      0,
      c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0)),
      nullptr,
      false);
}

} // namespace

TEST(StorageImplReceivedViaIpc, ReceivedViaIpcDefaultFalse) {
  auto storage = make_privateuse1_storage();
  EXPECT_FALSE(storage->received_via_ipc());
}

TEST(StorageImplReceivedViaIpc, SetAndGetReceivedViaIpc) {
  auto storage = make_privateuse1_storage();
  storage->set_received_via_ipc(true);
  EXPECT_TRUE(storage->received_via_ipc());
  storage->set_received_via_ipc(false);
  EXPECT_FALSE(storage->received_via_ipc());
}

TEST(StorageImplReceivedViaIpc, ReceivedViaIpcDoesNotAffectReceivedCuda) {
  auto storage = make_privateuse1_storage();
  storage->set_received_via_ipc(true);
  ASSERT_TRUE(storage->received_via_ipc());
  EXPECT_FALSE(storage->received_cuda());
}
