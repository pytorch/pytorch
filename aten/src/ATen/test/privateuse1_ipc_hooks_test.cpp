#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

// Concrete subclass using only the default implementations.
struct DefaultHooks : at::PrivateUse1HooksInterface {};

TEST(PrivateUse1IpcHooks, supportsIpcDefaultFalse) {
  DefaultHooks h;
  EXPECT_FALSE(h.supportsIpc());
}

TEST(PrivateUse1IpcHooks, requiresEventSyncThrows) {
  DefaultHooks h;
  EXPECT_THROW(h.requiresEventSync(), c10::NotImplementedError);
}

TEST(PrivateUse1IpcHooks, getIpcMemHandleThrows) {
  DefaultHooks h;
  EXPECT_THROW(h.getIpcMemHandle(nullptr), c10::NotImplementedError);
}

TEST(PrivateUse1IpcHooks, getIpcEventHandleThrows) {
  DefaultHooks h;
  EXPECT_THROW(h.getIpcEventHandle(), c10::NotImplementedError);
}

TEST(PrivateUse1IpcHooks, openIpcMemHandleThrows) {
  DefaultHooks h;
  EXPECT_THROW(h.openIpcMemHandle(""), c10::NotImplementedError);
}

TEST(PrivateUse1IpcHooks, waitIpcEventThrows) {
  DefaultHooks h;
  c10::Stream s = c10::Stream::unpack3(0, 0, c10::DeviceType::CPU);
  EXPECT_THROW(h.waitIpcEvent("", s), c10::NotImplementedError);
}