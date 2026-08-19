#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/core/Event.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/Context.h>

TEST(EventTest, testEventFlag) {
// This intentionally tests the deprecated EventFlag::BACKEND_DEFAULT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  c10::EventFlag flag = c10::EventFlag::BLOCKING | c10::EventFlag::TIMING | c10::EventFlag::INTERPROCESS;
  EXPECT_TRUE(flag & c10::EventFlag::BLOCKING);
  EXPECT_TRUE(flag & c10::EventFlag::TIMING);
  EXPECT_TRUE(flag & c10::EventFlag::INTERPROCESS);
  EXPECT_TRUE(flag == c10::EventFlag::BACKEND_DEFAULT);

  flag = c10::EventFlag::BLOCKING | c10::EventFlag::PYTORCH_DEFAULT;
  EXPECT_TRUE(flag & c10::EventFlag::BLOCKING);
  EXPECT_FALSE(flag & c10::EventFlag::TIMING);
  EXPECT_FALSE(flag & c10::EventFlag::INTERPROCESS);
  EXPECT_FALSE(flag == c10::EventFlag::BACKEND_DEFAULT);

  flag = c10::EventFlag::TIMING | c10::EventFlag::PYTORCH_DEFAULT;
  EXPECT_FALSE(flag & c10::EventFlag::BLOCKING);
  EXPECT_TRUE(flag & c10::EventFlag::TIMING);
  EXPECT_FALSE(flag & c10::EventFlag::INTERPROCESS);
  EXPECT_TRUE(flag == c10::EventFlag::BACKEND_DEFAULT);

  flag = c10::EventFlag::BACKEND_DEFAULT;
  EXPECT_FALSE(flag & c10::EventFlag::BLOCKING);
  EXPECT_TRUE(flag & c10::EventFlag::TIMING);
  EXPECT_FALSE(flag & c10::EventFlag::INTERPROCESS);
  EXPECT_TRUE(flag == c10::EventFlag::BACKEND_DEFAULT);
#pragma GCC diagnostic pop

  if (at::accelerator::deviceCount() <= 0) {
    GTEST_SKIP() << "No accelerator device available";
  }

  auto device_type = at::accelerator::getAccelerator(true).value();

  flag = c10::EventFlag::BLOCKING | c10::EventFlag::TIMING;
  auto event1 = c10::Event(device_type, flag);
  EXPECT_TRUE(event1.flag() == flag);

  auto stream = at::accelerator::getCurrentStream(0);
  at::globalContext().lazyInitDevice(device_type);

  auto t = at::ones({10, 10}, at::dtype(at::kFloat));
  event1.record(stream);
  auto t2 = t.to(device_type);
  t2 = t2 * 2;

  auto event2 = c10::Event(device_type, flag);
  event2.record(stream);
  event2.synchronize();

  EXPECT_TRUE(event1.elapsedTime(event2) > 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
