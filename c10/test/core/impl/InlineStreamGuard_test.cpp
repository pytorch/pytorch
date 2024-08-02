#include <gtest/gtest.h>

#include <c10/core/impl/FakeGuardImpl.h>
#include <c10/core/impl/InlineStreamGuard.h>

using namespace c10;
using namespace c10::impl;

constexpr auto TestDeviceType = DeviceType::CUDA;
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;

static Device dev(DeviceIndex index) {
  return Device(TestDeviceType, index);
}

static Stream stream(DeviceIndex index, StreamId sid) {
  return Stream(Stream::UNSAFE, dev(index), sid);
}

// -- InlineStreamGuard -------------------------------------------------------

using TestGuard = InlineStreamGuard<TestGuardImpl>;

TEST(InlineStreamGuard, Constructor) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 2));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineStreamGuard, ResetStreamSameSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(0, 2));
    g.reset_stream(stream(0, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 3);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(0, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(0));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineStreamGuard, ResetStreamDifferentSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    g.reset_stream(stream(1, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineStreamGuard, ResetStreamDifferentDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    g.reset_stream(stream(2, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(2, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(2));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// -- OptionalInlineStreamGuard
// -------------------------------------------------------

using OptionalTestGuard = InlineOptionalStreamGuard<TestGuardImpl>;

TEST(InlineOptionalStreamGuard, Constructor) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    OptionalTestGuard g(stream(1, 2));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), std::make_optional(stream(0, 0)));
    ASSERT_EQ(g.current_stream(), std::make_optional(stream(1, 2)));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  {
    OptionalTestGuard g(std::make_optional(stream(1, 2)));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), std::make_optional(stream(0, 0)));
    ASSERT_EQ(g.current_stream(), std::make_optional(stream(1, 2)));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  {
    OptionalTestGuard g;
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineOptionalStreamGuard, ResetStreamSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    OptionalTestGuard g;
    g.reset_stream(stream(1, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), std::make_optional(stream(0, 0)));
    ASSERT_EQ(g.current_stream(), std::make_optional(stream(1, 3)));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineOptionalStreamGuard, ResetStreamDifferentDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    OptionalTestGuard g;
    g.reset_stream(stream(2, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), std::make_optional(stream(0, 0)));
    ASSERT_EQ(g.current_stream(), std::make_optional(stream(2, 3)));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// -- InlineMultiStreamGuard
// -------------------------------------------------------

using MultiTestGuard = InlineMultiStreamGuard<TestGuardImpl>;

TEST(InlineMultiStreamGuard, Constructor) {
  TestGuardImpl::resetStreams();
  {
    std::vector<Stream> streams;
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  {
    std::vector<Stream> streams = {stream(0, 2)};
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  {
    std::vector<Stream> streams = {stream(1, 3)};
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  {
    std::vector<Stream> streams = {stream(0, 2), stream(1, 3)};
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
}
