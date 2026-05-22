#include <c10/util/Exception.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string_view>

namespace c10 {
namespace {

struct MyDebugInfo : public DebugInfoBase {
  explicit MyDebugInfo(int v) : value(v) {}
  int value;
};

TEST(ThreadLocalDebugInfo, GetCustomKind) {
  static constexpr std::string_view kKindStr = "CustomKind_Push";
  const DebugInfoKind kKind(&kKindStr);

  auto info = std::make_shared<MyDebugInfo>(42);
  ThreadLocalDebugInfo::_push(kKind, info);

  EXPECT_EQ(ThreadLocalDebugInfo::get(kKind), info.get());

  ThreadLocalDebugInfo::_pop(kKind);
}

TEST(ThreadLocalDebugInfo, PeekCustomKind) {
  static constexpr std::string_view kKindStr = "CustomKind_Peek";
  const DebugInfoKind kKind(&kKindStr);

  auto info = std::make_shared<MyDebugInfo>(42);
  ThreadLocalDebugInfo::_push(kKind, info);

  EXPECT_EQ(ThreadLocalDebugInfo::_peek(kKind), info);

  ThreadLocalDebugInfo::_pop(kKind);
}

TEST(ThreadLocalDebugInfo, PopCustomKind) {
  static constexpr std::string_view kKindStr = "CustomKind_Pop";
  const DebugInfoKind kKind(&kKindStr);

  auto info = std::make_shared<MyDebugInfo>(42);
  ThreadLocalDebugInfo::_push(kKind, info);

  EXPECT_EQ(ThreadLocalDebugInfo::_pop(kKind), info);
}

TEST(ThreadLocalDebugInfo, PopThrowsOnMismatch) {
  static constexpr std::string_view kKindStr1 = "CustomKind_PopMismatch1";
  const DebugInfoKind kKind1(&kKindStr1);
  static constexpr std::string_view kKindStr2 = "CustomKind_PopMismatch2";
  const DebugInfoKind kKind2(&kKindStr2);

  auto info = std::make_shared<MyDebugInfo>(42);
  ThreadLocalDebugInfo::_push(kKind1, info);

  EXPECT_THROW(ThreadLocalDebugInfo::_pop(kKind2), c10::Error);

  ThreadLocalDebugInfo::_pop(kKind1);
}

TEST(ThreadLocalDebugInfo, PeekThrowsOnMismatch) {
  static constexpr std::string_view kKindStr1 = "CustomKind_PeekMismatch1";
  const DebugInfoKind kKind1(&kKindStr1);
  static constexpr std::string_view kKindStr2 = "CustomKind_PeekMismatch2";
  const DebugInfoKind kKind2(&kKindStr2);

  auto info = std::make_shared<MyDebugInfo>(42);
  ThreadLocalDebugInfo::_push(kKind1, info);

  EXPECT_THROW(ThreadLocalDebugInfo::_peek(kKind2), c10::Error);

  ThreadLocalDebugInfo::_pop(kKind1);
}

TEST(DebugInfoKind, CtorThrowsOnNull) {
  EXPECT_THROW({ const DebugInfoKind kKind1(nullptr); }, c10::Error);
  const std::string_view* const kNullKindStr = nullptr;
  EXPECT_THROW(const DebugInfoKind kKind2(kNullKindStr), c10::Error);
}

TEST(DebugInfoKind, DifferentPointersWithSameStringAreDifferentKinds) {
  static constexpr std::string_view kKindStr1 = "CustomKind";
  static constexpr std::string_view kKindStr2 = "CustomKind";
  const DebugInfoKind kKind1(&kKindStr1);
  const DebugInfoKind kKind2(&kKindStr2);
  EXPECT_NE(kKind1, kKind2);
}

TEST(DebugInfoKind, CanBePutInSet) {
  std::set<DebugInfoKind> kinds;
  kinds.insert(DebugInfoKind::PRODUCER_INFO);
  kinds.insert(DebugInfoKind::PROFILER_STATE);
  EXPECT_EQ(kinds.size(), 2);
  EXPECT_TRUE(kinds.contains(DebugInfoKind::PRODUCER_INFO));
  EXPECT_TRUE(kinds.contains(DebugInfoKind::PROFILER_STATE));
}

} // namespace
} // namespace c10
