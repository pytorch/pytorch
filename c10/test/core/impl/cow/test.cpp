#include <c10/core/impl/cow/shadow_storage.h>
#include <c10/core/impl/cow/state_machine.h>
#include <c10/util/Exception.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utility>

namespace c10 {

// Provide a GoogleTest printer for the Warning type.
void PrintTo(Warning const& warning, std::ostream* out) {
  *out << warning.msg();
}
} // namespace c10

namespace c10::impl::cow {
namespace {

// Implements WarningHandler and just stores seen warnings.
//
// Users are responsible for ensuring that all warnings are checked
// for: any uninspected warnings will cause a test failure.
class CapturingWarningHandler final : public WarningHandler {
 public:
  ~CapturingWarningHandler() final {
    // Ensure that users do not leave uninspected warnings.
    EXPECT_THAT(release_warnings(), testing::IsEmpty());
  }

  // See WarningHandler::process.
  auto process(Warning const& warning) -> void final {
    warnings_.push_back(warning);
  }

  // Releases any captured warnings to the user.
  auto release_warnings() -> std::vector<Warning> {
    return std::move(warnings_);
  }

 private:
  // All captured warnings.
  std::vector<Warning> warnings_;
};

// A test fixture that captures any warnings.
class CaptureWarningsTest : public testing::Test {
 protected:
  // Releases any captured warnings to the caller.
  auto release_warnings() -> std::vector<Warning> {
    return warning_handler_.release_warnings();
  }

 private:
  // We install our capturing warning handler for the scope of each
  // test.
  CapturingWarningHandler warning_handler_;
  WarningUtils::WarningHandlerGuard override_warning_handler_{
      &warning_handler_};
};

// We don't need any other test setup, just capture any warnings in
// our test fixture.
using CopyOnWriteTest = CaptureWarningsTest;

TEST_F(CopyOnWriteTest, ShadowStorage) {
  cow::ShadowStorage shadow_storage(/*generation=*/0);
  ASSERT_THAT(shadow_storage.generation(), testing::Eq(0));
  shadow_storage.update_from_physical_generation(1); // must not warn
  ASSERT_THAT(shadow_storage.generation(), testing::Eq(1));
}

TEST_F(CopyOnWriteTest, StateMachineWithoutShadowStorage) {
  cow::StateMachine state;
  state.maybe_bump(nullptr);
  ASSERT_THAT(state.physical_generation(), testing::Eq(c10::nullopt));
}

// Implement a matcher that verifies a warning has a message as a
// substring.
MATCHER_P(IsWarning, msg, "") {
  return arg.msg().find(msg) != std::string::npos;
}

// Test our behavior with multiple view families.
TEST_F(CopyOnWriteTest, MultipleViewFamilies) {
  cow::StateMachine state;
  intrusive_ptr<cow::ShadowStorage> family_2 =
      state.simulate_lazy_copy(nullptr);

  // Bump the second view family.
  state.maybe_bump(&*family_2);

  ASSERT_THAT(release_warnings(), testing::IsEmpty());

  WarningUtils::WarnAlways enable_warn_always_mode;

  // Bump the first view family. This will warn since we've bumped,
  // i.e. written to, the second family.
  state.maybe_bump(/*shadow_storage=*/nullptr);

  ASSERT_THAT(
      release_warnings(),
      testing::ElementsAre(IsWarning(
          "You have written through to both aliases created by calling reshape")));

  // This will also warn, because we've just written the first view
  // family.
  state.maybe_bump(&*family_2);
  ASSERT_THAT(
      release_warnings(),
      testing::ElementsAre(IsWarning(
          "You have written through to both aliases created by calling reshape")));
}

// This test is mostly identical to the previous test, except it
// doesn't run in WarnAlways mode. This verifies that the warn-once
// setting is respected.
TEST_F(CopyOnWriteTest, OnlyWarnsOnce) {
  cow::StateMachine state;
  intrusive_ptr<cow::ShadowStorage> family_2 =
      state.simulate_lazy_copy(nullptr);

  // Bump the second view family.
  state.maybe_bump(&*family_2);

  // Bump the first view family. This will warn since we've bumped,
  // i.e. written to, the second family.
  state.maybe_bump(/*shadow_storage=*/nullptr);

  ASSERT_THAT(
      release_warnings(),
      testing::ElementsAre(IsWarning(
          "You have written through to both aliases created by calling reshape")));

  // This should also warn, because we've just written the first view
  // family. However, the code is configured to only warn once, so it
  // does not.
  state.maybe_bump(&*family_2);
  ASSERT_THAT(release_warnings(), testing::IsEmpty());
}

} // namespace
} // namespace c10::impl::cow
