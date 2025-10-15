#include <algorithm>
#include <optional>

#include <c10/util/ArrayRef.h>
#include <c10/util/Logging.h>
#include <gtest/gtest.h>

namespace c10_test {

using std::set;
using std::string;
using std::vector;

TEST(LoggingTest, TestEnforceTrue) {
  // This should just work.
  CAFFE_ENFORCE(true, "Isn't it?");
}

TEST(LoggingTest, TestEnforceFalse) {
  bool kFalse = false;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
  try {
    CAFFE_ENFORCE(false, "This throws.");
    // This should never be triggered.
    ADD_FAILURE();
    // NOLINTNEXTLINE(*catch*)
  } catch (const ::c10::Error&) {
  }
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kFalse);
}

TEST(LoggingTest, TestEnforceEquals) {
  int x = 4;
  int y = 5;
  int z = 0;
  try {
    CAFFE_ENFORCE_THAT(std::equal_to<void>(), ==, ++x, ++y, "Message: ", z++);
    // This should never be triggered.
    ADD_FAILURE();
  } catch (const ::c10::Error& err) {
    auto errStr = std::string(err.what());
    EXPECT_NE(errStr.find("5 vs 6"), string::npos);
    EXPECT_NE(errStr.find("Message: 0"), string::npos);
  }

  // arguments are expanded only once
  CAFFE_ENFORCE_THAT(std::equal_to<void>(), ==, ++x, y);
  EXPECT_EQ(x, 6);
  EXPECT_EQ(y, 6);
  EXPECT_EQ(z, 1);
}

namespace {
struct EnforceEqWithCaller {
  void test(const char* x) {
    CAFFE_ENFORCE_EQ_WITH_CALLER(1, 1, "variable: ", x, " is a variable");
  }
};
} // namespace

TEST(LoggingTest, TestEnforceMessageVariables) {
  const char* const x = "hello";
  CAFFE_ENFORCE_EQ(1, 1, "variable: ", x, " is a variable");

  EnforceEqWithCaller e;
  e.test(x);
}

TEST(
    LoggingTest,
    EnforceEqualsObjectWithReferenceToTemporaryWithoutUseOutOfScope) {
  std::vector<int> x = {1, 2, 3, 4};
  // This case is a little tricky. We have a temporary
  // std::initializer_list to which our temporary ArrayRef
  // refers. Temporary lifetime extension by binding a const reference
  // to the ArrayRef doesn't extend the lifetime of the
  // std::initializer_list, just the ArrayRef, so we end up with a
  // dangling ArrayRef. This test forces the implementation to get it
  // right.
  CAFFE_ENFORCE_EQ(x, (at::ArrayRef<int>{1, 2, 3, 4}));
}

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct Noncopyable {
  int x;

  explicit Noncopyable(int a) : x(a) {}

  Noncopyable(const Noncopyable&) = delete;
  Noncopyable(Noncopyable&&) = delete;
  Noncopyable& operator=(const Noncopyable&) = delete;
  Noncopyable& operator=(Noncopyable&&) = delete;

  bool operator==(const Noncopyable& rhs) const {
    return x == rhs.x;
  }
};

std::ostream& operator<<(std::ostream& out, const Noncopyable& nc) {
  out << "Noncopyable(" << nc.x << ")";
  return out;
}
} // namespace

TEST(LoggingTest, DoesntCopyComparedObjects) {
  CAFFE_ENFORCE_EQ(Noncopyable(123), Noncopyable(123));
}

TEST(LoggingTest, EnforceShowcase) {
  // It's not really a test but rather a convenient thing that you can run and
  // see all messages
  int one = 1;
  int two = 2;
  int three = 3;
#define WRAP_AND_PRINT(exp)                    \
  try {                                        \
    exp;                                       \
  } catch (const ::c10::Error&) {              \
    /* ::c10::Error already does LOG(ERROR) */ \
  }
  WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_NE(one * 2, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_GT(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_GE(one, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_LT(three, two));
  WRAP_AND_PRINT(CAFFE_ENFORCE_LE(three, two));

  WRAP_AND_PRINT(CAFFE_ENFORCE_EQ(
      one * two + three, three * two, "It's a pretty complicated expression"));

  WRAP_AND_PRINT(CAFFE_ENFORCE_THAT(
      std::equal_to<void>(), ==, one * two + three, three * two));
}

TEST(LoggingTest, Join) {
  auto s = c10::Join(", ", vector<int>({1, 2, 3}));
  EXPECT_EQ(s, "1, 2, 3");
  s = c10::Join(":", vector<string>());
  EXPECT_EQ(s, "");
  s = c10::Join(", ", set<int>({3, 1, 2}));
  EXPECT_EQ(s, "1, 2, 3");
}

TEST(LoggingTest, TestDanglingElse) {
  if (true)
    TORCH_DCHECK_EQ(1, 1);
  else
    GTEST_FAIL();
}

#if GTEST_HAS_DEATH_TEST
TEST(LoggingDeathTest, TestEnforceUsingFatal) {
  bool kTrue = true;
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_DEATH(CAFFE_ENFORCE(false, "This goes fatal."), "");
  std::swap(FLAGS_caffe2_use_fatal_for_enforce, kTrue);
}
#endif

#ifdef FBCODE_CAFFE2
static C10_NOINLINE void f1() {
  CAFFE_THROW("message");
}

static C10_NOINLINE void f2() {
  f1();
}

static C10_NOINLINE void f3() {
  f2();
}
TEST(LoggingTest, ExceptionWhat) {
  std::optional<::c10::Error> error;
  try {
    f3();
  } catch (const ::c10::Error& e) {
    error = e;
  }

  ASSERT_TRUE(error);
  std::string what = error->what();

  EXPECT_TRUE(what.find("c10_test::f1()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f2()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f3()") != std::string::npos) << what;

  // what() should be recomputed.
  error->add_context("NewContext");
  what = error->what();
  EXPECT_TRUE(what.find("c10_test::f1()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f2()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("c10_test::f3()") != std::string::npos) << what;
  EXPECT_TRUE(what.find("NewContext") != std::string::npos) << what;
}
#endif

TEST(LoggingTest, LazyBacktrace) {
  struct CountingLazyString : ::c10::OptimisticLazyValue<std::string> {
    mutable size_t invocations{0};

    std::string compute() const override {
      ++invocations;
      return "A string";
    }
  };

  auto backtrace = std::make_shared<CountingLazyString>();
  ::c10::Error ex("", backtrace);
  // The backtrace is not computed on construction, and then it is not computed
  // more than once.
  EXPECT_EQ(backtrace->invocations, 0);
  const char* w1 = ex.what();
  EXPECT_EQ(backtrace->invocations, 1);
  const char* w2 = ex.what();
  EXPECT_EQ(backtrace->invocations, 1);
  // what() should not be recomputed.
  EXPECT_EQ(w1, w2);

  ex.add_context("");
  ex.what();
  EXPECT_EQ(backtrace->invocations, 1);
}

} // namespace c10_test
