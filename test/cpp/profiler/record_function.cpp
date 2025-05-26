#include <array>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <c10/util/irange.h>

// Test that we can add and remove callbacks (both global and thread local.)
TEST(RecordFunctionTest, AddRemove) {
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());

  auto start_callback =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    return nullptr;
  };
  auto end_callback = [](const at::RecordFunction& fn, at::ObserverContext*) {};

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(start_callback, end_callback));

  ASSERT_TRUE(at::hasCallbacks());
  ASSERT_TRUE(at::hasThreadLocalCallbacks());
  ASSERT_FALSE(at::hasGlobalCallbacks());

  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());

  handle = at::addGlobalCallback(
      at::RecordFunctionCallback(start_callback, end_callback));

  ASSERT_TRUE(at::hasCallbacks());
  ASSERT_FALSE(at::hasThreadLocalCallbacks());
  ASSERT_TRUE(at::hasGlobalCallbacks());

  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());
}

// Test that the callbacks that we register are actually run.
TEST(RecordFunctionTest, ThreadLocalState) {
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());

  static int tls_test_start_counter;
  static int tls_test_end_counter;
  tls_test_start_counter = 0;
  tls_test_end_counter = 0;

  auto start_callback =
      [](const at::RecordFunction&) -> std::unique_ptr<at::ObserverContext> {
    ++tls_test_start_counter;
    return nullptr;
  };
  auto end_callback = [](const at::RecordFunction&, at::ObserverContext*) {
    ++tls_test_end_counter;
  };

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(start_callback, end_callback));

  {
    at::RecordFunction guard(at::RecordScope::USER_SCOPE);
    guard.before("Test");
    EXPECT_EQ(tls_test_start_counter, 1);
    EXPECT_EQ(tls_test_end_counter, 0);
  }
  EXPECT_EQ(tls_test_start_counter, 1);
  EXPECT_EQ(tls_test_end_counter, 1);

  {
    tls_test_start_counter = 0;
    tls_test_end_counter = 0;
    at::DisableRecordFunctionGuard no_profile_guard;
    at::RecordFunction guard(at::RecordScope::USER_SCOPE);
    guard.before("Test");
    EXPECT_EQ(tls_test_start_counter, 0);
    EXPECT_EQ(tls_test_end_counter, 0);
  }
  EXPECT_EQ(tls_test_start_counter, 0);
  EXPECT_EQ(tls_test_end_counter, 0);

  {
    tls_test_start_counter = 0;
    tls_test_end_counter = 0;
    RECORD_FUNCTION("Test", {});
    EXPECT_EQ(tls_test_start_counter, 1);
    EXPECT_EQ(tls_test_end_counter, 0);
  }
  EXPECT_EQ(tls_test_start_counter, 1);
  EXPECT_EQ(tls_test_end_counter, 1);

  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());
}

// Test that callbacks are run in the order that they are registered.
TEST(RecordFunctionTest, CallOrder) {
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());

  static int current_index;
  current_index = 0;

  static std::array<std::string, 8> expected_order = {
      "Start Callback 0 Outer",
      "Start Callback 1 Outer",
      "Start Callback 0 Inner",
      "Start Callback 1 Inner",
      "End Callback 0 Inner",
      "End Callback 1 Inner",
      "End Callback 0 Outer",
      "End Callback 1 Outer",
  };

#define REGISTER_CALLBACK(index)                                       \
  at::addThreadLocalCallback(                                          \
      at::RecordFunctionCallback(                                      \
          [](const at::RecordFunction& fn)                             \
              -> std::unique_ptr<at::ObserverContext> {                \
            EXPECT_EQ(                                                 \
                fmt::format("Start Callback {} {}", index, fn.name()), \
                expected_order[current_index++]);                      \
            return nullptr;                                            \
          },                                                           \
          [](const at::RecordFunction& fn, at::ObserverContext*) {     \
            EXPECT_EQ(                                                 \
                fmt::format("End Callback {} {}", index, fn.name()),   \
                expected_order[current_index++]);                      \
          })                                                           \
          .scopes({at::RecordScope::FUNCTION}))

  REGISTER_CALLBACK(0);
  REGISTER_CALLBACK(1);
#undef REGISTER_CALLBACK

  RECORD_FUNCTION("Outer", {});
  { RECORD_FUNCTION("Inner", {}); }

  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());
}

// Make sure TLS migrates when tasks are launched.
TEST(RecordFunctionTest, ThreadMigration) {
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());

  static int call_count;
  call_count = 0;

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction&)
              -> std::unique_ptr<at::ObserverContext> { return nullptr; },
          [](const at::RecordFunction&, at::ObserverContext*) { ++call_count; })
          .scopes({at::RecordScope::FUNCTION}));

  EXPECT_EQ(call_count, 0);

  std::condition_variable cv;
  std::mutex lock;
  at::launch([&cv]() {
    RECORD_FUNCTION("Test", {});
    cv.notify_all();
  });
  auto guard = std::unique_lock<std::mutex>(lock);
  cv.wait(guard, [] { return call_count > 0; });

  EXPECT_EQ(call_count, 1);

  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());
}

// Test sampling logic and validate that callbacks fire at the correct times.
TEST(RecordFunctionTest, Sampling) {
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());

  static int sample_test_counter;
  sample_test_counter = 0;

  uint32_t seed = 12345;
  double p = 0.25;

  at::set_record_function_seed_for_testing(seed);
  std::mt19937 generator;
  generator.seed(seed);
  auto dist = std::geometric_distribution<int>(p);

  // Make sure we know which steps should fire.
  auto outcomes = std::array<int, 5>{7, 0, 0, 6, 2};
  for (const auto i : c10::irange(outcomes.size())) {
    ASSERT_EQ(dist(generator), outcomes[i]);
  }

  std::vector<int> expected_counts;
  int running_count = 0;
  for (const auto i : c10::irange(outcomes.size())) {
    for ([[maybe_unused]] const auto j : c10::irange(outcomes[i])) {
      expected_counts.push_back(running_count);
    }
    expected_counts.push_back(++running_count);
  }

  auto start_callback =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    ++sample_test_counter;
    return nullptr;
  };
  auto end_callback = [](const at::RecordFunction& fn, at::ObserverContext*) {};

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(start_callback, end_callback)
          .samplingProb(p)
          .scopes({at::RecordScope::FUNCTION}));

  for (const auto i : c10::irange(expected_counts.size())) {
    RECORD_FUNCTION("Test", {});
    EXPECT_EQ(sample_test_counter, expected_counts[i]);
  }

  at::removeCallback(handle);
  ASSERT_FALSE(at::hasCallbacks());
}

// Validate sampling against a simple reference implementation for a complex set
// of registered callbacks.
TEST(RecordFunctionTest, MultipleCallbacks) {
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());

  uint32_t seed = 54321;

  std::mt19937 generator;
  generator.seed(seed);

  auto sample = [&](double p) {
    return (p < 1.0 ? std::geometric_distribution<int>(p)(generator) : 0) + 1;
  };

  std::array<double, 4> probabilities{0.1, 1.0, 1.0, 0.3};
  std::array<int, 4> next_call;
  std::array<int, 4> counts;
  static std::array<int, 4> counts_from_rec_fn;
  counts_from_rec_fn.fill(0);

  auto end_callback = [](const at::RecordFunction& fn, at::ObserverContext*) {};

#define REGISTER_CALLBACK(register_fn, index)                   \
  register_fn(at::RecordFunctionCallback(                       \
                  [](const at::RecordFunction& fn)              \
                      -> std::unique_ptr<at::ObserverContext> { \
                    ++counts_from_rec_fn[index];                \
                    return nullptr;                             \
                  },                                            \
                  end_callback)                                 \
                  .samplingProb(probabilities[index])           \
                  .scopes({at::RecordScope::FUNCTION}))

  REGISTER_CALLBACK(at::addGlobalCallback, 0);
  REGISTER_CALLBACK(at::addGlobalCallback, 1);
  REGISTER_CALLBACK(at::addThreadLocalCallback, 2);

  // The RecordFunction machinery will rebuild callbacks whenever a new observer
  // is registered, so we need to wait until the last callback to seed the
  // random number generator.
  at::set_record_function_seed_for_testing(seed);
  REGISTER_CALLBACK(at::addThreadLocalCallback, 3);
#undef REGISTER_CALLBACK

  for (const auto i : c10::irange(probabilities.size())) {
    next_call[i] = sample(probabilities[i]);
  }

  for ([[maybe_unused]] const auto i : c10::irange(50)) {
    RECORD_FUNCTION("Test", {});
    for (const auto j : c10::irange(next_call.size())) {
      if (!(--next_call[j])) {
        ++counts[j];
        next_call[j] = sample(probabilities[j]);
      }
      EXPECT_EQ(counts[j], counts_from_rec_fn[j]);
    }
  }

  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());
}

// Test that KwargsOnly callbacks are run in USER_SCOPE.
TEST(RecordFunctionTest, KwargsOnly) {
  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());
  static const std::unordered_map<std::string, c10::IValue> myMap = {
      {"a", 1}, {"b", 2.5}};

#define REGISTER_CALLBACK()                                          \
  at::addThreadLocalCallback(                                        \
      at::RecordFunctionCallback(                                    \
          [](const at::RecordFunction& fn)                           \
              -> std::unique_ptr<at::ObserverContext> {              \
            EXPECT_EQ(myMap, fn.kwinputs());                         \
            return nullptr;                                          \
          },                                                         \
          [](const at::RecordFunction& fn, at::ObserverContext*) {}) \
          .needsInputs(true)                                         \
          .scopes({at::RecordScope::USER_SCOPE}))

  REGISTER_CALLBACK();
#undef REGISTER_CALLBACK

  RECORD_USER_SCOPE_WITH_KWARGS_ONLY("Test", &myMap);

  at::clearCallbacks();
  ASSERT_FALSE(at::hasCallbacks());
}
