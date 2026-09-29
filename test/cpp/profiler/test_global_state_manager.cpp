// Unit tests for GlobalStateManager in torch/csrc/profiler/util.h.
//
// The profiler use-after-free fix rests on GlobalStateManager::get() returning
// an owning shared_ptr: a caller that has already fetched the state keeps it
// alive even after the manager drops its own reference in pop(). These tests
// pin that ownership contract. They are single threaded; the contract is what
// makes the concurrent teardown path safe, so it can be verified without
// threads or a sanitizer.

#include <gtest/gtest.h>

#include <memory>

#include <torch/csrc/profiler/util.h>

namespace {

// Each test uses a distinct tag type so the per-type process-global singleton
// inside GlobalStateManager stays isolated across tests, which may run in any
// order within the binary.
template <int Tag>
struct Tracked {
  explicit Tracked(int* destroyed_count) : destroyed_count_(destroyed_count) {}
  ~Tracked() {
    ++(*destroyed_count_);
  }
  int* destroyed_count_;
};

} // namespace

using torch::profiler::impl::GlobalStateManager;

TEST(GlobalStateManagerTest, GetReturnsOwningHandleThatOutlivesPop) {
  int destroyed = 0;
  GlobalStateManager<Tracked<0>>::push(
      std::make_shared<Tracked<0>>(&destroyed));

  auto held = GlobalStateManager<Tracked<0>>::get();
  ASSERT_NE(held, nullptr);

  // pop() drops the manager reference, but held still owns the object.
  auto popped = GlobalStateManager<Tracked<0>>::pop();
  ASSERT_EQ(held.get(), popped.get());

  popped.reset();
  EXPECT_EQ(destroyed, 0); // still alive: held still owns it

  held.reset();
  EXPECT_EQ(destroyed, 1); // last owner released

  EXPECT_EQ(GlobalStateManager<Tracked<0>>::get(), nullptr);
}

TEST(GlobalStateManagerTest, GetAndPopOnEmptyReturnNull) {
  // Tracked<1>'s singleton is never pushed to, so it starts empty.
  EXPECT_EQ(GlobalStateManager<Tracked<1>>::get(), nullptr);
  EXPECT_EQ(GlobalStateManager<Tracked<1>>::pop(), nullptr);
}

TEST(GlobalStateManagerTest, PushWhenAlreadySetIsNoOp) {
  int first_destroyed = 0;
  auto first = std::make_shared<Tracked<2>>(&first_destroyed);
  auto* first_raw = first.get();
  GlobalStateManager<Tracked<2>>::push(std::move(first));

  // A second push must not replace the state already held by the manager.
  int second_destroyed = 0;
  GlobalStateManager<Tracked<2>>::push(
      std::make_shared<Tracked<2>>(&second_destroyed));
  EXPECT_EQ(second_destroyed, 1); // the ignored push drops its object
  EXPECT_EQ(GlobalStateManager<Tracked<2>>::get().get(), first_raw);

  // Release the stored object so it does not outlive first_destroyed.
  GlobalStateManager<Tracked<2>>::pop();
  EXPECT_EQ(first_destroyed, 1);
}
