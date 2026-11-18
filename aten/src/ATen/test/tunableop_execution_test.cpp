#include <gtest/gtest.h>

#include <ATen/ThreadLocalState.h>
#include <ATen/TunableOpExecution.h>
#include <c10/util/Exception.h>

#include <future>
#include <memory>
#include <thread>

namespace {

at::TunableOpBinding enabledBinding(uint64_t generation) {
  return std::make_shared<const at::TunableOpExecutionBinding>(
      at::TunableOpExecutionBinding{
          at::TunableOpPermission::Enabled, 0, generation});
}

class TunableOpExecutionTest : public ::testing::Test {
 protected:
  at::TunableOpExecutionGuard restore_binding_{nullptr};
};

TEST_F(TunableOpExecutionTest, DisabledAndUnboundPermissions) {
  EXPECT_FALSE(at::GetTunableOpBinding());
  const auto disabled = at::DisabledTunableOpBinding();
  at::SetTunableOpBinding(disabled);
  EXPECT_EQ(at::GetTunableOpBinding(), disabled);
  EXPECT_EQ(disabled->permission, at::TunableOpPermission::Disabled);
  at::SetTunableOpBinding(nullptr);
  EXPECT_FALSE(at::GetTunableOpBinding());
}

TEST_F(TunableOpExecutionTest, NestedGuardsRestorePermission) {
  const auto enabled = enabledBinding(7);
  const auto other = enabledBinding(8);
  {
    at::TunableOpExecutionGuard outer(enabled);
    EXPECT_EQ(at::GetTunableOpBinding(), enabled);
    {
      at::TunableOpExecutionGuard denied(at::DisabledTunableOpBinding());
      EXPECT_EQ(
          at::GetTunableOpBinding()->permission,
          at::TunableOpPermission::Disabled);
      {
        at::TunableOpExecutionGuard inner(other);
        EXPECT_EQ(at::GetTunableOpBinding(), other);
      }
      EXPECT_EQ(at::GetTunableOpBinding(), at::DisabledTunableOpBinding());
    }
    EXPECT_EQ(at::GetTunableOpBinding(), enabled);
  }
  EXPECT_FALSE(at::GetTunableOpBinding());
}

TEST_F(TunableOpExecutionTest, ExceptionRestoresBinding) {
  const auto enabled = enabledBinding(7);
  at::TunableOpExecutionGuard outer(enabled);
  EXPECT_THROW(
      {
        at::TunableOpExecutionGuard inner(at::DisabledTunableOpBinding());
        TORCH_CHECK(false, "execution failed");
      },
      c10::Error);
  EXPECT_EQ(at::GetTunableOpBinding(), enabled);
}

TEST_F(TunableOpExecutionTest, SnapshotInstallsAndRestoresOnReusedWorker) {
  const auto first = enabledBinding(7);
  const auto second = enabledBinding(8);
  at::SetTunableOpBinding(first);
  const at::ThreadLocalState first_state;
  at::SetTunableOpBinding(second);
  const at::ThreadLocalState second_state;
  at::SetTunableOpBinding(nullptr);
  const at::ThreadLocalState unbound_state;

  std::thread worker([&] {
    EXPECT_FALSE(at::GetTunableOpBinding());
    const auto worker_binding = enabledBinding(9);
    at::TunableOpExecutionGuard worker_guard(worker_binding);
    for (const auto* state : {&first_state, &second_state, &unbound_state}) {
      {
        at::ThreadLocalStateGuard install(*state);
        const auto expected = state == &first_state ? first
            : state == &second_state ? second : nullptr;
        EXPECT_EQ(at::GetTunableOpBinding(), expected);
      }
      EXPECT_EQ(at::GetTunableOpBinding(), worker_binding);
    }
    EXPECT_THROW(
        {
          at::ThreadLocalStateGuard install(first_state);
          EXPECT_EQ(at::GetTunableOpBinding(), first);
          TORCH_CHECK(false, "worker failed");
        },
        c10::Error);
    EXPECT_EQ(at::GetTunableOpBinding(), worker_binding);
  });
  worker.join();
  EXPECT_FALSE(at::GetTunableOpBinding());
}

TEST_F(TunableOpExecutionTest, PropagatedCallbackRestoresWorkerAfterException) {
  const auto binding = enabledBinding(7);
  at::SetTunableOpBinding(binding);
  auto callback = at::wrapPropagateTLSState([&] {
    EXPECT_EQ(at::GetTunableOpBinding(), binding);
    TORCH_CHECK(false, "callback failed");
  });
  at::SetTunableOpBinding(nullptr);
  std::thread worker([&] {
    at::TunableOpExecutionGuard denied(at::DisabledTunableOpBinding());
    EXPECT_THROW(callback(), c10::Error);
    EXPECT_EQ(at::GetTunableOpBinding(), at::DisabledTunableOpBinding());
  });
  worker.join();
  EXPECT_FALSE(at::GetTunableOpBinding());
}

TEST_F(TunableOpExecutionTest, ConcurrentWorkersKeepIndependentBindings) {
  const auto first = enabledBinding(7);
  const auto second = enabledBinding(8);
  std::promise<void> ready;
  std::promise<void> release;
  auto released = release.get_future();
  std::thread worker([&] {
    EXPECT_FALSE(at::GetTunableOpBinding());
    {
      at::TunableOpExecutionGuard guard(first);
      ready.set_value();
      released.wait();
      EXPECT_EQ(at::GetTunableOpBinding(), first);
    }
    EXPECT_FALSE(at::GetTunableOpBinding());
  });
  ready.get_future().wait();
  {
    at::TunableOpExecutionGuard guard(second);
    EXPECT_EQ(at::GetTunableOpBinding(), second);
    release.set_value();
    worker.join();
    EXPECT_EQ(at::GetTunableOpBinding(), second);
  }
  EXPECT_FALSE(at::GetTunableOpBinding());
}

TEST_F(TunableOpExecutionTest, ManagedOptInIsProcessWide) {
  at::InitializeManagedTunableOp();
  EXPECT_TRUE(at::IsManagedTunableOp());
  std::thread worker([] {
    EXPECT_TRUE(at::IsManagedTunableOp());
    EXPECT_FALSE(at::GetTunableOpBinding());
  });
  worker.join();
  EXPECT_TRUE(at::IsManagedTunableOp());
}

} // namespace
