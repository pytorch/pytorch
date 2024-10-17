#include <c10/util/DeadlockDetection.h>
#include <c10/util/env.h>

#include <gtest/gtest.h>

using namespace ::testing;
using namespace c10::impl;

struct DummyPythonGILHooks : public PythonGILHooks {
  bool check_python_gil() const override {
    return true;
  }
};

TEST(DeadlockDetection, basic) {
  ASSERT_FALSE(check_python_gil());
  DummyPythonGILHooks hooks;
  SetPythonGILHooks(&hooks);
  ASSERT_TRUE(check_python_gil());
  SetPythonGILHooks(nullptr);
}

#ifndef _WIN32
TEST(DeadlockDetection, disable) {
  c10::utils::set_env("TORCH_DISABLE_DEADLOCK_DETECTION", "1");
  DummyPythonGILHooks hooks;
  SetPythonGILHooks(&hooks);
  SetPythonGILHooks(&hooks);
}
#endif
