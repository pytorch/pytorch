#include <gtest/gtest.h>

#include "e2e_test_base.h"

#include <c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/rpc/process_group_agent.h>
#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/torch.h>

namespace torch {
namespace distributed {
namespace rpc {


class TestE2EProcessGroup : public TestE2EBase {
 protected:
  void buildRpcAgent() override {
    auto options = c10d::ProcessGroupGloo::Options::create();
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname(serverAddress));
    std::chrono::milliseconds rpcTimeout(30000);
    options->timeout = rpcTimeout;

    // Initialize server rpc agent.
    auto pg = c10::make_intrusive<c10d::ProcessGroupGloo>(
        store, 0, numWorkers, options);

    rpcAgent = std::make_shared<ProcessGroupAgent>(
        store,
        "worker",
        pg,
        std::max(16U, std::thread::hardware_concurrency()),
        rpcTimeout,
        std::make_unique<RequestCallbackNoPython>());
  }
};

// End to end training loop test in C++ so that we can run LSAN on this test to
// catch memory leaks. Enabling LSAN with python multiprocessing has been
// challenging and we don't have a good solution yet.
TEST_F(TestE2EProcessGroup, TestTrainingLoop) {
  runTrainingLoop();
}

} // namespace rpc
} // namespace distributed
} // namespace torch
