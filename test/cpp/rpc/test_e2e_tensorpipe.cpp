#include <gtest/gtest.h>

#include "e2e_test_base.h"

#include <c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/rpc/request_callback_no_python.h>
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>
#include <torch/torch.h>

namespace torch {
namespace distributed {
namespace rpc {

#ifdef USE_TENSORPIPE

class TestE2ETensorPipe : public TestE2EBase {
 protected:
  void buildRpcAgent() override {
    c10d::ProcessGroupGloo::Options options;
    options.devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname(serverAddress));
    float rpcTimeout = 30;

    // Initialize server rpc agent.
    auto pg = c10::make_intrusive<c10d::ProcessGroupGloo>(
        store, 0, numWorkers, options);

    TensorPipeRpcBackendOptions opts(
        /*numWorkerThreads=*/std::max(16U, std::thread::hardware_concurrency()),
        /*transports=*/nullopt,
        /*channels=*/nullopt,
        /*rpc_timeout=*/rpcTimeout,
        /*init_method=*/"unused");

    rpcAgent = std::make_shared<TensorPipeAgent>(
        store,
        "worker",
        0,
        numWorkers,
        pg,
        opts,
        std::make_unique<RequestCallbackNoPython>());
  }
};

// End to end training loop test in C++ so that we can run LSAN on this test to
// catch memory leaks. Enabling LSAN with python multiprocessing has been
// challenging and we don't have a good solution yet.
TEST_F(TestE2ETensorPipe, TestTrainingLoop) {
  runTrainingLoop();
  // Ensure the tensorpipe internal state is cleared up.
  auto tensorpipeAgent = std::static_pointer_cast<TensorPipeAgent>(rpcAgent);
  // Wait a while for async RPCs to propagate through (ex: dist autograd
  // cleanup)
  while (tensorpipeAgent->numPendingResponses() != 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_EQ(0, tensorpipeAgent->numPendingResponses());
  ASSERT_EQ(0, tensorpipeAgent->timeoutMapSize());
}

#endif

} // namespace rpc
} // namespace distributed
} // namespace torch
