#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include <gtest/gtest.h>
#include <torch/cuda.h>

#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupGloo.hpp>
#include <c10d/test/TestUtils.hpp>

using namespace c10d::test;

class SignalTest {
 public:
  SignalTest(const std::string& path) : path_(path) {}

  ~SignalTest() {
    if (arm_.joinable()) {
      arm_.join();
    }
  }

  // Arms test to send signal to PID when the semaphore unlocks. This
  // happens as soon as the first collective completes successfully.
  void arm(int pid, int signal) {
    arm_ = std::thread([=] {
      sem_.wait();
      kill(pid, signal);
    });
  }

  std::shared_ptr<::c10d::ProcessGroup::Work> run(int rank, int size) {
    auto store = std::make_shared<::c10d::FileStore>(path_, size);

    ::c10d::ProcessGroupGloo::Options options;
    // Set a timeout that is small enough to make this test run fast, but also
    // make sure that we don't get timeouts in the ProcessGroupGloo constructor.
    options.timeout = std::chrono::milliseconds(1000);
    options.devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

    ::c10d::ProcessGroupGloo pg(store, rank, size, options);

    // Initialize tensor list
    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    // Loop until an exception happens
    std::shared_ptr<::c10d::ProcessGroup::Work> work;
    while (true) {
      work = pg.allreduce(tensors);
      try {
        work->wait();
      } catch (const std::exception& e) {
        break;
      }
      sem_.post();
    }

    return work;
  }

 protected:
  std::string path_;
  std::thread arm_;
  Semaphore sem_;
};

std::shared_ptr<::c10d::ProcessGroup::Work> testSignal(
    const std::string& path,
    int signal) {
  Fork fork;
  if (fork.isChild()) {
    SignalTest test(path);
    test.run(1, 2);
    exit(1);
  }

  SignalTest test(path);
  test.arm(fork.pid, signal);
  return test.run(0, 2);
}

class CollectiveTest {
 public:
  static std::vector<CollectiveTest> initialize(
      const std::string& path,
      int num) {
    std::vector<CollectiveTest> tests;
    for (auto i = 0; i < num; i++) {
      tests.push_back(CollectiveTest(path));
    }

    std::vector<std::thread> threads;
    for (auto i = 0; i < num; i++) {
      threads.push_back(
          std::thread([i, &tests] { tests[i].start(i, tests.size()); }));
    }
    for (auto& thread : threads) {
      thread.join();
    }

    return tests;
  }

  CollectiveTest(const std::string& path) : path_(path) {}

  CollectiveTest(CollectiveTest&& other) {
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  ::c10d::ProcessGroupGloo& getProcessGroup() {
    return *pg_;
  }

  void start(int rank, int size) {
    auto store = std::make_shared<::c10d::FileStore>(path_, size);

    // Set a timeout that is small enough to make this test run fast, but also
    // make sure that we don't get timeouts in the ProcessGroupGloo constructor.
    ::c10d::ProcessGroupGloo::Options options;
    options.timeout = std::chrono::milliseconds(1000);
    options.devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

    pg_ = std::unique_ptr<::c10d::ProcessGroupGloo>(
        new ::c10d::ProcessGroupGloo(store, rank, size, options));
  }

 protected:
  std::string path_;
  std::unique_ptr<::c10d::ProcessGroupGloo> pg_;
};

std::vector<std::vector<at::Tensor>> copyTensors(
    const std::vector<std::vector<at::Tensor>>& inputs) {
  std::vector<std::vector<at::Tensor>> outputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    const auto& input = inputs[i];
    std::vector<at::Tensor> output(input.size());
    for (size_t j = 0; j < input.size(); j++) {
      output[j] = input[j].cpu();
    }
    outputs[i] = output;
  }
  return outputs;
}

void testAllreduce(const std::string& path, const at::DeviceType b) {
  const auto size = 4;
  auto tests = CollectiveTest::initialize(path, size);

  // Generate inputs
  std::vector<std::vector<at::Tensor>> inputs(size);
  for (auto i = 0; i < size; i++) {
    auto tensor = at::ones({16, 16}, b) * i;
    inputs[i] = std::vector<at::Tensor>({tensor});
  }

  // Kick off work
  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> work(size);
  for (auto i = 0; i < size; i++) {
    work[i] = tests[i].getProcessGroup().allreduce(inputs[i]);
  }

  // Wait for work to complete
  for (auto i = 0; i < size; i++) {
    work[i]->wait();
  }

  // Verify outputs
  const auto expected = (size * (size - 1)) / 2;
  auto outputs = copyTensors(inputs);
  for (auto i = 0; i < size; i++) {
    auto& tensor = outputs[i][0];
    auto data = tensor.data_ptr<float>();
    for (auto j = 0; j < tensor.numel(); j++) {
      EXPECT_EQ(data[j], expected);
    }
  }
}

void testBroadcast(const std::string& path, const at::DeviceType b) {
  const auto size = 2;
  const auto stride = 2;
  auto tests = CollectiveTest::initialize(path, size);

  std::vector<std::vector<at::Tensor>> inputs(size);

  // Try every permutation of root rank and root tensoro
  for (auto i = 0; i < size; i++) {
    for (auto j = 0; j < stride; j++) {
      // Initialize inputs
      for (auto k = 0; k < size; k++) {
        inputs[k].resize(stride);
        // This won't work if we ever support sparse CUDA
        at::OptionalDeviceGuard deviceGuard;
        for (auto l = 0; l < stride; l++) {
          if (b == at::DeviceType::CUDA) {
            deviceGuard.reset_device(at::Device(at::kCUDA, l));
          }
          inputs[k][l] = at::ones({16, 16}, b) * (k * stride + l);
        }
      }

      ::c10d::BroadcastOptions options;
      options.rootRank = i;
      options.rootTensor = j;

      // Kick off work
      std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> work(size);
      for (auto i = 0; i < size; i++) {
        work[i] = tests[i].getProcessGroup().broadcast(inputs[i], options);
      }

      // Wait for work to complete
      for (auto i = 0; i < size; i++) {
        work[i]->wait();
      }

      // Verify outputs
      const auto expected = (i * stride + j);
      auto outputs = copyTensors(inputs);
      for (auto k = 0; k < size; k++) {
        for (auto l = 0; l < stride; l++) {
          auto& tensor = outputs[k][l];
          auto data = tensor.data_ptr<float>();
          for (auto n = 0; n < tensor.numel(); n++) {
            EXPECT_EQ(data[n], expected);
          }
        }
      }
    }
  }
}

void testBarrier(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  // Kick off work
  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> work(size);
  for (auto i = 0; i < size; i++) {
    work[i] = tests[i].getProcessGroup().barrier();
  }

  // Wait for work to complete
  for (auto i = 0; i < size; i++) {
    work[i]->wait();
  }
}

void testSend(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  constexpr uint64_t tag = 0x1337;
  // test that waiting for work to be sent can be aborted successfully.
  auto selfRank = 0;
  auto dstRank = 1;
  std::vector<at::Tensor> tensors = {
      at::ones({16, 16}),
  };
  auto& pg = tests[selfRank].getProcessGroup();
  auto sendWork = pg.send(tensors, dstRank, tag);
  bool sendCompleted;
  std::thread waitSendThreadAbort([&]() { sendCompleted = sendWork->wait(); });
  sendWork->abort();
  // Block until the sendWork gets successfully aborted
  waitSendThreadAbort.join();
  EXPECT_FALSE(sendCompleted);

  // Now create a separate sender thread to ensure that future waitsends can
  // complete successfully.

  // Helper receiver to simulate a real recv/send pair
  std::thread recvThread([&]() {
    auto selfRank = 1;
    auto srcRank = 0;
    auto& pg = tests[selfRank].getProcessGroup();
    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    auto recvWork = pg.recv(tensors, srcRank, tag);
    recvWork->wait();
  });

  // Sender thread
  std::thread sendThread([&]() { sendCompleted = sendWork->wait(); });
  sendThread.join();
  recvThread.join();
  EXPECT_TRUE(sendCompleted);
}

void testRecv(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);
  constexpr uint64_t tag = 0x1337;
  // test that waiting for work to be received can be aborted successfully.
  auto selfRank = 0;
  auto srcRank = 1;
  std::vector<at::Tensor> tensors = {
      at::ones({16, 16}),
  };
  auto& pg = tests[selfRank].getProcessGroup();
  auto recvWork = pg.recv(tensors, srcRank, tag);
  bool recvCompleted;
  std::thread waitRecvThreadAbort([&]() { recvCompleted = recvWork->wait(); });
  recvWork->abort();
  // Block until the first recv gets successfully aborted
  waitRecvThreadAbort.join();
  EXPECT_FALSE(recvCompleted);

  // Now create a separate receiver thread to ensure that future waits can
  // complete successfully.

  // Helper sender thread to simulate a real recv/send pair.
  std::thread senderThread([&]() {
    auto selfRank = 1;
    auto destRank = 0;

    auto& pg = tests[selfRank].getProcessGroup();

    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };
    auto sendWork = pg.send(tensors, destRank, tag);
    sendWork->wait();
  });
  // Receiver thread.
  std::thread receiverThread([&]() { recvCompleted = recvWork->wait(); });
  senderThread.join();
  receiverThread.join();
  EXPECT_TRUE(recvCompleted);
}

TEST(ProcessGroupGlooTest, testSIGSTOPException) {
  // test SIGSTOP
  // Fork() and TSAN don't play well together, so skip the test if we're testing
  // with TSAN.
  if (isTSANEnabled()) {
    LOG(INFO) << "Skipping test since Fork() + TSAN is broken";
    return;
  }

  TemporaryFile file;
  auto work = testSignal(file.path, SIGSTOP);
  EXPECT_FALSE(work->isSuccess());
  EXPECT_THROW(std::rethrow_exception(work->exception()), std::exception);
}

TEST(ProcessGroupGlooTest, testSIGKILLException) {
  // test SIGKILL
  // Fork() and TSAN don't play well together, so skip the test if we're testing
  // with TSAN.
  if (isTSANEnabled()) {
    LOG(INFO) << "Skipping test since Fork() + TSAN is broken";
    return;
  }

  TemporaryFile file;
  auto work = testSignal(file.path, SIGKILL);
  EXPECT_FALSE(work->isSuccess());
  EXPECT_THROW(std::rethrow_exception(work->exception()), std::exception);
}

TEST(ProcessGroupGlooTest, testAllReduceCPU) {
  {
    TemporaryFile file;
    testAllreduce(file.path, at::DeviceType::CPU);
  }
}

TEST(ProcessGroupGlooTest, testBroadcastCPU) {
  {
    TemporaryFile file;
    testBroadcast(file.path, at::DeviceType::CPU);
  }
}

TEST(ProcessGroupGlooTest, testBarrier) {
  {
    TemporaryFile file;
    testBarrier(file.path);
  }
}

TEST(ProcessGroupGlooTest, testSend) {
  {
    TemporaryFile file;
    testSend(file.path);
  }
}

TEST(ProcessGroupGlooTest, testRecv) {
  {
    TemporaryFile file;
    testRecv(file.path);
  }
}

#ifdef USE_CUDA
// CUDA-only tests
TEST(ProcessGroupGlooTest, testAllReduceCUDA) {
  {
    if (torch::cuda::is_available()) {
      TemporaryFile file;
      testAllreduce(file.path, at::DeviceType::CUDA);
    }
  }
}

TEST(ProcessGroupGlooTest, testBroadcastCUDA) {
  {
    if (torch::cuda::is_available()) {
      TemporaryFile file;
      testBroadcast(file.path, at::DeviceType::CUDA);
    }
  }
}

#endif
