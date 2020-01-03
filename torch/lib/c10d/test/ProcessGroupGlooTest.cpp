#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

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

    // Use tiny timeout to make this test run fast
    ::c10d::ProcessGroupGloo::Options options;
    options.timeout = std::chrono::milliseconds(50);
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

    // Use tiny timeout to make this test run fast
    ::c10d::ProcessGroupGloo::Options options;
    options.timeout = std::chrono::milliseconds(50);
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
      if (data[j] != expected) {
        throw std::runtime_error("BOOM!");
      }
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
            if (data[n] != expected) {
              throw std::runtime_error("BOOM!");
            }
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

  // Create a sender thread and ensure that sendWork can be aborted.
  std::thread senderThread([&]() {
    auto selfRank = 0;
    auto destRank = 1;

    auto& pg = tests[selfRank].getProcessGroup();

    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };
    // Ensure that we can abort SendWork
    auto work = pg.send(tensors, destRank, tag);
    bool sendCompleted;
    std::thread waitSendWorkThread([&]() { sendCompleted = work->wait(); });
    work->abort();
    waitSendWorkThread.join();
    TORCH_CHECK(!sendCompleted);

    // Ensure that future waits can complete successfully.
    work = pg.send(tensors, destRank, tag);
    sendCompleted = work->wait();
    TORCH_CHECK(sendCompleted);
  });

  // Helper receiver thread to simulate a real send/recv pair.
  std::thread receiverThread([&]() {
    auto selfRank = 1;
    auto srcRank = 0;

    auto& pg = tests[selfRank].getProcessGroup();

    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    // Receive for first send.
    auto work = pg.recv(tensors, srcRank, tag);
    work->wait();

    // Receive for 2nd send.
    work = pg.recv(tensors, srcRank, tag);
    work->wait();
  });

  senderThread.join();
  receiverThread.join();
}

void testRecv(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  constexpr uint64_t tag = 0x1337;

  // Create a receiver thread and ensure that recvWork can be aborted.
  std::thread receiverThread([&]() {
    auto selfRank = 0;
    auto srcRank = 1;

    auto& pg = tests[selfRank].getProcessGroup();

    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    // Ensure that we can abort recvWork
    auto work = pg.recv(tensors, srcRank, tag);
    bool recvCompleted;
    std::thread waitRecvWorkThread([&]() { recvCompleted = work->wait(); });
    work->abort();
    waitRecvWorkThread.join();
    TORCH_CHECK(!recvCompleted);

    // Ensure that future recvWork can complete successfully.
    work = pg.recv(tensors, srcRank, tag);
    recvCompleted = work->wait();
    TORCH_CHECK(recvCompleted);
  });

  // Helper sender thread to simluate a real recv/send pair.
  std::thread senderThread([&]() {
    auto selfRank = 1;
    auto destRank = 0;

    auto& pg = tests[selfRank].getProcessGroup();

    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };
    // Send for first recv.
    auto work = pg.send(tensors, destRank, tag);
    work->wait();

    // Send for 2nd recv.
    work = pg.send(tensors, destRank, tag);
    work->wait();
  });

  receiverThread.join();
  senderThread.join();
}

int main(int argc, char** argv) {
  {
    TemporaryFile file;
    auto work = testSignal(file.path, SIGSTOP);
    try {
      std::rethrow_exception(work->exception());
    } catch (const std::exception& ex) {
      std::cout << "SIGSTOP test got: " << ex.what() << std::endl;
    }
  }

  {
    TemporaryFile file;
    auto work = testSignal(file.path, SIGKILL);
    try {
      std::rethrow_exception(work->exception());
    } catch (const std::exception& ex) {
      std::cout << "SIGKILL test got: " << ex.what() << std::endl;
    }
  }

  {
    TemporaryFile file;
    testAllreduce(file.path, at::DeviceType::CPU);
  }

#ifdef USE_CUDA
  {
    TemporaryFile file;
    testAllreduce(file.path, at::DeviceType::CUDA);
  }
#endif

  {
    TemporaryFile file;
    testBroadcast(file.path, at::DeviceType::CPU);
  }

#ifdef USE_CUDA
  {
    TemporaryFile file;
    testBroadcast(file.path, at::DeviceType::CUDA);
  }
#endif

  {
    TemporaryFile file;
    testBarrier(file.path);
  }

  {
    TemporaryFile file;
    testSend(file.path);
  }

  {
    TemporaryFile file;
    testRecv(file.path);
  }

  std::cout << "Test successful" << std::endl;
  return 0;
}
