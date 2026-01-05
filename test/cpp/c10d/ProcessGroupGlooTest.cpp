#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#endif

#include <sys/types.h>

#include <memory>
#include <thread>

#include <gtest/gtest.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/cuda.h>

#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include "TestUtils.hpp"

using namespace c10d::test;
using namespace torch::autograd::profiler;

constexpr auto kSendDelay = std::chrono::milliseconds(100);
constexpr auto kWaitTimeout = std::chrono::milliseconds(1);

#ifndef _WIN32
class SignalTest {
 public:
  SignalTest(std::string path) : path_(std::move(path)) {}

  ~SignalTest() {
    if (arm_.joinable()) {
      arm_.join();
    }
  }

  // Arms test to send signal to PID when the semaphore unlocks. This
  // happens as soon as the first collective completes successfully.
  void arm(int pid, int signal) {
    arm_ = std::thread([this, pid, signal] {
      sem_.wait();
      kill(pid, signal);
    });
  }

  c10::intrusive_ptr<::c10d::Work> run(int rank, int size) {
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, size);

    auto options = ::c10d::ProcessGroupGloo::Options::create();
    // Set a timeout that is small enough to make this test run fast, but also
    // make sure that we don't get timeouts in the ProcessGroupGloo constructor.
    options->timeout = std::chrono::milliseconds(1000);
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

    ::c10d::ProcessGroupGloo pg(store, rank, size, options);

    // Initialize tensor list
    std::vector<at::Tensor> tensors = {
        at::ones({16, 16}),
    };

    // Loop until an exception happens
    c10::intrusive_ptr<::c10d::Work> work;
    while (true) {
      work = pg.allreduce(tensors);
      try {
        work->wait();
      } catch (const std::exception&) {
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

c10::intrusive_ptr<::c10d::Work> testSignal(
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
#endif

class ProcessGroupGlooDelayed : public ::c10d::ProcessGroupGloo {
 public:
  ProcessGroupGlooDelayed(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options)
      : ProcessGroupGloo(store, rank, size, std::move(options)) {}

  c10::intrusive_ptr<::c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    std::this_thread::sleep_for(kSendDelay);
    return ::c10d::ProcessGroupGloo::send(tensors, dstRank, tag);
  }
};

class CollectiveTest {
 public:
  static std::vector<CollectiveTest> initialize(
      const std::string& path,
      int num,
      bool delayed = false) {
    std::vector<CollectiveTest> tests;
    for ([[maybe_unused]] const auto i : c10::irange(num)) {
      tests.emplace_back(path);
    }

    std::vector<std::thread> threads;
    for (const auto i : c10::irange(num)) {
      threads.emplace_back(
          [i, &tests, delayed] { tests[i].start(i, tests.size(), delayed); });
    }
    for (auto& thread : threads) {
      thread.join();
    }

    return tests;
  }

  CollectiveTest(std::string path) : path_(std::move(path)) {}

  CollectiveTest(CollectiveTest&& other) noexcept = default;

  ::c10d::ProcessGroupGloo& getProcessGroup() {
    return *pg_;
  }

  void start(int rank, size_t size, bool delayed) {
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, size);

    // Set a timeout that is small enough to make this test run fast, but also
    // make sure that we don't get timeouts in the ProcessGroupGloo constructor.
    auto options = ::c10d::ProcessGroupGloo::Options::create();
    options->timeout = std::chrono::milliseconds(1000);
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

    if (!delayed) {
      pg_ = std::make_unique<::c10d::ProcessGroupGloo>(
          store, rank, size, options);
    } else {
      pg_ =
          std::make_unique<ProcessGroupGlooDelayed>(store, rank, size, options);
    }
  }

 protected:
  std::string path_;
  std::unique_ptr<::c10d::ProcessGroupGloo> pg_;
};

std::vector<std::vector<at::Tensor>> copyTensors(
    const std::vector<std::vector<at::Tensor>>& inputs) {
  std::vector<std::vector<at::Tensor>> outputs(inputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    const auto& input = inputs[i];
    std::vector<at::Tensor> output(input.size());
    for (const auto j : c10::irange(input.size())) {
      output[j] = input[j].cpu();
    }
    outputs[i] = output;
  }
  return outputs;
}

std::vector<std::vector<at::Tensor>> waitWork(
    const std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  std::vector<std::vector<at::Tensor>> outputTensors;
  for (auto& work : works) {
    try {
      work->wait();
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Exception received: " << ex.what() << '\n';
    }
    outputTensors.emplace_back(work->result());
  }
  return copyTensors(outputTensors);
}

std::vector<std::vector<at::Tensor>> waitFuture(
    const std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  std::vector<std::vector<at::Tensor>> outputTensors;
  for (auto& work : works) {
    auto fut = work->getFuture();
    try {
      fut->wait();
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Exception received: " << ex.what() << '\n';
    }
    auto result = fut->value();
    if (result.isNone()) {
      outputTensors.emplace_back();
    } else if (result.isTensorList()) {
      outputTensors.emplace_back(result.toTensorVector());
    } else {
      TORCH_CHECK(false, "future result should be tensor list or none");
    }
  }
  return copyTensors(outputTensors);
}

void checkProfiledEvents(
    const thread_event_lists& event_lists,
    const char* expected_profile_str,
    int expected_count,
    std::vector<std::vector<int64_t>> expected_shapes,
    bool verify_shapes = true) {
  if (verify_shapes) {
    EXPECT_EQ(expected_count, expected_shapes.size());
  }
  std::vector<bool> matched_shapes(expected_count);
  for (const auto& li : event_lists) {
    for (const auto& evt : li) {
      auto match = !strcmp(evt.name(), expected_profile_str);
      if (verify_shapes && match) {
        auto shapesVec = evt.shapes();
        for (const auto i : c10::irange(expected_count)) {
          // Assumptions: no two expected shapes are the same
          if (shapesVec[0] == expected_shapes[i]) {
            matched_shapes[i] = true;
          }
        }
      }
    }
  }
  if (verify_shapes) {
    for (bool match : matched_shapes) {
      EXPECT_TRUE(match);
    }
  }
}

void testAllreduce(
    const std::string& path,
    const at::DeviceType b,
    const at::ScalarType dtype = at::kFloat) {
  const auto size = 4;
  auto tests = CollectiveTest::initialize(path, size);

  // Generate inputs
  std::vector<std::vector<at::Tensor>> inputs(size);
  std::vector<std::vector<int64_t>> allShapes;
  std::vector<int64_t> shapes = {16, 16};
  for (const auto i : c10::irange(size)) {
    auto tensor = at::ones(shapes, at::dtype(dtype).device(b)) * i;
    std::vector<int64_t> shapesVec = shapes;
    allShapes.emplace_back(std::move(shapesVec));
    inputs[i] = std::vector<at::Tensor>({tensor});
  }

  // Kick off work
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);
  const char* GLOO_ALLREDUCE_STR = "gloo:all_reduce";
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  for (const auto i : c10::irange(size)) {
    work[i] = tests[i].getProcessGroup().allreduce(inputs[i]);
  }
  // Wait for work to complete
  auto outputs = waitFuture(work);

  auto event_lists = disableProfilerLegacy();
  checkProfiledEvents(event_lists, GLOO_ALLREDUCE_STR, size, allShapes);

  // Verify outputs
  const auto expected = (size * (size - 1)) / 2;
  for (const auto i : c10::irange(size)) {
    auto tensor = outputs[i][0].to(at::kFloat);
    auto data = tensor.data_ptr<float>();
    for (const auto j : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[j], expected);
    }
  }
}

// UsingWorkAPI tests are to make sure we still properly support work API.
// This should go away as we deprecate it.
void testAllreduceUsingWorkAPI(
    const std::string& path,
    const at::DeviceType b,
    const at::ScalarType dtype = at::kFloat) {
  const auto size = 4;
  auto tests = CollectiveTest::initialize(path, size);

  // Generate inputs
  std::vector<std::vector<at::Tensor>> inputs(size);
  std::vector<std::vector<int64_t>> allShapes;
  std::vector<int64_t> shapes = {16, 16};
  for (const auto i : c10::irange(size)) {
    auto tensor = at::ones(shapes, at::dtype(dtype).device(b)) * i;
    std::vector<int64_t> shapesVec = shapes;
    allShapes.emplace_back(std::move(shapesVec));
    inputs[i] = std::vector<at::Tensor>({tensor});
  }

  // Kick off work
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);
  const char* GLOO_ALLREDUCE_STR = "gloo:all_reduce";
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  for (const auto i : c10::irange(size)) {
    work[i] = tests[i].getProcessGroup().allreduce(inputs[i]);
  }
  // Wait for work to complete
  auto outputs = waitWork(work);

  auto event_lists = disableProfilerLegacy();
  checkProfiledEvents(event_lists, GLOO_ALLREDUCE_STR, size, allShapes);

  // Verify outputs
  const auto expected = (size * (size - 1)) / 2;
  for (const auto i : c10::irange(size)) {
    auto tensor = outputs[i][0].to(at::kFloat);
    auto data = tensor.data_ptr<float>();
    for (const auto j : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[j], expected);
    }
  }
}

void testBroadcast(
    const std::string& path,
    const at::DeviceType b,
    const at::ScalarType dtype = at::kFloat) {
  const auto size = 2;
  const auto stride = 2;
  auto tests = CollectiveTest::initialize(path, size);

  std::vector<std::vector<at::Tensor>> inputs(size);
  std::vector<int64_t> shapes = {16, 16};
  // Try every permutation of root rank and root tensor
  for (const auto i : c10::irange(size)) {
    for (const auto j : c10::irange(stride)) {
      std::vector<std::vector<int64_t>> allShapes;
      // Initialize inputs
      for (const auto k : c10::irange(size)) {
        std::vector<int64_t> shapesVec = shapes;
        allShapes.emplace_back(std::move(shapesVec));
        inputs[k].resize(stride);
        // This won't work if we ever support sparse CUDA
        at::OptionalDeviceGuard deviceGuard;
        for (const auto l : c10::irange(stride)) {
          if (b == at::DeviceType::CUDA) {
            deviceGuard.reset_device(
                at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(l)));
          }
          inputs[k][l] =
              at::ones(shapes, at::dtype(dtype).device(b)) * (k * stride + l);
        }
      }

      ::c10d::BroadcastOptions options;
      options.rootRank = i;
      options.rootTensor = j;

      // Kick off work
      const char* GLOO_BROADCAST_STR = "gloo:broadcast";
      enableProfilerLegacy(ProfilerConfig(
          ProfilerState::CPU, /* report_input_shapes */ true, false));
      std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);

      for (const auto i : c10::irange(size)) {
        work[i] = tests[i].getProcessGroup().broadcast(inputs[i], options);
      }

      // Wait for work to complete
      auto outputs = waitFuture(work);

      auto event_lists = disableProfilerLegacy();
      checkProfiledEvents(event_lists, GLOO_BROADCAST_STR, size, allShapes);

      // Verify outputs
      const auto expected = (i * stride + j);
      for (const auto k : c10::irange(size)) {
        for (const auto l : c10::irange(stride)) {
          auto tensor = outputs[k][l].to(at::kFloat);
          auto data = tensor.data_ptr<float>();
          for (const auto n : c10::irange(tensor.numel())) {
            EXPECT_EQ(data[n], expected);
          }
        }
      }
    }
  }
}

void testAlltoall(const std::string& path, const at::DeviceType b) {
  const auto size = 4;
  auto tests = CollectiveTest::initialize(path, size);

  // Generate inputs
  std::vector<at::Tensor> inputs(size);
  std::vector<std::vector<int32_t>> blobs = {
      {0, 1, 2, 3, 4, 5},
      {10, 11, 12, 13, 14, 15, 16, 17, 18},
      {20, 21, 22, 23, 24},
      {30, 31, 32, 33, 34, 35, 36},
  };
  for (const auto rank : c10::irange(size)) {
    std::vector<int32_t>& blob = blobs[rank];
    inputs[rank] =
        at::from_blob(blob.data(), static_cast<int64_t>(blob.size())).to(b);
  }

  // Allocate outputs
  std::vector<at::Tensor> outputs(size);
  std::vector<int> outputLengths = {9, 7, 6, 5};
  for (const auto rank : c10::irange(size)) {
    outputs[rank] =
        at::empty(outputLengths[rank], c10::TensorOptions(at::kInt).device(b));
  }

  // Generate splits
  std::vector<std::vector<int64_t>> inputSplits = {
      {2, 2, 1, 1},
      {3, 2, 2, 2},
      {2, 1, 1, 1},
      {2, 2, 2, 1},
  };
  std::vector<std::vector<int64_t>> outputSplits = {
      {2, 3, 2, 2},
      {2, 2, 1, 2},
      {1, 2, 1, 2},
      {1, 2, 1, 1},
  };

  // Kick off work
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);
  const char* GLOO_A2A_STR = "gloo:all_to_all";
  std::vector<std::vector<int64_t>> allShapes;
  for (const auto& vec : inputSplits) {
    // Due to concatenation of tensors, shape will actually be the sum
    int64_t sum = 0;
    for (const auto& s : vec) {
      sum += s;
    }
    allShapes.push_back({sum});
  }
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  for (const auto rank : c10::irange(size)) {
    work[rank] = tests[rank].getProcessGroup().alltoall_base(
        outputs[rank], inputs[rank], outputSplits[rank], inputSplits[rank]);
  }

  // Wait for work to complete
  for (const auto i : c10::irange(size)) {
    work[i]->wait();
  }

  auto event_lists = disableProfilerLegacy();
  checkProfiledEvents(event_lists, GLOO_A2A_STR, size, allShapes);
  // Verify outputs
  std::vector<std::vector<int32_t>> expected = {
      {0, 1, 10, 11, 12, 20, 21, 30, 31},
      {2, 3, 13, 14, 22, 32, 33},
      {4, 15, 16, 23, 34, 35},
      {5, 17, 18, 24, 36},
  };
  for (const auto rank : c10::irange(size)) {
    at::Tensor tensor = outputs[rank].cpu();
    EXPECT_EQ(tensor.numel(), expected[rank].size());
    auto data = tensor.data_ptr<int32_t>();
    for (const auto j : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[j], expected[rank][j]);
    }
  }
}

void testBarrier(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  // Kick off work
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  std::vector<c10::intrusive_ptr<::c10d::Work>> work(size);
  for (const auto i : c10::irange(size)) {
    work[i] = tests[i].getProcessGroup().barrier();
  }

  // Wait for work to complete
  waitFuture(work);

  auto event_lists = disableProfilerLegacy();
  const char* GLOO_STR = "gloo:barrier";
  std::vector<std::vector<int64_t>> allShapes;
  // Barrier does not use tensors, so skip shape checking.
  checkProfiledEvents(
      event_lists,
      GLOO_STR,
      size,
      allShapes,
      /* verify_shapes */ false);
}

void testMonitoredBarrier(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);
  // Non-failure case: all ranks pass the blocking monitored barrier.
  auto runMonitoredBarrier = [&](int i) {
    tests[i].getProcessGroup().monitoredBarrier();
  };
  std::vector<std::thread> threads;
  threads.reserve(size);
  for (const auto r : c10::irange(size)) {
    threads.emplace_back([=]() { runMonitoredBarrier(r); });
  }
  for (auto& t : threads) {
    t.join();
  }
  // Failure case: Only rank 0 calls into monitored barrier, should result in
  // error
  auto runMonitoredBarrierWithException = [&](int i) {
    if (i != 0) {
      return;
    }

    try {
      tests[i].getProcessGroup().monitoredBarrier();
      FAIL() << "Exception should have been thrown.";
    } catch (const std::exception& e) {
      auto pos = std::string(e.what()).find("Rank 1");
      EXPECT_TRUE(pos != std::string::npos);
    }
  };
  threads.clear();
  for (const auto r : c10::irange(size)) {
    threads.emplace_back([=]() { runMonitoredBarrierWithException(r); });
  }
  for (auto& t : threads) {
    t.join();
  }
}

void testSequenceNumInit(const std::string& path) {
  const auto size = 4;
  auto tests = CollectiveTest::initialize(path, size);
  for (const auto i : c10::irange(size)) {
    tests[i].getProcessGroup().setSequenceNumberForGroup();
  }

  std::unordered_set<uint64_t> nums;
  for (const auto i : c10::irange(size)) {
    auto seqNum = tests[i].getProcessGroup().getSequenceNumberForGroup();
    nums.insert(seqNum);
  }
  EXPECT_EQ(nums.size(), 1);
}

void testWaitDelay(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size, /* delay */ true);

  constexpr uint64_t tag = 0x1337;
  // test that waiting for work to be sent can be aborted successfully.
  auto selfRank = 0;
  auto dstRank = 1;
  std::vector<at::Tensor> tensors = {
      at::ones({16, 16}),
  };
  auto& pg = tests[selfRank].getProcessGroup();
  auto sendWork = pg.send(tensors, dstRank, tag);
  EXPECT_THROW(sendWork->wait(kWaitTimeout), std::exception);
}

void testSend(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);

  constexpr uint64_t tag = 0x1337;
  // test that waiting for work to be sent can be aborted successfully.
  auto selfRank = 0;
  auto dstRank = 1;
  std::vector<int64_t> shapes{16, 16};
  std::vector<std::vector<int64_t>> allShapes;
  allShapes.push_back(shapes);
  std::vector<at::Tensor> tensors = {
      at::ones(shapes),
  };
  auto& pg = tests[selfRank].getProcessGroup();
  const char* GLOO_SEND_STR = "gloo:send";
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  auto sendWork = pg.send(tensors, dstRank, tag);
  bool sendCompleted = false;
  std::thread waitSendThreadAbort([&]() { sendCompleted = sendWork->wait(); });
  sendWork->abort();
  // Block until the sendWork gets successfully aborted
  waitSendThreadAbort.join();
  EXPECT_FALSE(sendCompleted);
  auto event_lists = disableProfilerLegacy();
  checkProfiledEvents(event_lists, GLOO_SEND_STR, 1, allShapes);

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
  std::vector<int64_t> shapes = {16, 16};
  std::vector<std::vector<int64_t>> allShapes;
  allShapes.push_back(shapes);
  std::vector<at::Tensor> tensors = {
      at::ones(shapes),
  };
  const char* GLOO_RECV_STR = "gloo:recv";
  auto& pg = tests[selfRank].getProcessGroup();
  enableProfilerLegacy(ProfilerConfig(
      ProfilerState::CPU, /* report_input_shapes */ true, false));
  auto recvWork = pg.recv(tensors, srcRank, tag);
  bool recvCompleted = false;
  std::thread waitRecvThreadAbort([&]() { recvCompleted = recvWork->wait(); });
  recvWork->abort();
  // Block until the first recv gets successfully aborted
  waitRecvThreadAbort.join();
  EXPECT_FALSE(recvCompleted);
  auto event_lists = disableProfilerLegacy();
  checkProfiledEvents(event_lists, GLOO_RECV_STR, 1, allShapes);

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

void testStoreSetGet(const std::string& path) {
  const auto size = 2;
  auto tests = CollectiveTest::initialize(path, size);
  // test that get() gets the same value as the one that was set()
  std::vector<uint8_t> testVector = {1, 1, 1, 1};
  // Cast to ProcessGroupGloo::GlooStore to test specific GlooStore APIs.
  auto rank_0_glooStore = static_cast<c10d::ProcessGroupGloo::GlooStore*>(
      tests[0].getProcessGroup()._getStore().get());
  auto rank_1_glooStore = static_cast<c10d::ProcessGroupGloo::GlooStore*>(
      tests[1].getProcessGroup()._getStore().get());

  rank_0_glooStore->setUint("testKey", testVector);
  auto value = rank_1_glooStore->getUint("testKey");
  EXPECT_TRUE(value == testVector);
}

#ifndef _WIN32
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
#endif

TEST(ProcessGroupGlooTest, testAllReduceCPU) {
  {
    TemporaryFile file;
    testAllreduce(file.path, at::DeviceType::CPU);
    testAllreduceUsingWorkAPI(file.path, at::DeviceType::CPU);
  }
}

TEST(ProcessGroupGlooTest, testAllReduceBfloatCPU) {
  {
    TemporaryFile file;
    testAllreduce(file.path, at::DeviceType::CPU, at::kBFloat16);
    testAllreduceUsingWorkAPI(file.path, at::DeviceType::CPU);
  }
}

TEST(ProcessGroupGlooTest, testBroadcastCPU) {
  {
    TemporaryFile file;
    testBroadcast(file.path, at::DeviceType::CPU);
  }
}

TEST(ProcessGroupGlooTest, testBroadcastBfloatCPU) {
  {
    TemporaryFile file;
    testBroadcast(file.path, at::DeviceType::CPU, at::kBFloat16);
  }
}

TEST(ProcessGroupGlooTest, testAllToAllCPU) {
  {
    TemporaryFile file;
    testAlltoall(file.path, at::DeviceType::CPU);
  }
}

TEST(ProcessGroupGlooTest, testBarrier) {
  {
    TemporaryFile file;
    testBarrier(file.path);
  }
}

TEST(ProcessGroupGlooTest, testMonitoredBarrier) {
  TemporaryFile file;
  testMonitoredBarrier(file.path);
}

TEST(ProcessGroupGlooTest, testSequenceNumInit) {
  TemporaryFile file;
  testSequenceNumInit(file.path);
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

TEST(ProcessGroupGlooTest, testStoreSetGet) {
  TemporaryFile file;
  testStoreSetGet(file.path);
}

TEST(ProcessGroupGlooTest, testWaitDelay) {
  {
    TemporaryFile file;
    testWaitDelay(file.path);
  }
}

#ifdef USE_CUDA
// CUDA-only tests
TEST(ProcessGroupGlooTest, testAllReduceCUDA) {
  if (!torch::cuda::is_available()) {
    LOG(INFO) << "Skipping test - requires CUDA";
    return;
  }
  {
    TemporaryFile file;
    testAllreduce(file.path, at::DeviceType::CUDA);
    testAllreduceUsingWorkAPI(file.path, at::DeviceType::CUDA);
  }
}

TEST(ProcessGroupGlooTest, testBroadcastCUDA) {
  if (torch::cuda::device_count() <= 1) {
    LOG(INFO) << "Skipping test - requires multiple CUDA devices";
    return;
  }
  {
    TemporaryFile file;
    testBroadcast(file.path, at::DeviceType::CUDA);
  }
}

TEST(ProcessGroupGlooTest, testAlltoallCUDA) {
  if (!torch::cuda::is_available()) {
    LOG(INFO) << "Skipping test - requires CUDA";
    return;
  }
  {
    TemporaryFile file;
    testAlltoall(file.path, at::DeviceType::CUDA);
  }
}

TEST(ProcessGroupGlooTest, testBackendName) {
  {
    TemporaryFile file;
    const auto size = 2;
    auto tests = CollectiveTest::initialize(file.path, size);

    for (const auto i : c10::irange(size)) {
      EXPECT_EQ(
          tests[i].getProcessGroup().getBackendName(),
          std::string(c10d::GLOO_BACKEND_NAME));
    }
  }
}

#endif
