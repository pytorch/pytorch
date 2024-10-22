#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>

#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include "CUDATest.hpp"
#include "TestUtils.hpp"

using namespace c10d::test;

using at::cuda::CUDAStream;

template <typename T, typename... Args>
std::vector<T> initialize(const std::string& path, size_t N, Args&&... args) {
  std::vector<T> tests;
  for ([[maybe_unused]] const auto i : c10::irange(N)) {
    tests.push_back(std::move(T(path, std::forward<Args>(args)...)));
  }

  std::vector<std::thread> threads;
  for ([[maybe_unused]] const auto i : c10::irange(N)) {
    threads.push_back(std::thread([i, N, &tests] { tests[i].start(i, N); }));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return tests;
}

class AsyncTest {
 public:
  AsyncTest(std::string path) : path_(std::move(path)) {}

  AsyncTest(AsyncTest&& other) noexcept = default;

  ::c10d::ProcessGroupGloo& getProcessGroup() {
    return *pg_;
  }

  void start(int rank, int size) {
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, size);

    // Use tiny timeout to make this test run fast
    auto options = ::c10d::ProcessGroupGloo::Options::create();
    options->timeout = std::chrono::milliseconds(50);
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

    pg_ =
        std::make_unique<::c10d::ProcessGroupGloo>(store, rank, size, options);
  }

 protected:
  std::string path_;
  std::unique_ptr<::c10d::ProcessGroupGloo> pg_;
};

class AsyncInputIsOutputTest : public AsyncTest {
 public:
  AsyncInputIsOutputTest(const std::string& path, int numTensors)
      : AsyncTest(path),
        numTensors_(numTensors),
        numDevices_(cudaNumDevices()) {
    // Allocate inputs on available devices in a round robin fashion.
    ::at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    inputs_.resize(numTensors_);
    for (const auto i : c10::irange(numTensors_)) {
      inputs_[i] = at::empty(
          {16, 16},
          at::device(
              {at::kCUDA, static_cast<c10::DeviceIndex>(i % numDevices_)}));
    }

    // Allocate a stream per device.
    //
    // The "current stream" is set globally per device in THC, so we
    // can't make two tensors on the same device use different streams
    // and pass this along to the collective (since it uses the THC
    // getters to retrieve the current stream).
    //
    at::cuda::OptionalCUDAGuard deviceGuard;
    streams_.reserve(numDevices_);
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(i));
      streams_.push_back(at::cuda::getStreamFromPool());
    }
  }

  void wait(c10::intrusive_ptr<c10d::Work>& work) {
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    work->wait();
  }

  std::vector<at::Tensor> getCpuTensors(
      const std::vector<at::Tensor>& gpu_tensors) {
    std::vector<at::Tensor> outputs(gpu_tensors.size());

    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (unsigned i = 0; i < gpu_tensors.size(); i++) {
      outputs[i] = gpu_tensors[i].cpu();
    }

    return outputs;
  }

  std::vector<at::Tensor> getTensors() {
    return getCpuTensors(inputs_);
  }

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int numTensors_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int numDevices_;
  std::vector<at::Tensor> inputs_;
  std::vector<CUDAStream> streams_;
};

class AsyncAllreduceTest : public AsyncInputIsOutputTest {
 public:
  AsyncAllreduceTest(const std::string& path, int numTensors)
      : AsyncInputIsOutputTest(path, numTensors) {}

  c10::intrusive_ptr<c10d::Work> run() {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every stream
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(i));
      cudaSleep(streams_[i], 10ull * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (const auto i : c10::irange(numTensors_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(i % numDevices_));
      inputs_[i].fill_(pg_->getRank() * numTensors_ + i);
    }

    return pg_->allreduce(inputs_);
  }
};

class AsyncBroadcastTest : public AsyncInputIsOutputTest {
 public:
  AsyncBroadcastTest(const std::string& path, int numTensors)
      : AsyncInputIsOutputTest(path, numTensors) {}

  c10::intrusive_ptr<c10d::Work> run(size_t rootRank, size_t rootTensor) {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every stream
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(i));
      cudaSleep(streams_[i], 10ull * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (const auto i : c10::irange(numTensors_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(i % numDevices_));
      inputs_[i].fill_(pg_->getRank() * numTensors_ + i);
    }

    ::c10d::BroadcastOptions options;
    options.rootRank = static_cast<int64_t>(rootRank);
    options.rootTensor = static_cast<int64_t>(rootTensor);
    return pg_->broadcast(inputs_, options);
  }
};

void runAsyncAllreduceTest(
    const std::string& path,
    size_t numProcesses = 4,
    size_t numTensors = 2) {
  auto tests = initialize<AsyncAllreduceTest>(path, numProcesses, numTensors);
  std::vector<c10::intrusive_ptr<c10d::Work>> work(numProcesses);
  for (const auto i : c10::irange(numProcesses)) {
    work[i] = tests[i].run();
  }

  // Wait for work to complete
  for (const auto i : c10::irange(numProcesses)) {
    tests[i].wait(work[i]);
  }

  // Check results
  for (const auto i : c10::irange(numProcesses)) {
    const auto size = numProcesses * numTensors;
    const auto expected = (size * (size - 1)) / 2;
    auto tensors = tests[i].getTensors();
    auto results = tests[i].getCpuTensors(work[i]->result());
    EXPECT_EQ(tensors.size(), results.size());

    for (const auto j : c10::irange(tensors.size())) {
      auto& tensor = tensors[j];
      auto data = tensor.data_ptr<float>();

      auto& result_tensor = results[j];
      auto result_data = result_tensor.data_ptr<float>();

      EXPECT_EQ(tensor.numel(), result_tensor.numel());

      for (const auto k : c10::irange(tensor.numel())) {
        EXPECT_EQ(data[k], expected);
        EXPECT_EQ(result_data[k], expected);
      }
    }
  }
}

void runAsyncBroadcastTest(
    const std::string& path,
    size_t numProcesses = 4,
    size_t numTensors = 1) {
  auto tests = initialize<AsyncBroadcastTest>(path, numProcesses, numTensors);

  // Try every permutation of root rank and root tensor
  for (const auto rootRank : c10::irange(numProcesses)) {
    for (const auto rootTensor : c10::irange(numTensors)) {
      std::vector<c10::intrusive_ptr<c10d::Work>> work(numProcesses);
      for (const auto i : c10::irange(numProcesses)) {
        work[i] = tests[i].run(rootRank, rootTensor);
      }

      // Wait for work to complete
      for (const auto i : c10::irange(numProcesses)) {
        tests[i].wait(work[i]);
      }

      // Check results
      const auto expected = (rootRank * numTensors + rootTensor);
      for (const auto i : c10::irange(numProcesses)) {
        auto tensors = tests[i].getTensors();
        for (const auto& tensor : tensors) {
          const auto* const data = tensor.const_data_ptr<float>();
          for (const auto k : c10::irange(tensor.numel())) {
            EXPECT_EQ(data[k], expected);
          }
        }
      }
    }
  }
}

#ifdef USE_CUDA
TEST(ProcessGroupGlooAsyncTest, testAsyncAllreduce) {
  if (!at::cuda::is_available()) {
    LOG(INFO) << "CUDA not available, skipping testAsyncAllreduce";
    return;
  }
  TemporaryFile file;
  runAsyncAllreduceTest(file.path);
}

TEST(ProcessGroupGlooAsyncTest, testAsyncBroadcast) {
  if (!at::cuda::is_available()) {
    LOG(INFO) << "CUDA not available, skipping testAsyncBroadcast";
    return;
  }
  TemporaryFile file;
  runAsyncBroadcastTest(file.path);
}
#endif
