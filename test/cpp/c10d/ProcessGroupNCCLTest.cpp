#include <chrono>
#include <iostream>

#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include "CUDATest.hpp"
#include "TestUtils.hpp"
#include "c10d/Types.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>

#include <gtest/gtest.h>
#include <torch/csrc/autograd/profiler.h>

using namespace c10d::test;

using at::cuda::CUDAStream;

class NCCLTestBase {
 public:
  NCCLTestBase(
      std::string path,
      const std::chrono::milliseconds pgTimeout =
          c10d::kProcessGroupNCCLDefaultTimeout)
      : path_(std::move(path)), pgTimeout_(pgTimeout) {}

  NCCLTestBase(NCCLTestBase&& other) noexcept = default;

  std::shared_ptr<::c10d::ProcessGroupNCCL> getProcessGroup() {
    return pg_;
  }

  ::c10::intrusive_ptr<::c10d::Store>& getProcessGroupStore() {
    return store_;
  }

  void initialize(
      int rank,
      size_t size,
      std::optional<::std::shared_ptr<::c10d::ProcessGroupNCCL>> split_from =
          std::nullopt) {
    store_ = c10::make_intrusive<::c10d::FileStore>(path_, size);

    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts =
        c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
    opts->timeout = pgTimeout_;
#ifdef NCCL_HAS_COMM_SPLIT
    if (split_from) {
      opts->split_from = *split_from;
      opts->split_color = ++color_;
    }
#endif
    pg_ = std::make_unique<::c10d::ProcessGroupNCCL>(
        store_, rank, size, std::move(opts));
  }

 protected:
  std::string path_;
  std::shared_ptr<::c10d::ProcessGroupNCCL> pg_;
  std::chrono::milliseconds pgTimeout_;
  ::c10::intrusive_ptr<::c10d::Store> store_;
  int color_{1};
};

class NCCLTest : public NCCLTestBase {
 public:
  NCCLTest(
      const std::string& path,
      int rank,
      int worldSize,
      std::chrono::milliseconds pgTimeout =
          c10d::kProcessGroupNCCLDefaultTimeout,
      int inputDim = 3)
      : NCCLTestBase(path, pgTimeout), rank_(rank), worldSize_(worldSize) {
    // Each device has a single tensor to perf the NCCL op
    ::at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    tensors_.resize(numDevices_);
    inputs_.resize(numDevices_);
    outputs_.resize(numDevices_);
    at::cuda::OptionalCUDAGuard deviceGuard;
    assert(numDevices_ == 1);
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(rank_));
      tensors_[i] = at::empty({inputDim, inputDim}, at::kCUDA);
      inputs_[i].resize(static_cast<size_t>(worldSize_) * numDevices_);
      outputs_[i].resize(static_cast<size_t>(worldSize_) * numDevices_);
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        inputs_[i][j] = at::empty({inputDim, inputDim}, at::kCUDA);
        outputs_[i][j] = at::empty({inputDim, inputDim}, at::kCUDA);
      }
    }

    // Allocate a stream per device.
    //
    // The "current stream" is set globally per device in THC, so we
    // can't make two tensors on the same device use different streams
    // and pass this along to the collective (since it uses the THC
    // getters to retrieve the current stream).
    //
    // 1 device only, hence 1 stream only
    deviceGuard.set_index(static_cast<c10::DeviceIndex>(rank_));
    streams_.push_back(at::cuda::getStreamFromPool());
  }

  void wait(
      c10::intrusive_ptr<c10d::Work>& work,
      std::chrono::milliseconds timeout = kNoTimeout) {
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    work->wait(timeout);
  }

  std::vector<at::Tensor> getTensors() {
    std::vector<at::Tensor> outputs(numDevices_);

    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (const auto i : c10::irange(numDevices_)) {
      C10_CUDA_CHECK(cudaStreamSynchronize(streams_[i].stream()));
      outputs[i] = tensors_[i].cpu();
    }

    return outputs;
  }

  std::vector<std::vector<at::Tensor>> getInputTensors() {
    return getTensorLists(inputs_);
  }
  std::vector<std::vector<at::Tensor>> getOutputTensors() {
    return getTensorLists(outputs_);
  }

  int numDevices() const {
    return numDevices_;
  }

 private:
  std::vector<std::vector<at::Tensor>> getTensorLists(
      std::vector<std::vector<at::Tensor>>& tensor_lists) {
    std::vector<std::vector<at::Tensor>> outputs(numDevices_);
    for (auto& output : outputs) {
      output = std::vector<at::Tensor>(
          static_cast<size_t>(worldSize_ * numDevices_));
    }

    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (const auto i : c10::irange(numDevices_)) {
      C10_CUDA_CHECK(cudaStreamSynchronize(streams_[i].stream()));
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        outputs[i][j] = tensor_lists[i][j].cpu();
      }
    }
    return outputs;
  }

 protected:
  // Launches sleep on every CUDA device
  void launchDeviceSleep() {
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(rank_));
      cudaSleep(streams_[i], 2000ull * 1000 * 1000);
    }
  }

  // Launches value initialization for every tensor
  void valueInitialization() {
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(rank_));
      tensors_[i].fill_(pg_->getRank() * numDevices_ + i);
    }
  }

  at::Tensor to_sparse_row_indices_format(at::Tensor& tensor) {
    // Get the indices of all non-zero elements in the dense tensor
    // Get the unique row indices of the non-zero elements
    auto row_indices = std::get<0>(
        at::_unique(tensor.nonzero().select(/*dim=*/1, /*index=*/0)));
    at::Tensor sparse_values = tensor.index_select(
        /*dim=*/0, row_indices); // get the values at the non-zero indices
    return at::sparse_coo_tensor(
               row_indices.unsqueeze(0), sparse_values, tensor.sizes())
        .to(tensor.device());
  }

  // Launches value initialization for every sparse tensor
  void valueInitializationForSparse() {
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (const auto i : c10::irange(numDevices_)) {
      deviceGuard.set_index(static_cast<c10::DeviceIndex>(rank_));
      tensors_[i].fill_(pg_->getRank() * numDevices_ + i + 1);
      // Convert the dense tensor to a sparse tensor in COO row format
      tensors_[i] = to_sparse_row_indices_format(tensors_[i]);
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int numDevices_{1}; // one device per rank (thread)
  int rank_;
  int worldSize_;
  std::vector<at::Tensor> tensors_;
  std::vector<std::vector<at::Tensor>> inputs_;
  std::vector<std::vector<at::Tensor>> outputs_;
  std::vector<CUDAStream> streams_;
};

class AllreduceNCCLTest : public NCCLTest {
 public:
  AllreduceNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  c10::intrusive_ptr<c10d::Work> run() {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    using namespace torch::autograd::profiler;
    // Make sure enabling profile does not make any issue. Note, in single
    // process multi-device mode we do not expect any events be populated for
    // collective operations, since profiling for that mode is not supported.
    enableProfilerLegacy(ProfilerConfig(ProfilerState::CPU));
    auto results = pg_->allreduce(tensors_);
    disableProfilerLegacy();
    return results;
  }
};

class SparseAllreduceNCCLTest : public NCCLTest {
 public:
  SparseAllreduceNCCLTest(
      const std::string& path,
      int rank,
      int worldSize,
      int inputDim)
      : NCCLTest(
            path,
            rank,
            worldSize,
            c10d::kProcessGroupNCCLDefaultTimeout,
            inputDim) {}

  c10::intrusive_ptr<c10d::Work> run() {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    launchDeviceSleep();
    valueInitializationForSparse();
    auto results = pg_->allreduce_sparse(tensors_);
    return results;
  }
};

class BroadcastNCCLTest : public NCCLTest {
 public:
  BroadcastNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  c10::intrusive_ptr<c10d::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    ::c10d::BroadcastOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->broadcast(tensors_, options);
  }
};

class ReduceNCCLTest : public NCCLTest {
 public:
  ReduceNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  c10::intrusive_ptr<c10d::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    ::c10d::ReduceOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->reduce(tensors_, options);
  }
};

class AllgatherNCCLTest : public NCCLTest {
 public:
  AllgatherNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  c10::intrusive_ptr<c10d::Work> run() {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();

    return pg_->allgather(outputs_, tensors_);
  }
};

class AllgatherBaseNCCLTest : public NCCLTest {
 public:
  AllgatherBaseNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {
    output_tensor_ = at::empty({worldSize_, 3, 3}, at::kCUDA);
  }

  c10::intrusive_ptr<c10d::Work> run() {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    valueInitialization();
    // contains at least one element otherwise wouldn't run.
    // this is a flattened allgather, hence one rank contributes
    // only 1 tensor, regardless of number of devices
    return pg_->_allgather_base(output_tensor_, tensors_[0]);
  }

  at::Tensor getOutputTensor() {
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    return output_tensor_.cpu();
  }

  at::Tensor getInputTensor() {
    c10::cuda::CUDAMultiStreamGuard guard(streams_);
    return tensors_[0].cpu();
  }

 private:
  at::Tensor output_tensor_;
};

struct ReduceScatterNCCLTest : NCCLTest {
  ReduceScatterNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {}

  c10::intrusive_ptr<c10d::Work> run() {
    // For the duration of this function, make THC use our streams
    c10::cuda::CUDAMultiStreamGuard guard(streams_);

    at::cuda::OptionalCUDAGuard deviceGuard;
    launchDeviceSleep();

    // Launch value initialization for every tensor
    for (auto j = 0; j < worldSize_; ++j) {
      inputs_[0][j].fill_(rank_ * worldSize_ + j);
    }

    return pg_->reduce_scatter(tensors_, inputs_);
  }
};

class ReduceScatterBaseNCCLTest : public NCCLTest {
 public:
  ReduceScatterBaseNCCLTest(const std::string& path, int rank, int worldSize)
      : NCCLTest(path, rank, worldSize) {
    at::cuda::OptionalCUDAGuard deviceGuard;
    deviceGuard.set_index(static_cast<c10::DeviceIndex>(rank_));
    output_tensor_ = at::empty({1}, at::kCUDA);
    input_tensor_ = at::empty({worldSize}, at::kCUDA);
    for (const auto i : c10::irange(worldSize)) {
      input_tensor_[i] = i;
    }
  }

  c10::intrusive_ptr<c10d::Work> run() {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    launchDeviceSleep();
    return pg_->_reduce_scatter_base(output_tensor_, input_tensor_);
  }

  at::Tensor getOutputTensor() {
    at::cuda::CUDAMultiStreamGuard guard(streams_);
    return output_tensor_.cpu();
  }

  at::Tensor getInputTensor() {
    at::cuda::CUDAMultiStreamGuard guard(streams_);
    return input_tensor_.cpu();
  }

 private:
  at::Tensor output_tensor_;
  at::Tensor input_tensor_;
};

void testAllreduce(const std::string& path, int rank, int size) {
  auto test = AllreduceNCCLTest(path, rank, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  // Validation
  const int totalNumGPUs = test.numDevices() * size;
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  const auto tensors = test.getTensors();
  for (const auto& tensor : tensors) {
    const auto* const data = tensor.const_data_ptr<float>();
    for (const auto k : c10::irange(tensor.numel())) {
      EXPECT_EQ(data[k], expected)
          << "Allreduce outputs do not match expected outputs";
    }
  }
}

void testSparseAllreduce(const std::string& path, int rank, int size) {
  const int inputDim = 3;
  auto test = SparseAllreduceNCCLTest(path, rank, size, inputDim);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  const auto input_tensors = test.getTensors();

  // validate the work output is same as tensor
  auto output_tensor = work->result();
  // Validation
  int totalNumGPUs = test.numDevices() * size;
  // Add one since we are seeding with an additional 1 to prevent empty tensors
  totalNumGPUs++;
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  for (const auto i : c10::irange(input_tensors.size())) {
    const auto& tensor = input_tensors[i];

    // validate the tensor is sparse
    EXPECT_EQ(tensor.is_sparse(), true);

    auto indices = tensor._indices();
    auto values = tensor._values();

    // validate indices are expected size
    auto sizes = indices.sizes();
    EXPECT_EQ(sizes.size(), 2);
    if (sizes[0] == 1) {
      // row indices
      EXPECT_EQ(sizes[1], inputDim);
    } else if (sizes[0] == 2) {
      // coordinate indices
      EXPECT_EQ(sizes[1], inputDim * inputDim);
    }

    // validate all tensor values are expected value
    const auto* const data = values.const_data_ptr<float>();
    for (const auto k : c10::irange(values.numel())) {
      EXPECT_EQ(data[k], expected)
          << "Allreduce outputs do not match expected outputs";
    }

    // expect the input and output tensors should be the same
    auto input_dense = tensor.to_dense();
    auto output_dense = output_tensor[i].to(input_dense.device()).to_dense();
    EXPECT_TRUE(input_dense.allclose(output_dense));
  }
}

void testSparseAllreduceLarge(const std::string& path, int rank, int size) {
  const int inputDim = 2500;
  auto test = SparseAllreduceNCCLTest(path, rank, size, inputDim);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  const auto input_tensors = test.getTensors();

  // validate the work output is same as tensor
  auto output_tensor = work->result();
  // Validation
  int totalNumGPUs = test.numDevices() * size;
  // Add one since we are seeding with an additional 1 to prevent empty tensors
  totalNumGPUs++;
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  for (const auto i : c10::irange(input_tensors.size())) {
    const auto& tensor = input_tensors[i];

    // validate the tensor is sparse
    EXPECT_EQ(tensor.is_sparse(), true);

    auto indices = tensor._indices();
    auto values = tensor._values();

    // validate indices are expected size
    auto sizes = indices.sizes();
    EXPECT_EQ(sizes.size(), 2);
    if (sizes[0] == 1) {
      // row indices
      EXPECT_EQ(sizes[1], inputDim);
    } else if (sizes[0] == 2) {
      // coordinate indices
      EXPECT_EQ(sizes[1], inputDim * inputDim);
    }

    // validate all tensor values are expected value
    const auto* const data = values.const_data_ptr<float>();
    for (const auto k : c10::irange(values.numel())) {
      EXPECT_EQ(data[k], expected)
          << "Allreduce outputs do not match expected outputs";
    }

    // expect the input and output tensors should be the same
    auto input_dense = tensor.to_dense();
    auto output_dense = output_tensor[i].to(input_dense.device()).to_dense();
    EXPECT_TRUE(input_dense.allclose(output_dense));
  }
}

void testBroadcast(const std::string& path, int rank, int size) {
  auto test = BroadcastNCCLTest(path, rank, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // try every permutation of root rank and root tensor
  for (const auto rootRank : c10::irange(size)) {
    for (const auto rootTensor : c10::irange(numDevices)) {
      auto work = test.run(rootRank, rootTensor);

      // wait for work to complete
      test.wait(work);

      // Check results
      const auto expected = (rootRank * numDevices + rootTensor);
      const auto tensors = test.getTensors();
      for (const auto& tensor : tensors) {
        const auto* const data = tensor.const_data_ptr<float>();
        for (const auto k : c10::irange(tensor.numel())) {
          EXPECT_EQ(data[k], expected)
              << "Broadcast outputs do not match expected outputs";
        }
      }
    }
  }
}

void testReduce(const std::string& path, int rank, int size) {
  auto test = ReduceNCCLTest(path, rank, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // try every permutation of root rank and root tensor
  for (const auto rootRank : c10::irange(size)) {
    for (const auto rootTensor : c10::irange(numDevices)) {
      auto work = test.run(rootRank, rootTensor);

      // wait for work to complete
      test.wait(work);

      // Validation
      const int totalNumGPUs = numDevices * size;
      const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
      auto tensors = test.getTensors();
      if (rank == rootRank) {
        auto& tensor = tensors[rootTensor];
        auto data = tensor.data_ptr<float>();
        for (const auto k : c10::irange(tensor.numel())) {
          EXPECT_EQ(data[k], expected)
              << "Reduce outputs do not match expected outputs";
        }
      }
    }
  }
}

void testAllgather(const std::string& path, int rank, int size) {
  auto test = AllgatherNCCLTest(path, rank, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  // Validation
  auto tensors = test.getOutputTensors();
  // device index
  for (auto& device : tensors) {
    // rank index
    for (const auto j : c10::irange(device.size())) {
      const auto expected = j;
      auto& tensor = device[j];
      auto data = tensor.data_ptr<float>();
      for (const auto k : c10::irange(tensor.numel())) {
        EXPECT_EQ(data[k], expected)
            << "Allgather outputs do not match expected outputs";
      }
    }
  }
}

void testAllgatherBase(const std::string& path, int rank, int size) {
  auto test = AllgatherBaseNCCLTest(path, rank, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);
  // Validation
  auto output_tensor = test.getOutputTensor();
  auto input_tensor = test.getInputTensor();

  auto data = output_tensor.data_ptr<float>();

  // Rank index
  for (const auto i : c10::irange(output_tensor.numel())) {
    // expected is i // input.numel() <- rank, and each rank contributed rank *
    // num_gpu
    const auto expected = (i / input_tensor.numel()) * test.numDevices();
    EXPECT_EQ(data[i], expected)
        << "Allgather_base outputs do not match expected outputs";
  }
}
void testReduceScatterBase(const std::string& path, int rank, int size) {
  auto test = ReduceScatterBaseNCCLTest(path, rank, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);
  // Validation
  auto output_tensor = test.getOutputTensor();
  auto input_tensor = test.getInputTensor();

  auto data = output_tensor.data_ptr<float>();

  // Rank index
  for (const auto i : c10::irange(output_tensor.numel())) {
    // expected is i * input.numel() <- rank, and each rank contributed rank *
    // num_gpu
    const auto expected = size * rank * test.numDevices();
    EXPECT_EQ(data[i], expected)
        << "Reducescatter_base outputs do not match expected outputs";
  }
}

void testReduceScatter(const std::string& path, int rank, int size) {
  auto test = ReduceScatterNCCLTest(path, rank, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  const auto participants = size;
  const auto base = (participants * (participants - 1)) / 2;

  // Validation
  auto tensors = test.getTensors();
  const auto modifier = rank * participants;
  const auto expected = base * participants + modifier;
  auto& tensor = tensors[0];
  auto data = tensor.data_ptr<float>();
  for (const auto j : c10::irange(tensor.numel())) {
    EXPECT_EQ(data[j], expected)
        << "ReduceScatter outputs do not match expected outputs!";
  }
}

void testSequenceNumInit(const std::string& path, int rank, int size) {
  NCCLTest test(path, rank, size);
  test.initialize(rank, size);
  test.getProcessGroup()->setSequenceNumberForGroup();
  auto seqNum = test.getProcessGroup()->getSequenceNumberForGroup();
  EXPECT_EQ(seqNum, 0);
}

void testSplittingCommunicator(const std::string& path, int rank, int size) {
  auto test1 = BroadcastNCCLTest(path, rank, size);
  test1.initialize(rank, size);

  auto test2 = BroadcastNCCLTest(path, rank, size);
  test2.initialize(rank, size, test1.getProcessGroup());

  // Steal the broadcast test and issue it for both of our groups.
  // This ensures consistent full collective communication.  TODO:
  // maybe refactor the guts rather than copy-pasta, but it may not be
  // worth it.
  for (auto test : {&test1, &test2}) {
    const int numDevices = test->numDevices();
    // try every permutation of root rank and root tensor
    for (const auto rootRank : c10::irange(size)) {
      for (const auto rootTensor : c10::irange(numDevices)) {
        auto work = test->run(rootRank, rootTensor);
        test->wait(work);

        // Check results
        const auto expected = (rootRank * numDevices + rootTensor);
        const auto tensors = test->getTensors();
        for (const auto& tensor : tensors) {
          const auto* const data = tensor.const_data_ptr<float>();
          for (const auto k : c10::irange(tensor.numel())) {
            EXPECT_EQ(data[k], expected)
                << "Broadcast outputs do not match expected outputs";
          }
        }
      }
    }
  }

  // Now that we've run full operations on both the original and split process
  // group, ensure we saw exactly as many splits as we expected: 0 in the
  // original process group, and one per device in the second.
  EXPECT_EQ(test2.getProcessGroup()->getCommSplitCounter(), 0);
  EXPECT_EQ(test1.getProcessGroup()->getCommSplitCounter(), test1.numDevices());
}

// All testAbc's use this signature
using FuncType = void (*)(const std::string&, int, int);

class ProcessGroupNCCLTest : public ::testing::Test {
 protected:
  void SetUp() override {
    c10::initLogging();
    // Use WORLD_SIZE and RANK environmental variables to do multi-node
    // distributed testing
    auto sizeEnv = std::getenv("WORLD_SIZE");
    if (sizeEnv) {
      size_ = std::stoi(std::string(sizeEnv));
    }
    LOG(INFO) << "ProcessGroupNCCLTest world size: " << size_;
  }

  void TearDown() override {
    // Reset TORCH_NCCL_BLOCKING_WAIT environment variable after each run.
    ASSERT_TRUE(setenv(c10d::TORCH_NCCL_BLOCKING_WAIT[0].c_str(), "0", 1) == 0);
  }

  bool skipTest() {
    // Skip tests if CUDA is not available.
    if (!at::cuda::is_available()) {
      LOG(INFO) << "CUDA not available, skipping test";
      return true;
    }
    return false;
  }

  void multiThreadRun(FuncType testFunc) {
    TemporaryFile file;
    std::vector<std::thread> threads;
    threads.reserve(size_);
    for (const auto rank : c10::irange(size_)) {
      threads.emplace_back(testFunc, file.path, rank, size_);
    }
    for (const auto rank : c10::irange(size_)) {
      threads[rank].join();
    }
  }

  int size_{1};
};

TEST_F(ProcessGroupNCCLTest, CUDAEventCache) {
  if (skipTest()) {
    return;
  }

  // Test that the CUDAEventCache can be used to create CUDA events and reuse.
  auto event1 = c10d::ProcessGroupNCCL::CUDAEventCache::get().create(true);
  auto event2 = c10d::ProcessGroupNCCL::CUDAEventCache::get().create(false);

  auto event1_ptr = event1.get();
  auto event2_ptr = event2.get();
  // Mimic the behavior of the destroy of events.
  event1 = nullptr;
  event2 = nullptr;

  // Test that the CUDAEventCache is indeed reused.
  auto event3 = c10d::ProcessGroupNCCL::CUDAEventCache::get().create(true);
  auto event4 = c10d::ProcessGroupNCCL::CUDAEventCache::get().create(false);
  // The cache has been used up, new events should be created.
  auto event5 = c10d::ProcessGroupNCCL::CUDAEventCache::get().create(true);
  auto event6 = c10d::ProcessGroupNCCL::CUDAEventCache::get().create(false);
  EXPECT_EQ(event1_ptr, event3.get());
  EXPECT_EQ(event2_ptr, event4.get());
  EXPECT_NE(event1_ptr, event5.get());
  EXPECT_NE(event2_ptr, event6.get());
}

TEST_F(ProcessGroupNCCLTest, testAllreduce) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testAllreduce);
}

TEST_F(ProcessGroupNCCLTest, testBroadcast) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testBroadcast);
}

TEST_F(ProcessGroupNCCLTest, testReduce) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testReduce);
}

TEST_F(ProcessGroupNCCLTest, testAllgather) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testAllgather);
}

TEST_F(ProcessGroupNCCLTest, testAllgatherBase) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testAllgatherBase);
}

TEST_F(ProcessGroupNCCLTest, testReduceScatter) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testReduceScatter);
}

TEST_F(ProcessGroupNCCLTest, testSequenceNumInit) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testSequenceNumInit);
}

TEST_F(ProcessGroupNCCLTest, testReduceScatterBase) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testReduceScatterBase);
}

TEST_F(ProcessGroupNCCLTest, testBackendName) {
  if (skipTest()) {
    return;
  }
  TemporaryFile file;
  auto test = NCCLTestBase(file.path);
  test.initialize(/*rank=*/0, /*size=*/1);
  EXPECT_EQ(
      test.getProcessGroup()->getBackendName(),
      std::string(c10d::NCCL_BACKEND_NAME));
}

TEST_F(ProcessGroupNCCLTest, testSplittingCommunicator) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testSplittingCommunicator);
}

#ifdef IS_NCCLX
TEST_F(ProcessGroupNCCLTest, testSparseAllreduce) {
  if (skipTest()) {
    return;
  }
  multiThreadRun(testSparseAllreduce);
  multiThreadRun(testSparseAllreduceLarge);
}
#endif
