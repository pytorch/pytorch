#include <iostream>

#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/test/CUDATest.hpp>
#include <c10d/test/TestUtils.hpp>

#include <ATen/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAStream.h>

using namespace c10d::test;

using at::cuda::CUDAStream;
using c10d::ProcessGroup;

class NCCLTestBase {
 public:
  NCCLTestBase(const std::string& path) : path_(path) {}

  NCCLTestBase(NCCLTestBase&& other) {
    path_ = std::move(other.path_);
    pg_ = std::move(other.pg_);
  }

  ::c10d::ProcessGroupNCCL& getProcessGroup() {
    return *pg_;
  }

  void initialize(int rank, int size) {
    auto store = std::make_shared<::c10d::FileStore>(path_);

    pg_ = std::unique_ptr<::c10d::ProcessGroupNCCL>(
        new ::c10d::ProcessGroupNCCL(store, rank, size));
  }

 protected:
  std::string path_;
  std::unique_ptr<::c10d::ProcessGroupNCCL> pg_;
};

class NCCLTest : public NCCLTestBase {
 public:
  NCCLTest(const std::string& path, int worldSize)
      : NCCLTestBase(path),
        numDevices_(cudaNumDevices()),
        state_(::at::globalContext().lazyInitCUDA()),
        worldSize_(worldSize) {
    // Each device has a single tensor to perf the NCCL op
    inputs_.resize(numDevices_);
    outputs_.resize(numDevices_);
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; ++i) {
      deviceGuard.set_index(i);
      inputs_[i] = at::empty({3, 3}, at::kCUDA);
      outputs_[i].resize(worldSize_ * numDevices_);
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        outputs_[i][j] = at::empty({3, 3}, at::kCUDA);
      }
    }

    // Allocate a stream per device.
    //
    // The "current stream" is set globally per device in THC, so we
    // can't make two tensors on the same device use different streams
    // and pass this along to the collective (since it uses the THC
    // getters to retrieve the current stream).
    //
    streams_.reserve(numDevices_);
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      streams_.push_back(at::cuda::getStreamFromPool());
    }
  }

  void wait(std::shared_ptr<ProcessGroup::Work>& work) {
    at::cuda::CUDAMultiStreamGuard guard(streams_);
    work->wait();
  }

  std::vector<at::Tensor> getTensors() {
    std::vector<at::Tensor> outputs(numDevices_);

    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (auto i = 0; i < numDevices_; i++) {
      cudaStreamSynchronize(streams_[i].stream());
      outputs[i] = inputs_[i].cpu();
    }

    return outputs;
  }

  std::vector<std::vector<at::Tensor>> getOutputTensors() {
    std::vector<std::vector<at::Tensor>> outputs(numDevices_);
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i] = std::vector<at::Tensor>(worldSize_ * numDevices_);
    }

    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Copy inputs to outputs
    for (auto i = 0; i < numDevices_; ++i) {
      cudaStreamSynchronize(streams_[i].stream());
      for (auto j = 0; j < worldSize_ * numDevices_; ++j) {
        outputs[i][j] = outputs_[i][j].cpu();
      }
    }
    return outputs;
  }

  int numDevices() const {
    return numDevices_;
  }

 protected:
  const int numDevices_;
  THCState* state_;
  int worldSize_;
  std::vector<at::Tensor> inputs_;
  std::vector<std::vector<at::Tensor>> outputs_;
  std::vector<CUDAStream> streams_;
};

class AllreduceNCCLTest : public NCCLTest {
 public:
  AllreduceNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run() {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every device
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      inputs_[i].fill_(pg_->getRank() * numDevices_ + i);
    }

    return pg_->allreduce(inputs_);
  }
};

class BroadcastNCCLTest : public NCCLTest {
 public:
  BroadcastNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every device
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      inputs_[i].fill_(pg_->getRank() * numDevices_ + i);
    }

    ::c10d::BroadcastOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->broadcast(inputs_, options);
  }
};

class ReduceNCCLTest : public NCCLTest {
 public:
  ReduceNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every device
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      inputs_[i].fill_(pg_->getRank() * numDevices_ + i);
    }

    ::c10d::ReduceOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->reduce(inputs_, options);
  }
};

class AllgatherNCCLTest : public NCCLTest {
 public:
  AllgatherNCCLTest(const std::string& path, int worldSize)
      : NCCLTest(path, worldSize) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run() {
    // For the duration of this function, make THC use our streams
    at::cuda::CUDAMultiStreamGuard guard(streams_);

    // Launch sleep on every device
    at::cuda::OptionalCUDAGuard deviceGuard;
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numDevices_; i++) {
      deviceGuard.set_index(i);
      inputs_[i].fill_(pg_->getRank() * numDevices_ + i);
    }

    return pg_->allgather(outputs_, inputs_);
  }
};

void testAllreduce(const std::string& path, int rank, int size) {
  auto test = AllreduceNCCLTest(path, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  // Validation
  const int totalNumGPUs = test.numDevices() * size;
  const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
  auto tensors = test.getTensors();
  for (size_t j = 0; j < tensors.size(); j++) {
    auto& tensor = tensors[j];
    auto data = tensor.data<float>();
    for (auto k = 0; k < tensor.numel(); k++) {
      if (data[k] != expected) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
  std::cout << "Allreduce test successful" << std::endl;
}

void testBroadcast(const std::string& path, int rank, int size) {
  auto test = BroadcastNCCLTest(path, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // try every permutation of root rank and root tensor
  for (auto rootRank = 0; rootRank < size; rootRank++) {
    for (auto rootTensor = 0; rootTensor < numDevices; rootTensor++) {
      auto work = test.run(rootRank, rootTensor);

      // wait for work to complete
      test.wait(work);

      // Check results
      const auto expected = (rootRank * numDevices + rootTensor);
      auto tensors = test.getTensors();
      for (size_t j = 0; j < tensors.size(); j++) {
        auto& tensor = tensors[j];
        auto data = tensor.data<float>();
        for (auto k = 0; k < tensor.numel(); k++) {
          if (data[k] != expected) {
            throw std::runtime_error("BOOM!");
          }
        }
      }
    }
  }
  std::cout << "Broadcast test successful" << std::endl;
}

void testReduce(const std::string& path, int rank, int size) {
  auto test = ReduceNCCLTest(path, size);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // try every permutation of root rank and root tensor
  for (auto rootRank = 0; rootRank < size; rootRank++) {
    for (auto rootTensor = 0; rootTensor < numDevices; rootTensor++) {
      auto work = test.run(rootRank, rootTensor);

      // wait for work to complete
      test.wait(work);

      // Validation
      const int totalNumGPUs = numDevices * size;
      const auto expected = (totalNumGPUs * (totalNumGPUs - 1)) / 2;
      auto tensors = test.getTensors();
      if (rank == rootRank) {
        auto& tensor = tensors[rootTensor];
        auto data = tensor.data<float>();
        for (auto k = 0; k < tensor.numel(); k++) {
          if (data[k] != expected) {
            throw std::runtime_error("BOOM!");
          }
        }
      }
    }
  }
  std::cout << "Reduce test successful" << std::endl;
}

void testAllgather(const std::string& path, int rank, int size) {
  auto test = AllgatherNCCLTest(path, size);
  test.initialize(rank, size);
  auto work = test.run();
  // Wait for work to finish
  test.wait(work);

  // Validation
  auto tensors = test.getOutputTensors();
  // device index
  for (size_t i = 0; i < tensors.size(); ++i) {
    // rank index
    for (size_t j = 0; j < tensors[i].size(); ++j) {
      const auto expected = j;
      auto& tensor = tensors[i][j];
      auto data = tensor.data<float>();
      for (auto k = 0; k < tensor.numel(); k++) {
        if (data[k] != expected) {
          throw std::runtime_error("BOOM!");
        }
      }
    }
  }
  std::cout << "Allgather test successful" << std::endl;
}

int main(int argc, char** argv) {
  // Use WORLD_SIZE and RANK environmental variables to do multi-node
  // distributed testing
  auto sizeEnv = std::getenv("WORLD_SIZE");
  auto rankEnv = std::getenv("RANK");

  int size = 1;
  int rank = 0;

  if (sizeEnv && rankEnv) {
    size = std::stoi(std::string(sizeEnv));
    rank = std::stoi(std::string(rankEnv));
    std::cout << "Multi-node world size: " << size << " rank: " << rank
              << std::endl;
  }
  // TemporaryFile file;
  TemporaryFile file;

  testAllreduce(file.path, rank, size);
  testBroadcast(file.path, rank, size);
  testReduce(file.path, rank, size);
  testAllgather(file.path, rank, size);

  return EXIT_SUCCESS;
}
