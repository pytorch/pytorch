#include "ProcessGroupNCCL.hpp"
#include "CUDAUtils.hpp"
#include "FileStore.hpp"
#include "private/CUDAUtils.hpp"

#include "test/CUDATest.hpp"
#include "test/TestUtils.hpp"

#include <iostream>

using namespace c10d::test;

using c10d::CUDADevice;
using c10d::CUDAStream;
using c10d::ProcessGroup;
using c10d::THCStreamGuard;

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
  NCCLTest(const std::string& path)
      : NCCLTestBase(path),
        numDevices_(cudaNumDevices()),
        state_(::at::globalContext().lazyInitCUDA()) {
    const auto& type = at::getType(at::kCUDA, at::kFloat);

    // Each device has a single tensor to perf the NCCL op
    inputs_.resize(numDevices_);
    for (auto i = 0; i < numDevices_; i++) {
      CUDADevice device(i);
      inputs_[i] = type.tensor({3, 3});
    }

    // Allocate a stream per device.
    //
    // The "current stream" is set globally per device in THC, so we
    // can't make two tensors on the same device use different streams
    // and pass this along to the collective (since it uses the THC
    // getters to retrieve the current stream).
    //
    streams_.resize(numDevices_);
    for (auto i = 0; i < numDevices_; i++) {
      CUDADevice device(i);
      streams_[i] = CUDAStream::create();
    }
  }

  std::vector<THCStreamGuard> createStreamGuard() {
    std::vector<THCStreamGuard> guards;
    for (auto& stream : streams_) {
      guards.push_back(std::move(THCStreamGuard(state_, stream)));
    }
    return guards;
  }

  void wait(std::shared_ptr<ProcessGroup::Work>& work) {
    auto guards = createStreamGuard();
    work->wait();
  }

  std::vector<at::Tensor> getTensors() {
    std::vector<at::Tensor> outputs(numDevices_);

    // For the duration of this function, make THC use our streams
    auto guards = createStreamGuard();

    // Copy inputs to outputs
    for (auto i = 0; i < numDevices_; i++) {
      cudaStreamSynchronize(streams_[i].getStream());
      outputs[i] = inputs_[i].toBackend(at::kCPU);
    }

    return outputs;
  }

  int numDevices() const {
    return numDevices_;
  }

 protected:
  const int numDevices_;
  THCState* state_;
  std::vector<at::Tensor> inputs_;
  std::vector<CUDAStream> streams_;
};

class AllreduceNCCLTest : public NCCLTest {
 public:
  AllreduceNCCLTest(const std::string& path) : NCCLTest(path) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run() {
    // For the duration of this function, make THC use our streams
    auto guards = createStreamGuard();

    // Launch sleep on every device
    for (auto i = 0; i < numDevices_; i++) {
      CUDADevice device(i);
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numDevices_; i++) {
      CUDADevice device(i);
      inputs_[i].fill_(pg_->getRank() * numDevices_ + i);
    }

    return pg_->allreduce(inputs_);
  }
};

class BroadcastNCCLTest : public NCCLTest {
 public:
  BroadcastNCCLTest(const std::string& path) : NCCLTest(path) {}

  std::shared_ptr<c10d::ProcessGroup::Work> run(int rootRank, int rootTensor) {
    // For the duration of this function, make THC use our streams
    auto guards = createStreamGuard();

    // Launch sleep on every device
    for (auto i = 0; i < numDevices_; i++) {
      CUDADevice device(i);
      cudaSleep(streams_[i], 2000 * 1000 * 1000);
    }

    // Launch value initialization for every tensor
    for (auto i = 0; i < numDevices_; i++) {
      CUDADevice device(i);
      inputs_[i].fill_(pg_->getRank() * numDevices_ + i);
    }

    ::c10d::BroadcastOptions options;
    options.rootRank = rootRank;
    options.rootTensor = rootTensor;
    return pg_->broadcast(inputs_, options);
  }
};

void testAllreduce(const std::string& path, int rank, int size) {
  auto test = AllreduceNCCLTest(path);
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
  auto test = BroadcastNCCLTest(path);
  test.initialize(rank, size);

  const int numDevices = test.numDevices();
  // Try every permutation of root rank and root tensor
  for (auto rootRank = 0; rootRank < size; rootRank++) {
    for (auto rootTensor = 0; rootTensor < numDevices; rootTensor++) {
      auto work = test.run(rootRank, rootTensor);

      // Wait for work to complete
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
  {
    TemporaryFile file;
    testAllreduce(file.path, rank, size);
  }
  {
    TemporaryFile file;
    testBroadcast(file.path, rank, size);
  }
  return EXIT_SUCCESS;
}
