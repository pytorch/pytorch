#include <iostream>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <string>
#include <iostream>

#include "ProcessGroupMPI.hpp"

// Create the MPI process group
std::unique_ptr<::c10d::ProcessGroupMPI> createProcessGroup() {
  auto pg = std::unique_ptr<::c10d::ProcessGroupMPI>(
      new ::c10d::ProcessGroupMPI());
  return pg;
}

void testAllreduce(int iter = 1000) {
  auto pg = createProcessGroup();
  // Generate inputs
  std::vector<std::vector<at::Tensor>> allTensors(iter);
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones(at::CPU(at::kFloat), {16, 16}) * i;
    allTensors[i] = std::vector<at::Tensor>({tensor});
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto& tensors : allTensors) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->allreduce(tensors);
    works.push_back(std::move(work));
  }

  for (auto& work : works) {
    // Wait for work to complete
    if (!work->wait()) {
      pg->abort();
    }
  }

  // Get the world size
  auto worldSize = pg->getSize();

  // Verify outputs
  for (size_t i = 0; i < iter; ++i) {
    const auto expected = worldSize * i;
    auto data = allTensors[i][0].data<float>();
    for (auto i = 0; i < allTensors[i][0].numel(); ++i) {
      if (data[i] != expected) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
}

void testBroadcast(int iter = 10000) {
  auto pg = createProcessGroup();
  // Generate inputs
  std::vector<std::vector<at::Tensor>> allTensors(iter);

  for (auto i = 0; i < iter; ++i) {
    if (pg->getRank() == 0) {
      auto tensor = at::ones(at::CPU(at::kFloat), {16, 16}) * i;
      allTensors[i] = std::vector<at::Tensor>({tensor});
    } else {
      auto tensor = at::zeros(at::CPU(at::kFloat), {16, 16}) ;
      allTensors[i] = std::vector<at::Tensor>({tensor});
    }
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto& tensors : allTensors) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->broadcast(tensors);
    works.push_back(std::move(work));
  }

  for (auto& work : works) {
    // Wait for work to complete
    if (!work->wait()) {
      pg->abort();
    }
  }

  // Verify outputs
  for (size_t i = 0; i < iter; ++i) {
    const auto expected = i;
    auto data = allTensors[i][0].data<float>();
    for (auto i = 0; i < allTensors[i][0].numel(); ++i) {
      if (data[i] != expected) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
}

int main(int argc, char** argv) {
  testAllreduce();
  testBroadcast();

  // Needs to be called at the end
  ::c10d::ProcessGroupMPI::finalize();

  std::cout << "Test successful" << std::endl;
  return EXIT_SUCCESS;
}
