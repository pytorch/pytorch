#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <unistd.h>

#include "ProcessGroupMPI.hpp"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

void testAllreduce(int iter = 1000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
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
      std::cerr << "Exception received: " << work->exception().what()
                << std::endl;
      pg->abort();
    }
  }

  // Get the world size
  auto worldSize = pg->getSize();

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
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
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // Generate inputs
  std::vector<std::vector<at::Tensor>> allTensors(iter);

  for (auto i = 0; i < iter; ++i) {
    if (pg->getRank() == 0) {
      auto tensor = at::ones(at::CPU(at::kFloat), {16, 16}) * i;
      allTensors[i] = std::vector<at::Tensor>({tensor});
    } else {
      auto tensor = at::zeros(at::CPU(at::kFloat), {16, 16});
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
      std::cerr << "Exception received: " << work->exception().what()
                << std::endl;
      pg->abort();
    }
  }

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
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
#ifdef MPIEXEC
  // If we are within an openmpi mpirun, then skip the exec
  if (!std::getenv("OMPI_COMM_WORLD_SIZE")) {
    std::cout << "Execute mpiexec from: " << STR(MPIEXEC) << std::endl;
    execl(STR(MPIEXEC), "-np 2", argv[0]);
  }

  testAllreduce();
  testBroadcast();

  std::cout << "Test successful" << std::endl;
#else
  std::cout << "MPI executable not found, skipping test" << std::endl;
#endif
  return EXIT_SUCCESS;
}
