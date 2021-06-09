#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <unistd.h>

#include <c10d/ProcessGroupMPI.hpp>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// Wait for work to complete
std::vector<std::vector<at::Tensor>> waitWork(
    c10::intrusive_ptr<::c10d::ProcessGroupMPI> pg,
    std::vector<c10::intrusive_ptr<c10d::ProcessGroup::Work>> works) {
  std::vector<std::vector<at::Tensor>> outputTensors;
  for (auto& work : works) {
    try {
      work->wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception received: " << ex.what() << std::endl;
      pg->abort();
    }
    outputTensors.emplace_back(work->result());
  }
  return outputTensors;
}

// Wait using Futures
std::vector<std::vector<at::Tensor>> waitFuture(
    c10::intrusive_ptr<::c10d::ProcessGroupMPI> pg,
    std::vector<c10::intrusive_ptr<c10d::ProcessGroup::Work>> works) {
  std::vector<std::vector<at::Tensor>> outputTensors;
  for (auto& work : works) {
    auto fut = work->getFuture();
    try {
      fut->wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception received: " << ex.what() << std::endl;
      pg->abort();
    }
    auto result = fut->value();
    if (result.isNone()) {
      outputTensors.emplace_back();
    } else if (result.isTensorList()) {
      outputTensors.emplace_back(result.toTensorVector());
    } else {
      throw std::runtime_error("future result should be tensor list or none");
    }
  }
  return outputTensors;
}

void testAllreduce(int iter = 1000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();

  // Generate inputs
  std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i;
    std::vector<at::Tensor> tensors = {tensor};

    // Queue the work.
    c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
        pg->allreduce(tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Get the world size
  auto worldSize = pg->getSize();

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    const auto expected = worldSize * i;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != expected) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
}

void testBroadcast(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto i = 0; i < iter; ++i) {
    auto tensors = std::vector<at::Tensor>();
    if (pg->getRank() == 0) {
      auto tensor = at::ones({16, 16}) * i;
      tensors = std::vector<at::Tensor>({tensor});
    } else {
      auto tensor = at::zeros({16, 16});
      tensors = std::vector<at::Tensor>({tensor});
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
        pg->broadcast(tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    const auto expected = i;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != expected) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
}

void testReduce(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i;
    auto tensors = std::vector<at::Tensor>({tensor});

    // Queue the work.
    c10::intrusive_ptr<::c10d::ProcessGroup::Work> work = pg->reduce(tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Get the world size
  auto worldSize = pg->getSize();

  if (pg->getRank() == 0) {
    // Verify outputs
    for (int i = 0; i < iter; ++i) {
      const auto expected = worldSize * i;
      auto data = outputTensors[i][0].data_ptr<float>();
      for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
        if (data[j] != expected) {
          throw std::runtime_error("BOOM!");
        }
      }
    }
  }
}

void testAllgather(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i * rank;
    auto tensors = std::vector<at::Tensor>({tensor});
    auto outputs = std::vector<std::vector<at::Tensor>>(1);
    outputs[0].resize(worldSize);
    for (auto j = 0; j < worldSize; ++j) {
      outputs[0][j] = at::zeros({16, 16});
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
      pg->allgather(outputs, tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    for (int j = 0; j < worldSize; ++j) {
      const auto expected = i * j;
      auto data = outputTensors[i][j].data_ptr<float>();
      for (auto k = 0; k < outputTensors[i][j].numel(); ++k) {
        if (data[k] != expected) {
          throw std::runtime_error("BOOM!");
        }
      }
    }
  }
}

void testGather(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i * rank;
    auto tensors = std::vector<at::Tensor>({tensor});
    auto outputs = std::vector<std::vector<at::Tensor>>(0);
    if (rank == 0) {
      outputs = std::vector<std::vector<at::Tensor>>(1);
      outputs[0].resize(worldSize);
      for (auto j = 0; j < worldSize; ++j) {
        outputs[0][j] = at::zeros({16, 16});
      }
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
        pg->gather(outputs, tensors);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  if (rank == 0) {
    for (int i = 0; i < iter; ++i) {
      for (int j = 0; j < worldSize; ++j) {
        const auto expected = i * j;
        auto data = outputTensors[i][j].data_ptr<float>();
        for (auto k = 0; k < outputTensors[i][j].numel(); ++k) {
          if (data[k] != expected) {
            throw std::runtime_error("BOOM!");
          }
        }
      }
    }
  } else {
    for (int i = 0; i < iter; ++i) {
      if (outputTensors[i].size() != 0) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
}

void testScatter(int iter = 1) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::zeros({16, 16});
    auto tensors = std::vector<at::Tensor>({tensor});
    auto inputs = std::vector<std::vector<at::Tensor>>(0);
    if (rank == 0) {
      inputs = std::vector<std::vector<at::Tensor>>(1);
      inputs[0].resize(worldSize);
      for (auto j = 0; j < worldSize; ++j) {
        inputs[0][j] = at::ones({16, 16}) * i * j;
      }
    }

    // Queue the work.
    c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
      pg->scatter(tensors, inputs);
    works.push_back(std::move(work));
  }

  auto outputTensors = waitFuture(pg, works);

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    for (int j = 0; j < worldSize; ++j) {
      const auto expected = i * j;
      auto data = outputTensors[i][0].data_ptr<float>();
      for (auto k = 0; k < outputTensors[i][0].numel(); ++k) {
        if (data[k] != expected) {
          throw std::runtime_error("BOOM!");
        }
      }
    }
  }
}

void testSendRecv(bool recvAnysource, int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // Generate inputs
  std::vector<c10::intrusive_ptr<::c10d::ProcessGroup::Work>> works;

  // pg->send does not keep sent tensors alive, so we need to.
  std::vector<std::vector<at::Tensor>> sendTensors(iter);
  auto rank = pg->getRank();
  for (auto i = 0; i < iter; ++i) {
    if (rank == 0) {
      auto tensor = at::ones({16, 16}) * i;
      sendTensors[i] = std::vector<at::Tensor>({tensor});

      // Queue the work.
      c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
          pg->send(sendTensors[i], 1, 0);
      works.push_back(std::move(work));
    } else {
      auto tensor = at::zeros({16, 16});
      auto recvTensors = std::vector<at::Tensor>({tensor});

      // Queue the work.
      if (!recvAnysource) {
        c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
            pg->recv(recvTensors, 0, 0);
        works.push_back(std::move(work));
      } else {
        c10::intrusive_ptr<::c10d::ProcessGroup::Work> work =
            pg->recvAnysource(recvTensors, 0);
        works.push_back(std::move(work));
      }
    }
  }

  auto outputTensors = waitWork(pg, works);
  if (rank == 0) {
    return;
  }

  std::vector<int> srcRanks;
  if (recvAnysource) {
    for (const auto& work : works) {
      srcRanks.push_back(work->sourceRank());
    }
  }

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    if (recvAnysource && srcRanks[i] != 0) {
      throw std::runtime_error("src rank is wrong for recvAnysource");
    }
    const auto expected = i;
    auto data = outputTensors[i][0].data_ptr<float>();
    for (auto j = 0; j < outputTensors[i][0].numel(); ++j) {
      if (data[j] != expected) {
        throw std::runtime_error("BOOM!");
      }
    }
  }
}

void testBackendName() {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  if (pg->getBackendName() != std::string(c10d::MPI_BACKEND_NAME)) {
    throw std::runtime_error("BOOM!");
  }
}

int main(int argc, char** argv) {
#ifdef MPIEXEC
  // If we are within an openmpi mpirun, then skip the exec
  if (!std::getenv("OMPI_COMM_WORLD_SIZE")) {
    std::cout << "Execute mpiexec from: " << STR(MPIEXEC) << std::endl;
    execl(STR(MPIEXEC), "-np 2", argv[0], (char*)nullptr);
  }

  testAllreduce();
  testBroadcast();
  testReduce();
  testAllgather();
  testGather();
  testScatter();
  testSendRecv(false);
  testSendRecv(true);
  testBackendName();

  std::cout << "Test successful" << std::endl;
#else
  std::cout << "MPI executable not found, skipping test" << std::endl;
#endif
  return EXIT_SUCCESS;
}
