#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <unistd.h>

#include <c10d/ProcessGroupMPI.hpp>
#include <gtest/gtest.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// Wait for work to complete
void waitWork(
    std::shared_ptr<c10d::ProcessGroupMPI> pg,
    std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> works) {
  for (auto& work : works) {
    try {
      work->wait();
    } catch (const std::exception& ex) {
      std::cerr << "Exception received: " << ex.what() << std::endl;
      pg->abort();
    }
  }
}

void testAllreduce(int iter = 1000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // Generate inputs
  std::vector<std::vector<at::Tensor>> allTensors(iter);
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i;
    allTensors[i] = std::vector<at::Tensor>({tensor});
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto& tensors : allTensors) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->allreduce(tensors);
    works.push_back(std::move(work));
  }

  waitWork(pg, works);

  // Get the world size
  auto worldSize = pg->getSize();

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    const auto expected = worldSize * i;
    auto data = allTensors[i][0].data<float>();
    for (auto j = 0; j < allTensors[i][0].numel(); ++j) {
      if (data[j] != expected) {
        EXPECT_EQ(data[j], expected)
            << "Allreduce ouputs do not match expected outputs";
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
      auto tensor = at::ones({16, 16}) * i;
      allTensors[i] = std::vector<at::Tensor>({tensor});
    } else {
      auto tensor = at::zeros({16, 16});
      allTensors[i] = std::vector<at::Tensor>({tensor});
    }
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto& tensors : allTensors) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->broadcast(tensors);
    works.push_back(std::move(work));
  }

  waitWork(pg, works);

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    const auto expected = i;
    auto data = allTensors[i][0].data<float>();
    for (auto j = 0; j < allTensors[i][0].numel(); ++j) {
      if (data[j] != expected) {
        EXPECT_EQ(data[j], expected)
            << "Broadcast ouputs do not match expected outputs";
      }
    }
  }
}

void testReduce(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // Generate inputs
  std::vector<std::vector<at::Tensor>> allTensors(iter);

  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i;
    allTensors[i] = std::vector<at::Tensor>({tensor});
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (auto& tensors : allTensors) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->reduce(tensors);
    works.push_back(std::move(work));
  }

  waitWork(pg, works);

  // Get the world size
  auto worldSize = pg->getSize();

  if (pg->getRank() == 0) {
    // Verify outputs
    for (int i = 0; i < iter; ++i) {
      const auto expected = worldSize * i;
      auto data = allTensors[i][0].data<float>();
      for (auto j = 0; j < allTensors[i][0].numel(); ++j) {
        if (data[j] != expected) {
          EXPECT_EQ(data[j], expected)
              << "Reduce ouputs do not match expected outputs";
        }
      }
    }
  }
}

void testAllgather(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<std::vector<at::Tensor>> allTensors(iter);
  std::vector<std::vector<std::vector<at::Tensor>>> allOutputTensors(iter);

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i * rank;
    allTensors[i] = std::vector<at::Tensor>({tensor});
    allOutputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
    allOutputTensors[i][0].resize(worldSize);
    for (auto j = 0; j < worldSize; ++j) {
      allOutputTensors[i][0][j] = at::zeros({16, 16});
    }
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (size_t i = 0; i < allTensors.size(); ++i) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work =
        pg->allgather(allOutputTensors[i], allTensors[i]);
    works.push_back(std::move(work));
  }

  waitWork(pg, works);

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    for (int j = 0; j < worldSize; ++j) {
      const auto expected = i * j;
      auto data = allOutputTensors[i][0][j].data<float>();
      for (auto k = 0; k < allOutputTensors[i][0][j].numel(); ++k) {
        if (data[k] != expected) {
          EXPECT_EQ(data[k], expected)
              << "Allgather ouputs do not match expected outputs";
        }
      }
    }
  }
}

void testGather(int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  std::vector<std::vector<at::Tensor>> allTensors(iter);
  std::vector<std::vector<std::vector<at::Tensor>>> allOutputTensors(iter);

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::ones({16, 16}) * i * rank;
    allTensors[i] = std::vector<at::Tensor>({tensor});
    if (rank == 0) {
      allOutputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
      allOutputTensors[i][0].resize(worldSize);
      for (auto j = 0; j < worldSize; ++j) {
        allOutputTensors[i][0][j] = at::zeros({16, 16});
      }
    } else {
      allOutputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
    }
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (size_t i = 0; i < allTensors.size(); ++i) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work =
        pg->gather(allOutputTensors[i], allTensors[i]);
    works.push_back(std::move(work));
  }

  waitWork(pg, works);

  // Verify outputs
  if (rank == 0) {
    for (int i = 0; i < iter; ++i) {
      for (int j = 0; j < worldSize; ++j) {
        const auto expected = i * j;
        auto data = allOutputTensors[i][0][j].data<float>();
        for (auto k = 0; k < allOutputTensors[i][0][j].numel(); ++k) {
          if (data[k] != expected) {
            EXPECT_EQ(data[k], expected)
                << "Gather ouputs do not match expected outputs";
          }
        }
      }
    }
  }
}

void testScatter(int iter = 1) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();

  std::vector<std::vector<std::vector<at::Tensor>>> allInputTensors(iter);
  std::vector<std::vector<at::Tensor>> allTensors(iter);

  // Get the world size
  auto worldSize = pg->getSize();
  auto rank = pg->getRank();

  // Generate inputs
  for (auto i = 0; i < iter; ++i) {
    auto tensor = at::zeros({16, 16});
    allTensors[i] = std::vector<at::Tensor>({tensor});
    if (rank == 0) {
      allInputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
      allInputTensors[i][0].resize(worldSize);
      for (auto j = 0; j < worldSize; ++j) {
        allInputTensors[i][0][j] = at::ones({16, 16}) * rank * i;
      }
    } else {
      allInputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
    }
  }

  std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
  for (size_t i = 0; i < allTensors.size(); ++i) {
    // Kick off work
    std::shared_ptr<::c10d::ProcessGroup::Work> work =
        pg->scatter(allTensors[i], allInputTensors[i]);
    works.push_back(std::move(work));
  }

  waitWork(pg, works);

  // Verify outputs
  for (int i = 0; i < iter; ++i) {
    for (int j = 0; j < worldSize; ++j) {
      const auto expected = i * j;
      auto data = allTensors[i][0].data<float>();
      for (auto k = 0; k < allTensors[i][0].numel(); ++k) {
        if (data[k] != expected) {
          EXPECT_EQ(data[k], expected)
              << "Scatter ouputs do not match expected outputs";
        }
      }
    }
  }
}

void testSendRecv(bool recvAnysource, int iter = 10000) {
  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();
  // Generate inputs
  std::vector<std::vector<at::Tensor>> allTensors(iter);
  auto rank = pg->getRank();
  for (auto i = 0; i < iter; ++i) {
    if (rank == 0) {
      auto tensor = at::ones({16, 16}) * i;
      allTensors[i] = std::vector<at::Tensor>({tensor});
    } else {
      auto tensor = at::zeros({16, 16});
      allTensors[i] = std::vector<at::Tensor>({tensor});
    }
  }

  if (rank == 0) {
    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (auto& tensors : allTensors) {
      // Kick off work
      std::shared_ptr<::c10d::ProcessGroup::Work> work =
          pg->send(tensors, 1, 0);
      works.push_back(std::move(work));
    }
    waitWork(pg, works);
  }
  if (rank == 1) {
    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    std::vector<int> srcRanks(allTensors.size(), -1);
    size_t i = 0;
    for (auto& tensors : allTensors) {
      // Kick off work
      if (!recvAnysource) {
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->recv(tensors, 0, 0);
        works.push_back(std::move(work));
      } else {
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->recvAnysource(tensors, 0);
        works.push_back(std::move(work));
      }
      ++i;
    }
    waitWork(pg, works);
    // Verify outputs
    for (int i = 0; i < iter; ++i) {
      if (recvAnysource && srcRanks[i] != 0) {
        throw std::runtime_error("src rank is wrong for recvAnysource");
      }
      const auto expected = i;
      auto data = allTensors[i][0].data<float>();
      for (auto j = 0; j < allTensors[i][0].numel(); ++j) {
        if (data[j] != expected) {
          EXPECT_EQ(data[j], expected)
              << "SendRecv ouputs do not match expected outputs";
        }
      }
    }
  }
}

class ProcessGroupMPITest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef MPIEXEC
    // If we are within an openmpi mpirun, then skip the exec
    if (!std::getenv("OMPI_COMM_WORLD_SIZE")) {
      LOG(INFO) << "Execute mpiexec from: " << STR(MPIEXEC);
      execl(STR(MPIEXEC), "-np 2", argv[0], (char*)nullptr);
    }
#endif
  }

  void skipTest() {
#ifdef MPIEXEC
    return false;
#else
    LOG(INFO) << "MPI executable not found, skipping test";
    return true;
#endif
  }
};

TEST_F(ProcessGroupMPITest, testAllreduce) {
  if (skipTest()) {
    return;
  }
  testAllreduce();
}

TEST_F(ProcessGroupMPITest, testBroadcast) {
  if (skipTest()) {
    return;
  }
  testBroadcast();
}

TEST_F(ProcessGroupMPITest, testReduce) {
  if (skipTest()) {
    return;
  }
  testReduce();
}

TEST_F(ProcessGroupMPITest, testAllgather) {
  if (skipTest()) {
    return;
  }
  testAllgather();
}

TEST_F(ProcessGroupMPITest, testGather) {
  if (skipTest()) {
    return;
  }
  testGather();
}

TEST_F(ProcessGroupMPITest, testScatter) {
  if (skipTest()) {
    return;
  }
  testScatter();
}

TEST_F(ProcessGroupMPITest, testSendRecv) {
  if (skipTest()) {
    return;
  }
  testSendRecv(false);
}

TEST_F(ProcessGroupMPITest, testSendRecvAnySrc) {
  if (skipTest()) {
    return;
  }
  testSendRecv(true);
}
