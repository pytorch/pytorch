#include "../base/data_channels/DataChannelTCP.hpp"
#include "TestUtils.hpp"

#include <THPP/tensors/THTensor.hpp>

#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

constexpr int WORKERS_NUM = 2;
constexpr int MASTER_PORT = 45679;

std::vector<std::thread> g_all_workers;
std::mutex g_mutex;

void master()
{
  g_mutex.lock();
  setenv(WORLD_SIZE_ENV, std::to_string((WORKERS_NUM + 1)).data(), 1);
  setenv(RANK_ENV, "0", 1);
  setenv(MASTER_PORT_ENV, std::to_string(MASTER_PORT).data(), 1);
  auto masterChannel = std::make_shared<thd::DataChannelTCP>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  // wait a long time before init
  std::this_thread::sleep_for(std::chrono::seconds(4));

  assert(masterChannel->init());

  auto float_tensor = buildTensor<float>({1, 2, 3}, 4);
  masterChannel->broadcast(*float_tensor, 0); // send good tensor

  // wait for all workers to finish
  for (auto& worker : g_all_workers) {
    worker.join();
  }
}

void worker(int id)
{
  g_mutex.lock();
  setenv(RANK_ENV, std::to_string(id).data(), 1);
  setenv(MASTER_ADDR_ENV, std::string("127.0.0.1:" + std::to_string(MASTER_PORT)).data(), 1);
  auto workerChannel = std::make_shared<thd::DataChannelTCP>(thd::getInitConfig("env://"));  // reads all env variable
  g_mutex.unlock();

  assert(workerChannel->init());

  auto float_tensor = buildTensor<float>({1, 2, 3}, -1);
  workerChannel->broadcast(*float_tensor, 0);
  ASSERT_TENSOR_VALUE(float, *float_tensor, 4)
}


int main() {
  // start master
  std::thread master_thread(master);

  // start worker
  for (int id = 1; id <= WORKERS_NUM; ++id) {
    g_all_workers.push_back(std::thread(worker, id));
  }

  master_thread.join();
  std::cout << "OK" << std::endl;
  return 0;
}
