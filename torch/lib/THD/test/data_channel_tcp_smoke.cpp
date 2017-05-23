#include "../base/data_channels/DataChannelTCP.hpp"
#include "../base/ChannelEnvVars.hpp"

#include <THPP/tensors/THTensor.hpp>

#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

constexpr int WORKERS_NUM = 2;
constexpr int MASTER_PORT = 45678;

std::vector<std::thread> g_all_workers;
std::mutex g_mutex;

void master()
{
  g_mutex.lock();
  setenv(thd::WORLD_SIZE_ENV, std::to_string((WORKERS_NUM + 1)).data(), 1);
  setenv(thd::RANK_ENV, "0", 1);
  setenv(thd::MASTER_PORT_ENV, std::to_string(MASTER_PORT).data(), 1);
  auto masterChannel = std::make_shared<thd::DataChannelTCP>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(masterChannel->init());
  assert(masterChannel->getRank() == 0);
  assert(masterChannel->getNumProcesses() == WORKERS_NUM + 1);

  // wait for all workers to finish
  for (auto& worker : g_all_workers) {
    worker.join();
  }
}

void worker(int id)
{
  g_mutex.lock();
  setenv(thd::RANK_ENV, std::to_string(id).data(), 1);
  setenv(thd::MASTER_ADDR_ENV, std::string("127.0.0.1:" + std::to_string(MASTER_PORT)).data(), 1);
  auto workerChannel = std::make_shared<thd::DataChannelTCP>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(workerChannel->init());
  assert(workerChannel->getRank() == id);
  assert(workerChannel->getNumProcesses() == WORKERS_NUM + 1);
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
