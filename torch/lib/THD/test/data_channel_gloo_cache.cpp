#include "../base/data_channels/DataChannelGloo.hpp"
#include "../base/ChannelEnvVars.hpp"
#include "TestUtils.hpp"

#include <THPP/tensors/THTensor.hpp>

#include <unistd.h>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


constexpr std::array<int, 1> WORKERS_NUM = {10};
constexpr int MASTER_PORT = 45678;

std::vector<std::thread> g_all_workers;
std::mutex g_mutex;

void test(std::shared_ptr<thd::DataChannel> data_channel) {
  for (std::size_t dest = 0; dest < data_channel->getNumProcesses(); ++dest) {
    if (data_channel->getRank() == dest) {
      auto float_tensor = buildTensor<float>({1, 2, 3, 4, 5}, 10.123);
      data_channel->broadcast(*float_tensor, dest);
    } else {
      auto float_tensor = buildTensor<float>({1, 2, 3, 4, 5}, -1.0);
      data_channel->broadcast(*float_tensor, dest);
      ASSERT_TENSOR_VALUE(float, *float_tensor, 10.123)
    }
  }
}

void run_all_tests(std::shared_ptr<thd::DataChannel> data_channel, int workers) {
  // NOTE: without properly working GlooCache this test would create
  // about (1000 * WORKERS ^ 3) connections what is over 'normal' system configuration
  for (std::size_t i = 0; i < 1000; ++i) {
    test(data_channel);
  }
}


void init_gloo_master(int workers) {
  g_mutex.lock();
  setenv(thd::WORLD_SIZE_ENV, std::to_string((workers + 1)).data(), 1);
  setenv(thd::RANK_ENV, "0", 1);
  setenv(thd::MASTER_PORT_ENV, std::to_string(MASTER_PORT).data(), 1);
  auto masterChannel = std::make_shared<thd::DataChannelGloo>(getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(masterChannel->init());
  run_all_tests(masterChannel, workers);
}

void init_gloo_worker(unsigned int id, int workers) {
  g_mutex.lock();
  setenv(thd::RANK_ENV, std::to_string(id).data(), 1);
  setenv(thd::MASTER_ADDR_ENV, std::string("127.0.0.1:" + std::to_string(MASTER_PORT)).data(), 1);
  auto worker_channel = std::make_shared<thd::DataChannelGloo>(getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(worker_channel->init());
  run_all_tests(worker_channel, workers);
}


int main(void)
{
  for (auto workers : WORKERS_NUM) {
    std::cout << "Gloo (workers: " << workers << "):" << std::endl;
    // start gloo master
    std::thread gloo_master_thread(init_gloo_master, workers);

    // start gloo worker
    for (int id = 1; id <= workers; ++id) {
      g_all_workers.push_back(std::thread(init_gloo_worker, id, workers));
    }

    // wait for all workers to finish
    for (auto& worker : g_all_workers) {
      worker.join();
    }

    gloo_master_thread.join();
    g_all_workers.clear();

    std::cout << "Gloo - OK" << std::endl;
  }
}
