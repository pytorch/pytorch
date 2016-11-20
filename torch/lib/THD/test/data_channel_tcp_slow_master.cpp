#include "../base/channels/DataChannelTCP.hpp"
#include "../base/tensors/THTensor.hpp"

#include <cassert>
#include <iostream>
#include <thread>

constexpr int WORKERS_NUM = 2;
constexpr int MASTER_PORT = 45679;

std::vector<std::thread> g_all_workers;
std::mutex g_mutex;

void master()
{
  g_mutex.lock();
  setenv("WORLD_SIZE", std::to_string((WORKERS_NUM + 1)).data(), 1);
  setenv("RANK", "0", 1);
  setenv("MASTER_PORT", std::to_string(MASTER_PORT).data(), 1);
  auto masterChannel = std::make_shared<thd::DataChannelTCP>(); // reads all env variable
  g_mutex.unlock();

  std::this_thread::sleep_for(std::chrono::seconds(5));

  assert(masterChannel->init());

  FloatTensor *float_tensor = new THTensor<float>();
  float_tensor->resize({1, 2, 3});
  float_tensor->fill(4);

  // send good tensor
  masterChannel->broadcast(*float_tensor, 0);


  // wait for all workers to finish
  for (auto& worker : g_all_workers) {
    worker.join();
  }

  delete float_tensor;
}

void worker(int id)
{
  g_mutex.lock();
  setenv("RANK", std::to_string(id).data(), 1);
  setenv("MASTER_ADDR", std::string("127.0.0.1:" + std::to_string(MASTER_PORT)).data(), 1);
  auto workerChannel = std::make_shared<thd::DataChannelTCP>();  // reads all env variable
  g_mutex.unlock();

  /*
   * Wait for other processes to initialize.
   * It is to avoid race in acquiring socket and port for listening (in init function).
   */
  std::this_thread::sleep_for(std::chrono::milliseconds(100 * workerChannel->getRank()));
  assert(workerChannel->init());

  FloatTensor *float_tensor = new THTensor<float>();
  float_tensor->resize({1, 2, 3});

  workerChannel->broadcast(*float_tensor, 0);
  for (int i = 0; i < float_tensor->numel(); i++) {
    assert(((float*)float_tensor->data())[i] == 4);
  }

  delete float_tensor;
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
