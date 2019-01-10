#include "../master_worker/common/CommandChannel.hpp"
#include "TestUtils.hpp"

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

using namespace thd;

std::vector<std::thread> g_all_workers;
std::mutex g_mutex;
std::unique_ptr<Barrier> g_barrier;

void init_worker(const int& rank, const std::string& master_addr) {
  g_mutex.lock();
  setenv(RANK_ENV, std::to_string(rank).data(), 1);
  setenv(MASTER_ADDR_ENV, master_addr.data(), 1);
  auto channel = std::make_shared<thd::WorkerCommandChannel>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(channel->init());

  auto msg = channel->recvMessage();
  std::string expected = std::string("hello to worker ") +
      std::to_string(rank) + " from master";
  fprintf(stderr, "Worker %d: received '%.*s'\n", rank,
      (int)msg.get()->bytes().length(), msg.get()->bytes().data());
  assert(expected.compare(msg.get()->bytes().to_string()) == 0);

  /*
   * We need to wait until master will do all receiving and sending. This
   * is because when worker is destroyed it closes all sockets what results in
   * triggering `poll` function in master's error_handler and throwing exception.
   */
  g_barrier->wait();
}

void init_master(int world_size, const std::string& master_port) {
  g_mutex.lock();
  setenv(WORLD_SIZE_ENV, std::to_string(world_size).data(), 1);
  setenv(RANK_ENV, "0", 1);
  setenv(MASTER_PORT_ENV, master_port.data(), 1);
  auto channel = std::make_shared<thd::MasterCommandChannel>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(channel->init());

  for (int worker_rank = 1; worker_rank < world_size; ++worker_rank) {
    rpc::ByteArray arr;
    arr.append("hello to worker ", sizeof("hello to worker ") - 1);
    arr.append(std::to_string(worker_rank).c_str(), std::to_string(worker_rank).size());
    arr.append(" from master", sizeof(" from master") - 1);

    fprintf(stderr, "master: about to send a message to worker %d\n", worker_rank);
    auto rpc_msg = std::unique_ptr<rpc::RPCMessage>(new rpc::RPCMessage(arr));
    channel->sendMessage(std::move(rpc_msg), worker_rank);
  }

  g_barrier->wait();

  // wait for all workers to finish
  for (auto& worker : g_all_workers) {
    worker.join();
  }
}

void run_test_case(const std::string& name, int world_size,
                   const std::string& master_addr, const std::string& master_port) {
  g_barrier.reset(new Barrier(world_size));
  for (int rank = 1; rank < world_size; ++rank) {
    g_all_workers.push_back(
      std::thread(init_worker, rank, master_addr + ":" + master_port)
    );
  }

  std::thread master_thread(init_master, world_size, master_port);
  master_thread.join();
  g_all_workers.clear();

  fprintf(stderr, "\nPassed %s:\n"
      "world size =\t\t%d\n"
      "master address =\t%s\n"
      "master port =\t\t%s\n"
      "----------------------------------------------------\n\n",
      name.c_str(), world_size, master_addr.c_str(), master_port.c_str());
}

int main() {
  int world_size;
  std::string master_addr;
  std::string master_port;
  std::string test_name;

  try {
    test_name = "Master test";
    world_size = 1;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port);

    test_name = "Basic test";
    world_size = 4;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port);

    test_name = "Many workers test";
    world_size = 12;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port);

    test_name = "IPv6 test";
    world_size = 12;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port);

    test_name = "Hostname resolution test";
    world_size = 12;
    master_addr = "localhost";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "test for world size = " + std::to_string(world_size) +
        ", master address = " + master_addr + ", master port = " +
        master_port + " failed because of: `" + e.what() + "`");
  }

  fprintf(stdout, "OK\n");
  return 0;
}
