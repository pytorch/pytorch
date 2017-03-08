#include "../master_worker/common/CommandChannel.hpp"
#include "../base/ChannelEnvVars.hpp"

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <string>
#include <system_error>
#include <thread>

using namespace thd;

std::vector<std::thread> g_all_workers;
std::mutex g_mutex;

void init_worker(const int& rank, const std::string& master_addr) {
  g_mutex.lock();
  setenv("RANK", std::to_string(rank).data(), 1);
  setenv("MASTER_ADDR", master_addr.data(), 1);
  auto channel = std::make_shared<thd::WorkerCommandChannel>(); // reads all env variable
  g_mutex.unlock();

  assert(channel->init());

  // Send.
  rpc::ByteArray arr;
  arr.append("hello to master from worker ",
      sizeof("hello to master from worker ") - 1);
  arr.append(std::to_string(rank).c_str(), std::to_string(rank).size());

  fprintf(stderr, "worker %d: about to send a message\n", rank);

  auto rpc_msg = std::unique_ptr<rpc::RPCMessage>(new rpc::RPCMessage(std::move(arr)));
  channel->sendMessage(std::move(rpc_msg));

  fprintf(stderr, "worker %d: sent message\n", rank);

  // Recieve.
  auto msg = channel->recvMessage();
  std::string expected = std::string("hello to worker ") +
      std::to_string(rank) + " from master";
  fprintf(stderr, "Worker %d: received '%.*s'\n", rank,
      (int)msg.get()->bytes().length(), msg.get()->bytes().data());
  assert(expected.compare(msg.get()->bytes().to_string()) == 0);
}

void init_master(int world_size, const std::string& master_port) {
  g_mutex.lock();
  setenv("WORLD_SIZE", std::to_string(world_size).data(), 1);
  setenv("RANK", "0", 1);
  setenv("MASTER_PORT", master_port.data(), 1);
  auto channel = std::make_shared<thd::MasterCommandChannel>(); // reads all env variable
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

  for (int worker_rank = 1; worker_rank < world_size; ++worker_rank) {
    std::unique_ptr<rpc::RPCMessage> msg;
    fprintf(stderr, "master: about to recv a message from worker %d\n", worker_rank);
    msg = channel->recvMessage(worker_rank);
    std::string expected = std::string("hello to master from worker ") +
      std::to_string(worker_rank);
    fprintf(stderr, "Master: received '%.*s' from worker %d\n",
        (int)msg.get()->bytes().length(), msg.get()->bytes().data(),
        worker_rank);
    assert(expected.compare(msg.get()->bytes().to_string()) == 0);
  }

  // wait for all workers to finish
  for (auto& worker : g_all_workers) {
    worker.join();
  }
}

void run_test_case(const std::string& name, int world_size,
                   const std::string& master_addr, const std::string& master_port) {
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
