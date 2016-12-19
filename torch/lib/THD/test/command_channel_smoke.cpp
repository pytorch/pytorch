#include "../master_worker/common/CommandChannel.hpp"
#include "../base/ChannelEnvVars.hpp"

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <string>
#include <system_error>
#include <stdlib.h>
#include <unistd.h>

using namespace thd;

void overwrite_env(const std::string& name, const std::string& value) {
  int err = setenv(name.c_str(), value.c_str(), 1);
  if (err == -1) {
    throw std::system_error(errno, std::system_category(),
        "unable to set an environmental variable " + name + " to " + value);
  }
}

void init_worker(const int& rank, const std::string& master_addr) {
  overwrite_env(RANK_ENV, std::to_string(rank));
  overwrite_env(MASTER_ADDR_ENV, master_addr);

  fprintf(stderr, "worker %d: about to construct a worker\n", rank);
  WorkerCommandChannel channel(rank);
  fprintf(stderr, "worker %d: constructed\n", rank);

  // Send.
  rpc::ByteArray arr;
  arr.append("hello to master from worker ",
      sizeof("hello to master from worker ") - 1);
  arr.append(std::to_string(rank).c_str(), std::to_string(rank).size());

  fprintf(stderr, "worker %d: about to send a message\n", rank);

  channel.sendMessage(std::unique_ptr<rpc::RPCMessage>(
        new rpc::RPCMessage(std::move(arr))));

  fprintf(stderr, "worker %d: sent message\n", rank);

  // Recieve.
  auto msg = channel.recvMessage();
  std::string expected = std::string("hello to worker ") +
      std::to_string(rank) + " from master";
  fprintf(stderr, "Worker %d: received '%.*s'\n", rank,
      (int)msg.get()->bytes().length(), msg.get()->bytes().data());
  assert(expected.compare(msg.get()->bytes().to_string()) == 0);
}

void init_master(const int& world_size, const std::string& master_port) {
  overwrite_env(RANK_ENV, std::to_string(0));
  overwrite_env(WORLD_SIZE_ENV, std::to_string(world_size));
  overwrite_env(MASTER_PORT_ENV, master_port);

  MasterCommandChannel channel;

  for (int wi = 1; wi < world_size; ++wi) {
    rpc::ByteArray arr;
    arr.append("hello to worker ", sizeof("hello to worker ") - 1);
    arr.append(std::to_string(wi).c_str(), std::to_string(wi).size());
    arr.append(" from master", sizeof(" from master") - 1);

    fprintf(stderr, "master: about to send a message to worker %d\n", wi);
    channel.sendMessage(
        std::unique_ptr<rpc::RPCMessage>(new rpc::RPCMessage(arr)),
        wi
    );
  }

  for (int wi = 1; wi < world_size; ++wi) {
    std::unique_ptr<rpc::RPCMessage> msg;
    msg = channel.recvMessage(wi);
    std::string expected = std::string("hello to master from worker ") +
      std::to_string(wi);
    fprintf(stderr, "Master: received '%.*s' from worker %d\n",
        (int)msg.get()->bytes().length(), msg.get()->bytes().data(),
        wi);
    assert(expected.compare(msg.get()->bytes().to_string()) == 0);
  }
}

void run_test_case(
    const std::string& name,
    const int& world_size,
    const std::string& master_addr,
    const std::string& master_port,
    int& rank

) {
  rank = 0;
  for (int worker_rank = 1; worker_rank < world_size; ++worker_rank) {
    pid_t pid = fork();
    if (pid == 0) {
      rank = worker_rank;
      init_worker(worker_rank, master_addr + ":" + master_port);
      std::exit(0);
    } else if (pid < 0) {
      throw std::system_error(errno, std::system_category(), "failed to fork");
    }
  }

  init_master(world_size, master_port);
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
  int rank;

  try {
    test_name = "Master test";
    world_size = 1;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port, rank);

    test_name = "Basic test";
    world_size = 4;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port, rank);

    test_name = "Many workers test";
    world_size = 12;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port, rank);

    test_name = "IPv6 test";
    world_size = 12;
    master_addr = "127.0.0.1";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port, rank);

    test_name = "Hostname resolution test";
    world_size = 12;
    master_addr = "localhost";
    master_port = "55555";
    run_test_case(test_name, world_size, master_addr, master_port, rank);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string() + "rank = " + std::to_string(rank) +
        "; test for world size = " + std::to_string(world_size) +
        ", master address = " + master_addr + ", master port = " +
        master_port + " failed because of: " + e.what());
  }

  fprintf(stdout, "OK\n");

  return 0;
}
