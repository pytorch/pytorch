#include "../_THD.h"

#include <cerrno>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <unistd.h>

constexpr int kWORKERS_NUM = 3;
constexpr int kWORLD_SIZE = kWORKERS_NUM + 1;
constexpr char kMASTER_IP_ADDR[] = "127.0.0.1";
constexpr char kMASTER_PORT[] = "5555";

using namespace thd;

void overwrite_env(const std::string& name, const std::string& value) {
  int err = setenv(name.c_str(), value.c_str(), 1);
  if (err == -1) {
    throw std::system_error(errno,
                            std::system_category(),
                            "unable to set an environmental variable " + name +
                            " to " + value);
  }
}

rpc::ByteArray to_bytes(std::string s) {
  return rpc::ByteArray::fromData(s.c_str(), s.size());
}

void init_worker(int rank) {
  overwrite_env(RANK_ENV, std::to_string(rank));
  assert(std::to_string(rank) == std::getenv(RANK_ENV));

  overwrite_env(MASTER_ADDR_ENV,
                std::string(kMASTER_IP_ADDR) + ":" + kMASTER_PORT);

  WorkerCommandChannel channel(rank);

  rpc::ByteArray arr;
  arr.append("hello to master from worker",
             sizeof("hello to master from worker") - 1);
  arr.append(std::to_string(rank).c_str(), std::to_string(rank).size());

  std::cerr << "worker #" << rank << ": " << "about to send a message" <<
      std::endl;

  channel.sendMessage(std::unique_ptr<rpc::RPCMessage>(
        new rpc::RPCMessage(std::move(arr))));

  std::cerr << "worker #" << rank << ": " << "mesage sent" << std::endl;

  std::unique_ptr<rpc::RPCMessage> reply;
  do {
    reply = channel.recvMessage();
  } while (reply == nullptr);
  std::cerr << "worker #" << rank << ": received: " << reply.get()->data() <<
      std::endl;
}

void init_master() {
  overwrite_env(RANK_ENV, std::to_string(0));
  assert(std::to_string(0) == std::getenv(RANK_ENV));

  overwrite_env(WORLD_SIZE_ENV, std::to_string(kWORLD_SIZE));
  assert(std::to_string(kWORLD_SIZE) == std::getenv(WORLD_SIZE_ENV));

  overwrite_env(MASTER_PORT_ENV, kMASTER_PORT);

  MasterCommandChannel channel;

  for (int wi = 1; wi <= kWORKERS_NUM; ++wi) {
    rpc::ByteArray arr;
    arr.append("hello to worker# ", sizeof("hello to worker# ") - 1);
    arr.append(std::to_string(wi).c_str(), std::to_string(wi).size());
    arr.append(" from master", sizeof(" from master") - 1);

  std::cerr << "master: about to send a message to worker #" << wi << std::endl;
    channel.sendMessage(std::unique_ptr<rpc::RPCMessage>(new
                                                         rpc::RPCMessage(arr)),
                        wi);
  }

  for (int wi = 1; wi <= kWORKERS_NUM; ++wi) {
    std::unique_ptr<rpc::RPCMessage> msg;
    do {
      msg = channel.recvMessage(wi);
    } while (msg == nullptr);
    std::cerr << "master: received: " << msg.get()->data() << std::endl;
  }
}

int main() {
  for (int worker_rank = 1; worker_rank <= kWORKERS_NUM; ++worker_rank) {
    std::cout.flush();
    std::cerr.flush();
    pid_t pid = fork();
    if (pid == 0) {
      init_worker(worker_rank);
      return 0;
    } else if (pid < 0) {
      throw std::system_error(errno, std::system_category(), "failed to fork");
    }
  }

  init_master();
  std::cout << "OK" << std::endl;

  return 0;
}
