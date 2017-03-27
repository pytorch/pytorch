#include "../base/channels/DataChannelMPI.hpp"

#include <unistd.h>
#include <cassert>
#include <iostream>
#include <memory>

constexpr int WORKERS_NUM = 2;

int main(int argc, char **argv) {
  if (argc == 1) {
    execlp("mpirun", "mpirun", "-n", std::to_string(WORKERS_NUM + 1).data(), argv[0], "1", NULL);
  }

  auto dataChannel = std::make_shared<thd::DataChannelMPI>();
  assert(dataChannel->init());
  assert(dataChannel->getNumProcesses() == (WORKERS_NUM + 1));
  std::cout << "OK (id: " << dataChannel->getRank() << ")" << std::endl;
  return 0;
}
