#include "../base/channels/DataChannelMPI.hpp"
#include "../base/tensors/THTensor.hpp"

#include <unistd.h>
#include <cassert>
#include <iostream>
#include <memory>

constexpr int WORKERS_NUM = 2;

void master(std::shared_ptr<thd::DataChannelMPI> dataChannel) {
  FloatTensor *float_tensor = new THTensor<float>();
  float_tensor->resize({1, 2, 3});
  float_tensor->fill(4);

  dataChannel->send(*float_tensor, 1);

  IntTensor *int_tensor = new THTensor<int>();
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(1000000000);

  dataChannel->broadcast(*int_tensor, 0);

  // reduce
  int_tensor->resize({1, 2, 3, 4});
  int_tensor->fill(100);
  dataChannel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, 0);
  for (int i = 0; i < int_tensor->numel(); i++) {
    assert(reinterpret_cast<int*>(int_tensor->data())[i] == (100 + 10 * WORKERS_NUM));
  }

  // allReduce
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(1000);
  dataChannel->allReduce(*int_tensor, THDReduceOp::THDReduceMAX);
  for (int i = 0; i < int_tensor->numel(); i++) {
    assert(reinterpret_cast<int*>(int_tensor->data())[i] == 1000);
  }

  delete float_tensor;
  delete int_tensor;
}

void worker(std::shared_ptr<thd::DataChannelMPI> dataChannel) {
  FloatTensor *float_tensor = new THTensor<float>();
  float_tensor->resize({1, 2, 3});

  if (dataChannel->getRank() == 1) {
    dataChannel->receive(*float_tensor, 0);

    for (int i = 0; i < float_tensor->numel(); i++) {
      assert(reinterpret_cast<float*>(float_tensor->data())[i] == 4);
    }
  }

  IntTensor *int_tensor = new THTensor<int>();
  int_tensor->resize({1, 2, 3, 4, 5});

  dataChannel->broadcast(*int_tensor, 0);

  for (int i = 0; i < int_tensor->numel(); i++) {
    assert(reinterpret_cast<int*>(int_tensor->data())[i] == 1000000000);
  }

  // reduce
  int_tensor->resize({1, 2, 3, 4});
  int_tensor->fill(10);
  dataChannel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, 0);

  // allReduce
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(1);
  dataChannel->allReduce(*int_tensor, THDReduceOp::THDReduceMAX);
  for (int i = 0; i < int_tensor->numel(); i++) {
    assert(reinterpret_cast<int*>(int_tensor->data())[i] == 1000);
  }

  delete float_tensor;
  delete int_tensor;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    execlp("mpirun", "mpirun", "-n", std::to_string(WORKERS_NUM + 1).data(), "-iface", "en0", argv[0], "1", NULL);
  }

  auto dataChannel = std::make_shared<thd::DataChannelMPI>();
  assert(dataChannel->init());

  if (dataChannel->getRank() == 0) {
    master(dataChannel);
  } else {
    worker(dataChannel);
  }

  std::cout << "OK (id: " << dataChannel->getRank() << ")" << std::endl;
  return 0;
}
