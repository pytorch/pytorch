#include "../base/channels/DataChannelMPI.hpp"
#include "../base/tensors/THTensor.hpp"

#include <iostream>
#include <cassert>
#include <memory>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

constexpr int WORKERS_NUM = 2;
constexpr int BARRIER_WAIT_TIME = 500; // milliseconds

void master(std::shared_ptr<thd::DataChannelMPI> dataChannel) {
    thd::FloatTensor *float_tensor = new thd::THTensor<float>();
  float_tensor->resize({1, 2, 3});
  float_tensor->fill(4);

  dataChannel->send(*float_tensor, 1);

  thd::IntTensor *int_tensor = new thd::THTensor<int>();
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

  // scatter
  // TODO: change number of tensosrs to WORKERS + 1
  thd::IntTensor* t1 = new thd::THTensor<int>();
  thd::IntTensor* t2 = new thd::THTensor<int>();
  thd::IntTensor* t3 = new thd::THTensor<int>();
  t1->resize({1, 2, 3, 4, 5});
  t1->fill(10);
  t2->resize({1, 2, 3, 4, 5});
  t2->fill(10);
  t3->resize({1, 2, 3, 4, 5});
  t3->fill(10);
  std::vector<thd::Tensor*> v_scatter = {t1, t2, t3};
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(-1);
  dataChannel->scatter(v_scatter, *int_tensor, 0);
  for (int i = 0; i < int_tensor->numel(); i++)
    assert(reinterpret_cast<int*>(int_tensor->data())[i] == 10);

  // gather
  t1->resize({1, 2, 3, 4, 5});
  t1->fill(-1);
  t2->resize({1, 2, 3, 4, 5});
  t2->fill(-1);
  t3->resize({1, 2, 3, 4, 5});
  t3->fill(-1);
  std::vector<thd::Tensor*> v_gather = {t1, t2, t3};
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(10);
  dataChannel->gather(v_gather, *int_tensor, 0);
  for (auto tensor : v_gather) {
    for (int i = 0; i < tensor->numel(); i++)
      assert(reinterpret_cast<int*>(tensor->data())[i] == 10);
  }

  // allGather
  t1->resize({1, 2, 3, 4, 5});
  t1->fill(-1);
  t2->resize({1, 2, 3, 4, 5});
  t2->fill(-1);
  t3->resize({1, 2, 3, 4, 5});
  t3->fill(-1);
  std::vector<thd::Tensor*> v_allGather = {t1, t2, t3};
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(10);
  dataChannel->allGather(v_allGather, *int_tensor);
  for (auto tensor : v_allGather) {
    for (int i = 0; i < tensor->numel(); i++)
      assert(reinterpret_cast<int*>(tensor->data())[i] == 10);
  }

  // barrier
  for (int i = 0; i < dataChannel->getNumProcesses(); ++i) {
    if (dataChannel->getRank() == i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_WAIT_TIME));
      dataChannel->barrier();
    } else {
      auto start = std::chrono::system_clock::now();
      dataChannel->barrier();
      auto end = std::chrono::system_clock::now();

      std::chrono::duration<double> elapsed = end - start;
      std::chrono::milliseconds ms_elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

      assert(ms_elapsed.count() >= (BARRIER_WAIT_TIME / 2));
    }
  }

  // groups
  THDGroup group = dataChannel->newGroup({1, 2});
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(1000);

  /*
   * We call this functions to check if our data does not change and if it will not
   * affect any computations when process outside group join any of this functions.
   *
   * Processes which do not belong to group do not have to call those methods!
   */
  dataChannel->allReduce(*int_tensor, THDReduceOp::THDReduceSUM, group);
  auto tensor_data_ptr = reinterpret_cast<int*>(int_tensor->data());
  for (int i = 0; i < int_tensor->numel(); i++)
    assert(tensor_data_ptr[i] == 1000);

  dataChannel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, 1, group);
  for (int i = 0; i < int_tensor->numel(); i++)
    assert(tensor_data_ptr[i] == 1000);

  dataChannel->broadcast(*int_tensor, 1, group);
  for (int i = 0; i < int_tensor->numel(); i++)
    assert(tensor_data_ptr[i] == 1000);

  delete t1; delete t2; delete t3;
  delete float_tensor;
  delete int_tensor;
}

void worker(std::shared_ptr<thd::DataChannelMPI> dataChannel) {
  thd::FloatTensor *float_tensor = new thd::THTensor<float>();
  float_tensor->resize({1, 2, 3});

  if (dataChannel->getRank() == 1) {
    dataChannel->receive(*float_tensor, 0);

    for (int i = 0; i < float_tensor->numel(); i++) {
      assert(reinterpret_cast<float*>(float_tensor->data())[i] == 4);
    }
  }

  thd::IntTensor *int_tensor = new thd::THTensor<int>();
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

  // scatter
  std::vector<thd::Tensor*> v;
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(-1);
  dataChannel->scatter(v, *int_tensor, 0);
  for (int i = 0; i < int_tensor->numel(); i++)
    assert(reinterpret_cast<int*>(int_tensor->data())[i] == 10);

  // gather
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(10);
  dataChannel->gather(v, *int_tensor, 0);

  // allGather
  thd::IntTensor* t1 = new thd::THTensor<int>();
  thd::IntTensor* t2 = new thd::THTensor<int>();
  thd::IntTensor* t3 = new thd::THTensor<int>();
  t1->resize({1, 2, 3, 4, 5});
  t1->fill(-1);
  t2->resize({1, 2, 3, 4, 5});
  t2->fill(-1);
  t3->resize({1, 2, 3, 4, 5});
  t3->fill(-1);
  std::vector<thd::Tensor*> v_allGather = {t1, t2, t3};
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(10);
  dataChannel->allGather(v_allGather, *int_tensor);
  for (auto tensor : v_allGather) {
    for (int i = 0; i < tensor->numel(); i++)
      assert(reinterpret_cast<int*>(tensor->data())[i] == 10);
  }

  // barrier
  for (int i = 0; i < dataChannel->getNumProcesses(); ++i) {
    if (dataChannel->getRank() == i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_WAIT_TIME));
      dataChannel->barrier();
    } else {
      auto start = std::chrono::system_clock::now();
      dataChannel->barrier();
      auto end = std::chrono::system_clock::now();

      std::chrono::duration<double> elapsed = end - start;
      std::chrono::milliseconds ms_elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

      assert(ms_elapsed.count() >= (BARRIER_WAIT_TIME / 2));
    }
  }

  // group
  THDGroup group = dataChannel->newGroup({1, 2});
  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(10);
  dataChannel->allReduce(*int_tensor, THDReduceOp::THDReduceSUM, group);
  if (dataChannel->getRank() == 1 || dataChannel->getRank() == 2) {
    for (int i = 0; i < int_tensor->numel(); i++) {
      assert(reinterpret_cast<int*>(int_tensor->data())[i] == (10 * WORKERS_NUM));
    }

    // rank 0 (master) is not part of group, we cannot perform reduce to it
    try {
      dataChannel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, 0, group);
      assert(false);
    } catch (const std::logic_error& e) {}
  }

  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(10);
  dataChannel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, 1, group);
  for (int i = 0; i < int_tensor->numel(); i++) {
    assert(reinterpret_cast<int*>(int_tensor->data())[i] == (dataChannel->getRank() == 1 ? 20 : 10));
  }

  int_tensor->resize({1, 2, 3, 4, 5});
  int_tensor->fill(10);
  if (dataChannel->getRank() == 1)
    int_tensor->fill(2000);

  dataChannel->broadcast(*int_tensor, 1, group);
  if (dataChannel->getRank() == 1 || dataChannel->getRank() == 2) {
    for (int i = 0; i < int_tensor->numel(); i++) {
      assert(reinterpret_cast<int*>(int_tensor->data())[i] == 2000);
    }
  }

  delete float_tensor;
  delete int_tensor;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    execlp("mpirun", "mpirun", "-n", std::to_string(WORKERS_NUM + 1).data(), argv[0], "1", NULL);
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
