#ifdef WITH_GLOO
#include "../base/data_channels/DataChannelGloo.hpp"
#endif // WITH_GLOO
#ifdef WITH_MPI
#include "../base/data_channels/DataChannelMPI.hpp"
#endif // WITH_MPI
#include "../base/data_channels/DataChannelTCP.hpp"
#include "../base/ChannelEnvVars.hpp"
#include "TestUtils.hpp"

#include <THPP/tensors/THTensor.hpp>

#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <array>

constexpr std::array<int, 4> WORKERS_NUM = {2, 4, 7, 13};
constexpr int MASTER_PORT = 45678;
constexpr int BARRIER_WAIT_TIME = 200; // milliseconds

std::vector<std::thread> g_all_workers;
std::mutex g_mutex;
std::string g_data_channel_type;
std::unique_ptr<Barrier> g_barrier;


void test_send_recv_tensor(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support send/recv
  }

  if (data_channel->getRank() == 0) {
    auto float_tensor = buildTensor<float>({1, 2, 3}, 4.2);
    data_channel->send(*float_tensor, 1);
  } else if (data_channel->getRank() == 1) {
    auto float_tensor = buildTensor<float>({1, 2, 3}, -1.0);
    data_channel->receive(*float_tensor, 0);
    ASSERT_TENSOR_VALUE(float, *float_tensor, 4.2);
  }
}

void test_send_recv_tensor_any_source(std::shared_ptr<thd::DataChannel> data_channel,
                                      int workers) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support send/recv from any source
  }

  if (data_channel->getRank() == 0) {
    std::set<int> ranks;
    for (int i = 0; i < workers; i++) {
      auto int_tensor = buildTensor<int>({1, 2, 3}, -1);
      data_channel->receive(*int_tensor);
      ranks.insert(static_cast<int*>(int_tensor->data())[0]);
    }

    assert(ranks.size() == workers);
  } else {
    auto int_tensor = buildTensor<int>({1, 2, 3}, data_channel->getRank());
    data_channel->send(*int_tensor, 0);
  }
}

void test_send_recv_scalar(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support send/recv
  }

  if (data_channel->getRank() == 0) {
    thd::ScalarWrapper<int> scalar((int)1232);
    data_channel->send(scalar, 1);
  } else if (data_channel->getRank() == 1) {
    thd::ScalarWrapper<int> scalar((int)-1);
    data_channel->receive(scalar, 0);
    assert(scalar.value() == 1232);
  }
}

void test_broadcast(std::shared_ptr<thd::DataChannel> data_channel) {
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

void _test_reduce_helper(std::shared_ptr<thd::DataChannel> data_channel,
                         THDReduceOp op_type, long init_value, long expected_value) {
  if (data_channel->getRank() == 0) {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, init_value);
    data_channel->reduce(*int_tensor, op_type, 0);
    ASSERT_TENSOR_VALUE(int, *int_tensor, expected_value)
  } else {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, data_channel->getRank());
    data_channel->reduce(*int_tensor, op_type, 0);
  }
}

void test_reduce(std::shared_ptr<thd::DataChannel> data_channel, int workers) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support reduce
  }

  _test_reduce_helper(data_channel, THDReduceOp::THDReduceSUM,
                      2, 2 + (workers * (workers + 1) / 2));
  _test_reduce_helper(data_channel, THDReduceOp::THDReducePRODUCT,
                      2, 2 * factorial(workers));
  _test_reduce_helper(data_channel, THDReduceOp::THDReduceMIN, 10010, 1);
  _test_reduce_helper(data_channel, THDReduceOp::THDReduceMAX,
                      -1, data_channel->getNumProcesses() - 1);
}

void _test_allReduce_helper(std::shared_ptr<thd::DataChannel> data_channel,
                            THDReduceOp op_type, long init_value, long expected_value) {
  if (data_channel->getRank() == 0) {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5, 6, 7, 100}, init_value);
    data_channel->allReduce(*int_tensor, op_type, 0);
    ASSERT_TENSOR_VALUE(int, *int_tensor, expected_value)
  } else {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5, 6, 7, 100}, data_channel->getRank());
    data_channel->allReduce(*int_tensor, op_type, 0);
    ASSERT_TENSOR_VALUE(int, *int_tensor, expected_value)
  }
}

void test_allReduce(std::shared_ptr<thd::DataChannel> data_channel, int workers) {
  _test_allReduce_helper(data_channel, THDReduceOp::THDReduceSUM,
                         2, 2 + (workers * (workers + 1) / 2));
  _test_allReduce_helper(data_channel, THDReduceOp::THDReducePRODUCT,
                         2, 2 * factorial(workers));
  _test_allReduce_helper(data_channel, THDReduceOp::THDReduceMIN, 10010, 1);
  _test_allReduce_helper(data_channel, THDReduceOp::THDReduceMAX,
                         -1, data_channel->getNumProcesses() - 1);
}

void test_scatter(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support scatter
  }

  std::vector<std::shared_ptr<thpp::IntTensor>> tensors;
  std::vector<thpp::Tensor*> raw_tensors;
  if (data_channel->getRank() == 0) {
    for (std::size_t i = 0; i < data_channel->getNumProcesses(); ++i) {
      tensors.push_back(buildTensor<int>({1, 2, 3, 4, 5}, i));
      raw_tensors.push_back(tensors.back().get());
    }
  }

  auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, -1);
  data_channel->scatter(raw_tensors, *int_tensor, 0);
  ASSERT_TENSOR_VALUE(int, *int_tensor, data_channel->getRank())
}

void test_gather(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support gather
  }

  std::vector<std::shared_ptr<thpp::IntTensor>> tensors;
  std::vector<thpp::Tensor*> raw_tensors;
  auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, data_channel->getRank());
  if (data_channel->getRank() == 0) {
    for (std::size_t i = 0; i < data_channel->getNumProcesses(); ++i) {
      tensors.push_back(buildTensor<int>({1, 2, 3, 4, 5}, -1));
      raw_tensors.push_back(tensors.back().get());
    }

    data_channel->gather(raw_tensors, *int_tensor, 0);
    for (std::size_t i = 0; i < tensors.size(); ++i)
      ASSERT_TENSOR_VALUE(int, *(tensors[i]), i)
  } else {
    data_channel->gather(raw_tensors, *int_tensor, 0);
  }
}

void test_allGather(std::shared_ptr<thd::DataChannel> data_channel) {
  std::vector<std::shared_ptr<thpp::IntTensor>> tensors;
  std::vector<thpp::Tensor*> raw_tensors;
  auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, data_channel->getRank());
  for (std::size_t i = 0; i < data_channel->getNumProcesses(); ++i) {
    tensors.push_back(buildTensor<int>({1, 2, 3, 4, 5}, -1));
    raw_tensors.push_back(tensors.back().get());
  }

  data_channel->allGather(raw_tensors, *int_tensor, 0);
  for (std::size_t i = 0; i < tensors.size(); ++i)
    ASSERT_TENSOR_VALUE(int, *(tensors[i]), i)
}

void test_barrier(std::shared_ptr<thd::DataChannel> data_channel) {
  for (int i = 0; i < data_channel->getNumProcesses(); ++i) {
    if (data_channel->getRank() == i) {
      long time_after_barrier = nowInMilliseconds() + BARRIER_WAIT_TIME;
      auto time_tensor = buildTensor<long>({1}, time_after_barrier);
      data_channel->broadcast(*time_tensor, i);
      std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_WAIT_TIME + 10));
      data_channel->barrier();
    } else {
      auto time_tensor = buildTensor<long>({1}, -1);
      data_channel->broadcast(*time_tensor, i); // get expected time after barrier
      data_channel->barrier();
      assert(nowInMilliseconds() >= reinterpret_cast<long*>(time_tensor->data())[0]);
    }
  }
}

void test_isend(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support isend
  }

  if (data_channel->getRank() == 0) {
    std::vector<std::shared_ptr<thd::DataChannel::Request>> requests;
    for (std::size_t i = 1; i < data_channel->getNumProcesses(); ++i) {
      auto tensor = buildTensor<int>({1, 2, 3, 4, 5}, i);
      requests.push_back(std::shared_ptr<thd::DataChannel::Request>(
        data_channel->isend(*tensor, i)
      ));
    }

    for (auto request : requests) {
      request->wait();
      assert(request->isCompleted());
    }
  } else {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, -1);
    data_channel->receive(*int_tensor, 0);
    ASSERT_TENSOR_VALUE(int, *int_tensor, data_channel->getRank())
  }
}

void test_irecv(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support irecv
  }

  if (data_channel->getRank() == 0) {
    std::vector<std::shared_ptr<thd::DataChannel::Request>> requests;
    std::vector<std::shared_ptr<thpp::IntTensor>> tensors;
    for (std::size_t i = 1; i < data_channel->getNumProcesses(); ++i) {
      tensors.push_back(buildTensor<int>({1, 2, 3, 4, 5}, -1));
      requests.push_back(std::shared_ptr<thd::DataChannel::Request>(
        data_channel->ireceive(*tensors.back(), i)
      ));
    }

    for (std::size_t i = 0; i < requests.size(); ++i) {
      requests.at(i)->wait();
      assert(requests.at(i)->isCompleted());
      ASSERT_TENSOR_VALUE(int, *tensors.at(i), i + 1)
    }
  } else {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, data_channel->getRank());
    data_channel->send(*int_tensor, 0);
  }
}


void test_interlaces(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support isend, irecv, send, recv
  }

  if (data_channel->getRank() == 0) {
    std::vector<std::shared_ptr<thd::DataChannel::Request>> requests;
    for (std::size_t i = 1; i < data_channel->getNumProcesses(); ++i) {
      auto tensor = buildTensor<int>({1, 2, 3, 4, 5}, 10);
      requests.push_back(std::shared_ptr<thd::DataChannel::Request>(
        data_channel->isend(*tensor, i)
      ));
    }

    data_channel->barrier();

    for (std::size_t i = 1; i < data_channel->getNumProcesses(); ++i) {
      auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, 20);
      data_channel->send(*int_tensor, i);
    }
  } else {
    auto int_tensor1 = buildTensor<int>({1, 2, 3, 4, 5}, -1);
    auto request = std::shared_ptr<thd::DataChannel::Request>(
      data_channel->ireceive(*int_tensor1, 0)
    );

    data_channel->barrier();

    auto int_tensor2 = buildTensor<int>({1, 2, 3, 4, 5}, -1);
    data_channel->receive(*int_tensor2, 0);
    request->wait();

    ASSERT_TENSOR_VALUE(int, *int_tensor1, 10)
    ASSERT_TENSOR_VALUE(int, *int_tensor2, 20)
  }
}

/*
 * In group tests we call same functions in processes which do not belong to those
 * groups to check if it will not affect any computations.
 *
 * Processes which do not belong to group do not have to call those methods!
 */

////////////
// GROUPS //
////////////

void test_broadcast_group(std::shared_ptr<thd::DataChannel> data_channel,
                          THDGroup group, std::vector<thd::rank_type> group_ranks) {
  if (contains(group_ranks, data_channel->getRank())) {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5}, -1);
    if (data_channel->getRank() == group_ranks[0])
      int_tensor->fill(2000);

    data_channel->broadcast(*int_tensor, group_ranks[0], group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 2000)
  } else {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5}, 1000);
    data_channel->broadcast(*int_tensor, group_ranks[0], group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 1000)
  }
}

void test_reduce_group(std::shared_ptr<thd::DataChannel> data_channel,
                       THDGroup group, std::vector<thd::rank_type> group_ranks) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support reduce
  }

  if (contains(group_ranks, data_channel->getRank())) {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5}, 10);
    data_channel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, group_ranks[0], group);
    if (data_channel->getRank() == group_ranks[0]) {
      ASSERT_TENSOR_VALUE(int, *int_tensor, 10 * group_ranks.size())
    } else {
      ASSERT_TENSOR_VALUE(int, *int_tensor, 10)
    }
  } else {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5}, 1000);
    data_channel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, group_ranks[0], group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 1000)
  }
}

void test_allReduce_group(std::shared_ptr<thd::DataChannel> data_channel,
                          THDGroup group, std::vector<thd::rank_type> group_ranks) {
  if (contains(group_ranks, data_channel->getRank())) {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5, 6, 7, 100}, 10);
    data_channel->allReduce(*int_tensor, THDReduceOp::THDReduceSUM, group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 10 * group_ranks.size())
  } else {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5, 6, 7, 100}, 1000);
    data_channel->allReduce(*int_tensor, THDReduceOp::THDReduceSUM, group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 1000)
  }
}

void test_scatter_group(std::shared_ptr<thd::DataChannel> data_channel,
                        THDGroup group, std::vector<thd::rank_type> group_ranks) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support scatter
  }

  std::vector<std::shared_ptr<thpp::IntTensor>> tensors;
  std::vector<thpp::Tensor*> raw_tensors;
  if (contains(group_ranks, data_channel->getRank())) {
    if (data_channel->getRank() == group_ranks[0]) {
      for (std::size_t i = 0; i < group_ranks.size(); ++i) {
        tensors.push_back(buildTensor<int>({1, 2, 3, 4, 5}, group_ranks[i]));
        raw_tensors.push_back(tensors.back().get());
      }
    }

    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, -1);
    data_channel->scatter(raw_tensors, *int_tensor, group_ranks[0], group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, data_channel->getRank())
  } else {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5}, 1000);
    data_channel->scatter(raw_tensors, *int_tensor, group_ranks[0], group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 1000)
  }
}


void test_gather_group(std::shared_ptr<thd::DataChannel> data_channel,
                       THDGroup group, std::vector<thd::rank_type> group_ranks) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support gather
  }

  std::vector<std::shared_ptr<thpp::IntTensor>> tensors;
  std::vector<thpp::Tensor*> raw_tensors;
  if (contains(group_ranks, data_channel->getRank())) {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, data_channel->getRank());
    if (data_channel->getRank() == group_ranks[0]) {
      for (std::size_t i = 0; i < group_ranks.size(); ++i) {
        tensors.push_back(buildTensor<int>({1, 2, 3, 4, 5}, -1));
        raw_tensors.push_back(tensors.back().get());
      }

      data_channel->gather(raw_tensors, *int_tensor, group_ranks[0], group);
      for (std::size_t i = 0; i < tensors.size(); ++i)
        ASSERT_TENSOR_VALUE(int, *(tensors[i]), group_ranks[i])
    } else {
      data_channel->gather(raw_tensors, *int_tensor, group_ranks[0], group);
    }
  } else {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5}, 1000);
    data_channel->gather(raw_tensors, *int_tensor, group_ranks[0], group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 1000)
  }
}

void test_allGather_group(std::shared_ptr<thd::DataChannel> data_channel,
                          THDGroup group, std::vector<thd::rank_type> group_ranks) {
  std::vector<std::shared_ptr<thpp::IntTensor>> tensors;
  std::vector<thpp::Tensor*> raw_tensors;
  if (contains(group_ranks, data_channel->getRank())) {
    auto int_tensor = buildTensor<int>({1, 2, 3, 4, 5}, data_channel->getRank());
    for (std::size_t i = 0; i < group_ranks.size(); ++i) {
      tensors.push_back(buildTensor<int>({1, 2, 3, 4, 5}, -1));
      raw_tensors.push_back(tensors.back().get());
    }

    data_channel->allGather(raw_tensors, *int_tensor, group);
    for (std::size_t i = 0; i < tensors.size(); ++i)
      ASSERT_TENSOR_VALUE(int, *(tensors[i]), group_ranks[i])
  } else {
    auto int_tensor = buildTensor({1, 2, 3, 4, 5}, 1000);
    data_channel->allGather(raw_tensors, *int_tensor, group);
    ASSERT_TENSOR_VALUE(int, *int_tensor, 1000)
  }
}

void test_barrier_group(std::shared_ptr<thd::DataChannel> data_channel,
                        THDGroup group, std::vector<thd::rank_type> group_ranks) {
  if (contains(group_ranks, data_channel->getRank())) {
    for (int i = 0; i < group_ranks.size(); ++i) {
      if (data_channel->getRank() == group_ranks[i]) {
        long time_after_barrier = nowInMilliseconds() + BARRIER_WAIT_TIME;
        auto time_tensor = buildTensor<long>({1}, time_after_barrier);
        data_channel->broadcast(*time_tensor, group_ranks[i], group);
        std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_WAIT_TIME + 10));
        data_channel->barrier(group);
      } else {
        auto time_tensor = buildTensor<long>({1}, -1);
        data_channel->broadcast(*time_tensor, group_ranks[i], group); // get expected time after barrier
        data_channel->barrier(group);
        assert(nowInMilliseconds() >= reinterpret_cast<long*>(time_tensor->data())[0]);
      }
    }
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_WAIT_TIME + 100));
    data_channel->barrier(group);
  }
}

////////////////
// EXCEPTIONS //
////////////////

void test_send_recv_invalid_rank(std::shared_ptr<thd::DataChannel> data_channel) {
  if (g_data_channel_type == "gloo") {
    return; // XXX: Gloo does not support send/recv
  }

  if (g_data_channel_type == "mpi") {
    return; // XXX: MPI does not throw exceptions
  }

  auto rank = data_channel->getRank();
  auto int_tensor = buildTensor({1, 2, 3, 4, 5}, -1);

  { // cannot send or receive to self
    ASSERT_THROWS(std::logic_error, data_channel->send(*int_tensor, rank))
    ASSERT_THROWS(std::logic_error, data_channel->receive(*int_tensor, rank))
  }

  { // cannot send or receive to/from process with rank -1
    ASSERT_THROWS(std::out_of_range, data_channel->send(*int_tensor, -1))
    ASSERT_THROWS(std::out_of_range, data_channel->receive(*int_tensor, -1))
  }
}

// Cannot create empty group or group will be null
void test_empty_group(std::shared_ptr<thd::DataChannel> data_channel) {
  // in MPI there will be created NULL_COMM
  if (g_data_channel_type == "tcp" || g_data_channel_type == "gloo") {
    ASSERT_THROWS(std::logic_error, data_channel->newGroup({}))
  }
}

// Process with rank 0 is not part of group, we cannot perform operation to it
void test_process_not_in_group(std::shared_ptr<thd::DataChannel> data_channel) {
  auto int_tensor = buildTensor({1, 2, 3, 4, 5}, -1);

  THDGroup group = data_channel->newGroup({1});
  std::vector<std::shared_ptr<thpp::IntTensor>> tensors = {
    buildTensor<int>({1, 2, 3, 4, 5}, -1)
  };
  std::vector<thpp::Tensor*> raw_tensors = {
    tensors.back().get()
  };

  if (data_channel->getRank() == 1) {
    ASSERT_THROWS(
      std::logic_error,
      data_channel->broadcast(*int_tensor, 0, group)
    )

    if (g_data_channel_type == "gloo") {
      return; // XXX: Gloo does not support scatter/gather/reduce
    }

    ASSERT_THROWS(
      std::logic_error,
      data_channel->reduce(*int_tensor, THDReduceOp::THDReduceSUM, 0, group)
    )

    ASSERT_THROWS(
      std::logic_error,
      data_channel->scatter(raw_tensors, *int_tensor, 0, group)
    )

    ASSERT_THROWS(
      std::logic_error,
      data_channel->gather(raw_tensors, *int_tensor, 0, group)
    )
  }
}

// input_tensors does not match size of group
void test_tensors_do_not_match_group_size(std::shared_ptr<thd::DataChannel> data_channel) {
  auto int_tensor = buildTensor({1, 2, 3, 4, 5}, -1);
  THDGroup group = data_channel->newGroup({1, 2});
  std::vector<std::shared_ptr<thpp::IntTensor>> tensors = {
    buildTensor<int>({1, 2, 3, 4, 5}, -1)
  };
  std::vector<thpp::Tensor*> raw_tensors = {
    tensors.back().get()
  };

  if (data_channel->getRank() == 1 || data_channel->getRank() == 2) {
    ASSERT_THROWS(
      std::logic_error,
      data_channel->allGather(raw_tensors, *int_tensor, group)
    )

    if (g_data_channel_type == "gloo") {
      return; // XXX: Gloo does not support scatter/gather
    }

    if (data_channel->getRank() == 1) {
      ASSERT_THROWS(
        std::logic_error,
        data_channel->scatter(raw_tensors, *int_tensor, 1, group)
      )

      ASSERT_THROWS(
        std::logic_error,
        data_channel->gather(raw_tensors, *int_tensor, 1, group)
      )
    }
  }
}

// input_tensors are not the same
void test_tensors_are_not_the_same(std::shared_ptr<thd::DataChannel> data_channel) {
  auto int_tensor = buildTensor({1, 2, 3, 4, 5}, -1);
  THDGroup group = data_channel->newGroup({1, 2});
  std::vector<std::shared_ptr<thpp::IntTensor>> tensors = {
    buildTensor<int>({1, 2, 3, 4, 5}, -1),
    buildTensor<int>({1, 2, 3, 4}, -1)
  };
  std::vector<thpp::Tensor*> raw_tensors = {
    tensors[0].get(),
    tensors[1].get()
  };

  if (data_channel->getRank() == 1 || data_channel->getRank() == 2) {
    ASSERT_THROWS(
      std::logic_error,
      data_channel->allGather(raw_tensors, *int_tensor, group)
    )

    if (g_data_channel_type == "gloo") {
      return; // XXX: Gloo does not support scatter/gather
    }

    if (data_channel->getRank() == 1) {
      ASSERT_THROWS(
        std::logic_error,
        data_channel->scatter(raw_tensors, *int_tensor, 1, group)
      )

      ASSERT_THROWS(
        std::logic_error,
        data_channel->gather(raw_tensors, *int_tensor, 1, group)
      )
    }
  }
}

void run_all_tests(std::shared_ptr<thd::DataChannel> data_channel, int workers) {
  test_send_recv_tensor(data_channel);
  test_send_recv_tensor_any_source(data_channel, workers);
  test_send_recv_scalar(data_channel);
  test_broadcast(data_channel);
  test_reduce(data_channel, workers);
  test_allReduce(data_channel, workers);
  test_scatter(data_channel);
  test_gather(data_channel);
  test_allGather(data_channel);
  test_barrier(data_channel);
  test_isend(data_channel);
  test_irecv(data_channel);
  test_interlaces(data_channel);

  std::vector<thd::rank_type> group_ranks = {1, 2};
  THDGroup group = data_channel->newGroup(group_ranks);
  test_broadcast_group(data_channel, group, group_ranks);
  test_reduce_group(data_channel, group, group_ranks);
  test_allReduce_group(data_channel, group, group_ranks);
  test_scatter_group(data_channel, group, group_ranks);
  test_gather_group(data_channel, group, group_ranks);
  test_allGather_group(data_channel, group, group_ranks);
  test_barrier_group(data_channel, group, group_ranks);

  test_send_recv_invalid_rank(data_channel);
  test_empty_group(data_channel);
  test_process_not_in_group(data_channel);
  test_tensors_do_not_match_group_size(data_channel);
  test_tensors_are_not_the_same(data_channel);
}


void init_tcp_master(int workers) {
  g_mutex.lock();
  setenv(thd::WORLD_SIZE_ENV, std::to_string((workers + 1)).data(), 1);
  setenv(thd::RANK_ENV, "0", 1);
  setenv(thd::MASTER_PORT_ENV, std::to_string(MASTER_PORT).data(), 1);
  auto masterChannel = std::make_shared<thd::DataChannelTCP>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(masterChannel->init());
  run_all_tests(masterChannel, workers);

  // wait for all workers to finish
  for (auto& worker : g_all_workers) {
    worker.join();
  }
}


void init_tcp_worker(unsigned int id, int workers) {
  g_mutex.lock();
  setenv(thd::RANK_ENV, std::to_string(id).data(), 1);
  setenv(thd::MASTER_ADDR_ENV, std::string("127.0.0.1:" + std::to_string(MASTER_PORT)).data(), 1);
  auto worker_channel = std::make_shared<thd::DataChannelTCP>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(worker_channel->init());
  run_all_tests(worker_channel, workers);
}

#ifdef WITH_GLOO
void init_gloo_master(int workers) {
  g_mutex.lock();
  setenv(thd::WORLD_SIZE_ENV, std::to_string((workers + 1)).data(), 1);
  setenv(thd::RANK_ENV, "0", 1);
  setenv(thd::MASTER_PORT_ENV, std::to_string(MASTER_PORT).data(), 1);
  auto masterChannel = std::make_shared<thd::DataChannelGloo>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(masterChannel->init());
  run_all_tests(masterChannel, workers);

  g_barrier->wait();
}

void init_gloo_worker(unsigned int id, int workers) {
  g_mutex.lock();
  setenv(thd::RANK_ENV, std::to_string(id).data(), 1);
  setenv(thd::MASTER_ADDR_ENV, std::string("127.0.0.1:" + std::to_string(MASTER_PORT)).data(), 1);
  auto worker_channel = std::make_shared<thd::DataChannelGloo>(thd::getInitConfig("env://")); // reads all env variable
  g_mutex.unlock();

  assert(worker_channel->init());
  run_all_tests(worker_channel, workers);

  g_barrier->wait();
}
#endif // WITH_GLOO

#ifdef WITH_MPI
void init_mpi_process() {
  auto data_channel = std::make_shared<thd::DataChannelMPI>();
  assert(data_channel->init());
  run_all_tests(data_channel, WORKERS_NUM[0]);

  std::cout << "MPI OK (id: " << data_channel->getRank() << ")" << std::endl;
}
#endif // WITH_MPI


int main(int argc, char const *argv[]) {
#ifdef WITH_MPI
  if (argc == 1) {
#endif // WITH_MPI
    g_data_channel_type = "tcp";
    for (auto workers : WORKERS_NUM) {
      std::cout << "TCP (workers: " << workers << "):" << std::endl;
      // start tcp master
      std::thread tcp_master_thread(init_tcp_master, workers);

      // start tcp worker
      for (int id = 1; id <= workers; ++id) {
        g_all_workers.push_back(std::thread(init_tcp_worker, id, workers));
      }

      tcp_master_thread.join();
      g_all_workers.clear();

      std::cout << "TCP - OK" << std::endl;
    }

#ifdef WITH_GLOO
    g_data_channel_type = "gloo";
    for (auto workers : WORKERS_NUM) {
      g_barrier.reset(new Barrier(workers + 1));
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
#endif // WITH_GLOO

#ifdef WITH_MPI
    std::cout << "--------------------------" << std::endl;

    // start MPI processes
    std::cout << "MPI:" << std::endl;
    execlp("mpirun", "mpirun", "-n", std::to_string(WORKERS_NUM[0] + 1).data(), argv[0], "1", NULL);
  } else {
    g_data_channel_type = "mpi";
    init_mpi_process();
  }
#endif // WITH_MPI
  return 0;
}
