#include "DataChannelGloo.hpp"
#include "DataChannelUtils.hpp"
#include "Store.hpp"

#include "gloo/allgather_ring.h"
#include "gloo/allreduce_ring.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/rendezvous/prefix_store.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>


namespace thd {

using ::gloo::rendezvous::PrefixStore;
using ::gloo::rendezvous::Context;

namespace {

template<typename T>

const ::gloo::ReductionFunction<T> *THDToGlooReduceOp(THDReduceOp op);

} // anonymous namespace

DataChannelGloo::DataChannelGloo()
  : _rank(load_rank_env())
{
  _store = std::unique_ptr<Store>(new thd::Store());

  ::gloo::transport::tcp::attr attr; // default options listen on all interfaces
  _device = ::gloo::transport::tcp::CreateDevice(attr);
  if (_rank == 0) {
    _num_processes = load_world_size_env();
    auto num_proc_str = std::to_string(_num_processes);
    _store->set("world_size",
            std::vector<char>(num_proc_str.begin(), num_proc_str.end()));
  } else {
    auto world_size = _store->get("world_size");
    _num_processes = std::atoll(
            std::string(world_size.begin(), world_size.end()).c_str());
  }
}


DataChannelGloo::~DataChannelGloo() {
}


bool DataChannelGloo::init() {
  // there is a separate context for every operation
  for (uint8_t i = 0; i < static_cast<uint8_t>(DataOperation::LAST); i++) {
    _contexts[static_cast<DataOperation>(i)] =
        std::make_shared<Context>(_rank, _num_processes);
    auto store = getStore();
    _contexts[static_cast<DataOperation>(i)]->connectFullMesh(store, _device);
  }
  return true;
}

PrefixStore DataChannelGloo::getStore() {
  static std::uint64_t id = 0; // TODO: that's not a good solution
  return PrefixStore(std::to_string(id++), *_store);
}


rank_type DataChannelGloo::getRank() {
  return _rank;
}


rank_type DataChannelGloo::getNumProcesses() {
  return _num_processes;
}

std::shared_ptr<Context> DataChannelGloo::getFullMeshCtx(DataOperation op) {
  return _contexts[op];
}

template<typename T>
void DataChannelGloo::allGatherT(std::vector<thpp::Tensor*>& output,
                                 thpp::Tensor& input, THDGroup group_id) {
  std::uint64_t tensor_bytes = input.elementSize() * input.numel();
  std::uint64_t all_tensor_bytes = tensor_bytes * output.size();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[all_tensor_bytes]);
  ::gloo::AllgatherRing<T> algo(
        getFullMeshCtx(DataOperation::ALL_GATHER),
        {reinterpret_cast<T*>(input.data())},
        reinterpret_cast<T*>(tmp_data.get()),
        input.numel()
      );
  algo.run();
  for (std::size_t i = 0; i < output.size(); i++) {
    memcpy(output.at(i)->data(), tmp_data.get() + (i * tensor_bytes), tensor_bytes);
  }
}

void DataChannelGloo::allGather(std::vector<thpp::Tensor*>& output,
                                thpp::Tensor& input, THDGroup group_id) {
  if (input.type() == ::thpp::Type::FLOAT) {
    allGatherT<float>(output, input, group_id);
  }
}

// gather and scatter are in fact not supported by gloo
void DataChannelGloo::gather(std::vector<thpp::Tensor*>& output,
                             thpp::Tensor& input, rank_type dst_rank,
                             THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support gather");
}

void DataChannelGloo::scatter(std::vector<thpp::Tensor*>& input,
                              thpp::Tensor& output,
                              rank_type src_rank, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support scatter");
}

template<typename T>
void DataChannelGloo::allReduceT(thpp::Tensor& t, THDReduceOp operation,
                                 THDGroup group_id) {
  ::gloo::AllreduceRing<T> algo(
        getFullMeshCtx(DataOperation::ALL_REDUCE),
        {reinterpret_cast<T*>(t.data())},
        t.numel(),
        THDToGlooReduceOp<T>(operation)
      );
  algo.run();
}

void DataChannelGloo::allReduce(thpp::Tensor& data, THDReduceOp operation,
                                THDGroup group_id) {
  if (data.type() == ::thpp::Type::FLOAT) {
    allReduceT<float>(data, operation, group_id);
  }
}

void DataChannelGloo::reduce(thpp::Tensor& data, THDReduceOp operation,
                             rank_type dst_rank, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support reduce");
}

template<typename T>
void DataChannelGloo::broadcastT(thpp::Tensor& data, rank_type src_rank,
                                THDGroup group_id) {
  ::gloo::BroadcastOneToAll<T> algo(
        getFullMeshCtx(DataOperation::BROADCAST),
        {reinterpret_cast<T*>(data.data())},
        data.numel(),
        src_rank
      );
  algo.run();
}

void DataChannelGloo::broadcast(thpp::Tensor& data, rank_type src_rank,
                                THDGroup group_id) {
  if (data.type() == ::thpp::Type::FLOAT) {
    broadcastT<float>(data, src_rank, group_id);
  }
}


void DataChannelGloo::send(const Scalar& data, rank_type dst_rank) {
  std::unique_ptr<Scalar> data_copy(data.clone());
  auto ctx = getFullMeshCtx(DataOperation::SEND);
  auto& pair = ctx->getPair(dst_rank);
  pair->createSendBuffer(ctx->nextSlot(), data_copy->data(), data_copy->elementSize())->send();
}


void DataChannelGloo::send(thpp::Tensor& data, rank_type dst_rank) {
  auto ctx = getFullMeshCtx(DataOperation::SEND);
  auto& pair = ctx->getPair(dst_rank);
  uint64_t tensor_bytes = data.elementSize() * data.numel();
  pair->createSendBuffer(ctx->nextSlot(), data.data(), tensor_bytes)->send();
}


void DataChannelGloo::receive(Scalar& data, rank_type src_rank) {
  auto ctx = getFullMeshCtx(DataOperation::SEND);
  auto& pair = ctx->getPair(src_rank);
  pair->createRecvBuffer(ctx->nextSlot(), data.data(), data.elementSize())->waitRecv();
}


void DataChannelGloo::receive(thpp::Tensor& data) {
  throw std::runtime_error("DataChannelGloo doesn't support anonymous receive");
}


void DataChannelGloo::receive(thpp::Tensor& data, rank_type src_rank) {
  auto ctx = getFullMeshCtx(DataOperation::SEND);
  auto& pair = ctx->getPair(src_rank);
  uint64_t tensor_bytes = data.elementSize() * data.numel();
  pair->createRecvBuffer(ctx->nextSlot(), data.data(), tensor_bytes)->waitRecv();
}

void DataChannelGloo::barrier(THDGroup group_id) {
  ::gloo::BarrierAllToAll algo(
          getFullMeshCtx(DataOperation::BARRIER)
        );
  algo.run();
}


auto DataChannelGloo::isend(thpp::Tensor& data, rank_type dst_rank) -> Request* {
  throw std::runtime_error("DataChannelGloo doesn't support isend");
}


auto DataChannelGloo::ireceive(thpp::Tensor& data, rank_type src_rank) -> Request*{
  throw std::runtime_error("DataChannelGloo doesn't support ireceive");
}

THDGroup DataChannelGloo::newGroup(const std::vector<rank_type>& ranks) {
  throw std::runtime_error("DataChannelGloo doesn't support creation of new groups");
}

namespace {

template<typename T>
const ::gloo::ReductionFunction<T> *THDToGlooReduceOp(THDReduceOp op) {
  switch (op) {
    case THDReduceMIN:
      return ::gloo::ReductionFunction<T>::min;
    case THDReduceMAX:
      return ::gloo::ReductionFunction<T>::max;
    case THDReduceSUM:
      return ::gloo::ReductionFunction<T>::sum;
    case THDReducePRODUCT:
      return ::gloo::ReductionFunction<T>::product;
    default:
      throw std::invalid_argument("unknown reduce operation");
  }
}

} // anonymous namespace

} // namespace thd

