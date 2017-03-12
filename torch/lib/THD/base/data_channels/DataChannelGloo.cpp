#include "DataChannelGloo.hpp"
#include "DataChannelUtils.hpp"
#include "Store.hpp"

#include "gloo/allreduce_ring.h"
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

DataChannelGloo::DataChannelGloo()
  : _rank(load_rank_env())
  , _num_processes(load_world_size_env())
{
  std::string redis_addr;
  port_type redis_port;
  std::tie(redis_addr, redis_port) = load_worker_env();

  _store = std::unique_ptr<Store>(new thd::Store());

  ::gloo::transport::tcp::attr attr; // default options listen on all interfaces
  _device = ::gloo::transport::tcp::CreateDevice(attr);
}


DataChannelGloo::~DataChannelGloo() {
}


bool DataChannelGloo::init() {
  getFullMeshCtx(); // wait for everyone to connect
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

std::shared_ptr<Context> DataChannelGloo::getFullMeshCtx() {
  auto ctx = std::make_shared<Context>(_rank, _num_processes);
  auto store = getStore();
  ctx->connectFullMesh(store, _device); // TODO: connect only the ring
  return ctx;
}


void DataChannelGloo::allGather(std::vector<thpp::Tensor*>& output,
                               thpp::Tensor& input, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support allGather");
}


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
  ::gloo::AllreduceRing<T> algo(getFullMeshCtx(), {reinterpret_cast<T*>(t.data())}, t.numel());
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

void DataChannelGloo::broadcast(thpp::Tensor& data, rank_type src_rank,
                               THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support broadcast");
}


void DataChannelGloo::send(const Scalar& data, rank_type dst_rank) {
  throw std::runtime_error("DataChannelGloo doesn't support send");
}


void DataChannelGloo::send(thpp::Tensor& data, rank_type dst_rank) {
  throw std::runtime_error("DataChannelGloo doesn't support send");
}


void DataChannelGloo::receive(Scalar& data, rank_type src_rank) {
  throw std::runtime_error("DataChannelGloo doesn't support receive");
}


void DataChannelGloo::receive(thpp::Tensor& data) {
  throw std::runtime_error("DataChannelGloo doesn't support receive");
}


void DataChannelGloo::receive(thpp::Tensor& data, rank_type src_rank) {
  throw std::runtime_error("DataChannelGloo doesn't support receive");
}


void DataChannelGloo::barrier(THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support barrier");
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

} // namespace thd

