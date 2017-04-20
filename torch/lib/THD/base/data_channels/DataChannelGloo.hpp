#pragma once

#include "../ChannelUtils.hpp"
#include "../DataChannel.hpp"
#include "DataChannelUtils.hpp"

#include "gloo/algorithm.h"
#include "gloo/allgather_ring.h"
#include "gloo/allreduce_ring.h"
#include "gloo/barrier_all_to_all.h"
#include "gloo/broadcast_one_to_all.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"
#include "gloo/rendezvous/prefix_store.h"

#include <cstdint>
#include <map>
#include <vector>
#include <tuple>

namespace std {

template<>
struct hash<std::tuple<::thd::DataOperation, THDGroup>> {
  std::size_t operator()(const std::tuple<::thd::DataOperation, THDGroup>& k) const {
    return (
      hash<::thd::DataOperation>()(std::get<0>(k)) ^
      hash<THDGroup>()(std::get<1>(k))
    );
  }
};

template<>
struct hash<std::tuple<::thd::DataOperation, THDGroup, std::size_t,
                       std::size_t, THDReduceOp, thd::rank_type>> {
  std::size_t operator()(const std::tuple<::thd::DataOperation, THDGroup,
                                          std::size_t, std::size_t, THDReduceOp,
                                          thd::rank_type>& k) const {
    return (
      hash<::thd::DataOperation>()(std::get<0>(k)) ^
      hash<THDGroup>()(std::get<1>(k)) ^
      hash<std::size_t>()(std::get<2>(k)) ^
      hash<std::size_t>()(std::get<3>(k)) ^
      hash<THDReduceOp>()(std::get<4>(k)) ^
      hash<thd::rank_type>()(std::get<5>(k))
    );
  }
};

} // namespace std

namespace thd {

struct GlooCache;

struct DataChannelGloo : DataChannel {
  using store_type = ::gloo::rendezvous::Store;


  struct RequestGloo : DataChannel::Request {
    RequestGloo(QueueWorker::Request&& request);
    virtual ~RequestGloo();

    virtual bool isCompleted() override;
    virtual void wait() override;

  private:
    QueueWorker::Request _request;
  };

  DataChannelGloo();
  DataChannelGloo(int timeout);
  virtual ~DataChannelGloo();

  bool init() override;

  rank_type getRank() override;
  rank_type getNumProcesses() override;

  void allGather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
                 THDGroup group_id = THDGroupWORLD) override;
  void gather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
              rank_type dst_rank, THDGroup group_id = THDGroupWORLD) override;
  void scatter(std::vector<thpp::Tensor*>& input, thpp::Tensor& output,
               rank_type src_rank, THDGroup group_id = THDGroupWORLD) override;
  void allReduce(thpp::Tensor& data, THDReduceOp operation,
                 THDGroup group_id = THDGroupWORLD) override;
  void reduce(thpp::Tensor& data, THDReduceOp operation, rank_type dst_rank,
              THDGroup group_id = THDGroupWORLD) override;
  void broadcast(thpp::Tensor& data, rank_type src_id,
                 THDGroup group_id = THDGroupWORLD) override;
  void send(const Scalar& data, rank_type dst_id) override;
  void send(thpp::Tensor& data, rank_type dst_id) override;
  void receive(Scalar& data, rank_type src_id) override;
  void receive(thpp::Tensor& data) override;
  void receive(thpp::Tensor& data, rank_type src_id) override;
  RequestGloo* isend(thpp::Tensor& data, rank_type dst_rank) override;
  RequestGloo* ireceive(thpp::Tensor& data, rank_type src_rank) override;

  void barrier(THDGroup group_id = THDGroupWORLD) override;

  THDGroup newGroup(const std::vector<rank_type>& ranks) override;

private:
  template<typename T>
  void allGatherT(std::vector<thpp::Tensor*>& output,
                  thpp::Tensor& input, THDGroup group_id);

  template<typename T>
  void allReduceT(thpp::Tensor& data, THDReduceOp operation,
                  THDGroup group_id = THDGroupWORLD);

  template<typename T>
  void broadcastT(thpp::Tensor& data, rank_type src_rank,
                  THDGroup group_id = THDGroupWORLD);

  rank_type _rank; // Current process' rank
  std::string _addr;
  port_type _port;
  rank_type _num_processes; // Number of processes in network
  std::shared_ptr<store_type> _store;
  std::shared_ptr<::gloo::transport::Device> _device;
  std::unordered_map<THDGroup, DataChannel::Group> _groups;
  
  std::unique_ptr<GlooCache> _cache;

  // Workers
  QueueWorker _send_worker, _receive_worker;
};


template<DataOperation D, typename T>
struct algorithm_spec;

struct GlooCache {
  using buffer_type = char;
  using algorithm_type = ::gloo::Algorithm;
  using context_type = ::gloo::rendezvous::Context;
  using store_type = ::gloo::rendezvous::PrefixStore;

  using key_type = std::tuple<
    DataOperation, // operation
    THDGroup,      // group
    std::size_t,   // input buffer bytes
    std::size_t,   // output buffer bytes
    THDReduceOp,   // reduce op
    rank_type      // src/dest rank
  >;
  using value_type = std::tuple<
    std::shared_ptr<algorithm_type>, // algorithm
    std::shared_ptr<buffer_type>,    // input buffer (nullptr if not used)
    std::shared_ptr<buffer_type>     // output buffer (nullptr if not used)
  >;

  GlooCache(rank_type rank, std::shared_ptr<::gloo::transport::Device> device,
            std::shared_ptr<DataChannelGloo::store_type> store)
   : _rank(rank)
   , _device(device)
   , _store(store)
  {}

  GlooCache(GlooCache const&)      = delete;
  void operator=(GlooCache const&) = delete;

  std::shared_ptr<context_type> createContext(
    const DataChannel::Group& group,
    store_type& store
  ) {
    auto context = std::make_shared<context_type>(group.mustGetGroupRank(_rank), group.size());
    context->connectFullMesh(store, _device);
    return context;
  }

  std::shared_ptr<buffer_type> createBuffer(std::size_t bytes) const {
    return std::shared_ptr<buffer_type>(new char[bytes],
                                        std::default_delete<char[]>());
  }

  template<DataOperation D, typename T, typename... Args>
  value_type getAlgorithm(THDGroup group_id, const DataChannel::Group& group,
                          Args... args) {
    auto key = algorithm_spec<D, T>::key(group_id, args...);
    if (_algorithms.find(key) == _algorithms.end()) {
      // create prefix store with unique prefix
      store_type prefix_store(print_key(key), *_store);
      _algorithms[key] = algorithm_spec<D, T>::create(*this, group, prefix_store, args...);
    }

    return _algorithms[key];
  }

private:
  std::string print_key(const key_type& k) { // TODO: define aka `to_string` for tuples instead of this function
    return std::to_string(static_cast<uint8_t>(std::get<0>(k))) + "-"
      + std::to_string(std::get<1>(k)) + "-"
      + std::to_string(std::get<2>(k)) + "-"
      + std::to_string(std::get<3>(k)) + "-"
      + std::to_string(std::get<4>(k)) + "-"
      + std::to_string(std::get<5>(k));
  }

  rank_type _rank;
  std::shared_ptr<::gloo::transport::Device> _device;
  std::shared_ptr<DataChannelGloo::store_type> _store;

  std::unordered_map<key_type, value_type> _algorithms;
};

template<typename T>
const ::gloo::ReductionFunction<T>* THDToGlooReduceOp(THDReduceOp op) {
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

template<typename T>
struct algorithm_spec<DataOperation::ALL_GATHER, T> {
  static GlooCache::key_type key(
    THDGroup group_id, std::size_t input_bytes, std::size_t output_bytes,
    std::size_t unused_count
  ) {
    return std::make_tuple(DataOperation::ALL_GATHER, group_id,
                           input_bytes, output_bytes, THDReduceMIN, 0);
  }

  static GlooCache::value_type create(GlooCache& cache,
    const DataChannel::Group& group, GlooCache::store_type& store,
    std::size_t input_bytes, std::size_t output_bytes, std::size_t count
  ) {
    auto context = cache.createContext(group, store);
    auto input_buffer = cache.createBuffer(input_bytes);
    auto output_buffer = cache.createBuffer(output_bytes);

    return std::make_tuple(
      std::make_shared<::gloo::AllgatherRing<T>>(
        context,
        std::initializer_list<T*>{reinterpret_cast<T*>(input_buffer.get())},
        reinterpret_cast<T*>(output_buffer.get()),
        count),
      input_buffer,
      output_buffer
    );
  }
};

template<typename T>
struct algorithm_spec<DataOperation::ALL_REDUCE, T> {
  static GlooCache::key_type key(
    THDGroup group_id, std::size_t input_bytes,
    std::size_t unused_count, THDReduceOp op
  ) {
    return std::make_tuple(DataOperation::ALL_REDUCE, group_id,
                           input_bytes, input_bytes, op, 0);
  }

  static GlooCache::value_type create(GlooCache& cache,
    const DataChannel::Group& group, GlooCache::store_type& store,
    std::size_t input_bytes, std::size_t count, THDReduceOp op
  ) {
    auto context = cache.createContext(group, store);
    auto input_buffer = cache.createBuffer(input_bytes);
    return std::make_tuple(
      std::make_shared<::gloo::AllreduceRing<T>>(
        context,
        std::initializer_list<T*>{reinterpret_cast<T*>(input_buffer.get())},
        count,
        THDToGlooReduceOp<T>(op)),
      input_buffer,
      input_buffer // we get the result in same buffer
    );
  }
};

template<typename T>
struct algorithm_spec<DataOperation::BROADCAST, T> {
  static GlooCache::key_type key(
    THDGroup group_id, std::size_t input_bytes,
    std::size_t unused_count, rank_type src_rank
  ) {
    return std::make_tuple(DataOperation::BROADCAST, group_id,
                           input_bytes, input_bytes, THDReduceMIN, src_rank);
  }

  static GlooCache::value_type create(GlooCache& cache,
    const DataChannel::Group& group, GlooCache::store_type& store,
    std::size_t input_bytes, std::size_t count, rank_type src_rank
  ) {
    auto context = cache.createContext(group, store);
    auto input_buffer = cache.createBuffer(input_bytes);
    return std::make_tuple(
      std::make_shared<::gloo::BroadcastOneToAll<T>>(
        context,
        std::initializer_list<T*>{reinterpret_cast<T*>(input_buffer.get())},
        count,
        src_rank),
      input_buffer,
      input_buffer // we get the result in same buffer
    );
  }
};

template<typename T> // unused
struct algorithm_spec<DataOperation::BARRIER, T> {
  static GlooCache::key_type key(THDGroup group_id) {
    return std::make_tuple(DataOperation::BARRIER, group_id, 0, 0, THDReduceMIN, 0);
  }

  static GlooCache::value_type create(GlooCache& cache,
    const DataChannel::Group& group, GlooCache::store_type& store
  ) {
    auto context = cache.createContext(group, store);
    return std::make_tuple(
      std::make_shared<::gloo::BarrierAllToAll>(context),
      nullptr,
      nullptr
    );
  }
};

} // namespace thd

