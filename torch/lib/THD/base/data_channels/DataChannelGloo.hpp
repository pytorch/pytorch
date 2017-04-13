#pragma once

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
struct hash<std::tuple<::thd::DataOperation, THDGroup, std::size_t, std::size_t, THDReduceOp>> {
  std::size_t operator()(const std::tuple<::thd::DataOperation, THDGroup, std::size_t, std::size_t, THDReduceOp>& k) const {
    return (
      hash<::thd::DataOperation>()(std::get<0>(k)) ^
      hash<THDGroup>()(std::get<1>(k)) ^
      hash<std::size_t>()(std::get<2>(k)) ^ 
      hash<std::size_t>()(std::get<3>(k)) ^
      hash<THDReduceOp>()(std::get<4>(k))
    );
  }
};

} // namespace std

namespace thd {

struct DataChannelGloo : DataChannel {
  using store_type = ::gloo::rendezvous::PrefixStore;


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

  store_type getStore();

  void _send(const Scalar& data, rank_type dst_id);
  void _send(thpp::Tensor& data, rank_type dst_id);
  void _receive(Scalar& data, rank_type src_id);
  void _receive(thpp::Tensor& data, rank_type src_id);

  rank_type _rank; // Current process' rank
  rank_type _num_processes; // Number of processes in network
  std::unique_ptr<::gloo::rendezvous::Store> _store;
  std::shared_ptr<::gloo::transport::Device> _device;
  std::unordered_map<THDGroup, DataChannel::Group> _groups;

  // Workers
  QueueWorker _send_worker, _receive_worker;
};


template<DataOperation D, typename T>
struct algorithm_spec;

struct GlooCache {
  using buffer_type = char;
  using algorithm_type = ::gloo::Algorithm;
  using context_type = ::gloo::rendezvous::Context;

  using key_type = std::tuple<
    DataOperation, // operation
    THDGroup,      // group
    std::size_t,   // input buffer bytes
    std::size_t,   // output buffer bytes
    THDReduceOp    // reduce op
  >;
  using value_type = std::tuple<
    std::shared_ptr<algorithm_type>, // algorithm
    std::shared_ptr<buffer_type>,    // input buffer (nullptr if not used)
    std::shared_ptr<buffer_type>     // output buffer (nullptr if not used)
  >;

  GlooCache(GlooCache const&)      = delete;
  void operator=(GlooCache const&) = delete;

  // singleton instance
  static GlooCache& get() {
    static GlooCache instance;
    return instance;
  }

  // TODO: enforce setters to be the first thing called (maybe some kind of factory)
  void setRank(rank_type rank) {
    _rank = rank;
  }

  void setDevice(std::shared_ptr<::gloo::transport::Device> device) {
    _device = device;
  }

  std::shared_ptr<context_type> createContext(
    const DataChannel::Group& group,
    DataChannelGloo::store_type& store
  ) {
    auto context = std::make_shared<context_type>(group.mustGetGroupRank(_rank), group.size());
    context->connectFullMesh(store, _device);
    return context;
  }

  template<DataOperation op>
  std::shared_ptr<context_type> getSharedContext(
    const DataChannel::Group& group,
    DataChannelGloo::store_type& store
  ) {
    if (_shared_contexts.find(op) == _shared_contexts.end()) {
      _shared_contexts[op] = createContext(group, store);
    }

    return _shared_contexts[op];
  }

  std::shared_ptr<buffer_type> createBuffer(std::size_t bytes) {
    return std::shared_ptr<buffer_type>(new char[bytes],
                                        std::default_delete<char[]>());
  }

  template<DataOperation D, typename T, typename... Args>
  value_type getAlgorithm(THDGroup group_id, const DataChannel::Group& group,
                          DataChannelGloo::store_type& store, Args... args) {
    auto key = algorithm_spec<D, T>::key(group_id, args...);
    if (_algorithms.find(key) == _algorithms.end()) {
      _algorithms[key] = algorithm_spec<D, T>::create(group, store, args...);
    }

    return _algorithms[key];
  }

private:
  GlooCache() {}

  rank_type _rank;
  std::shared_ptr<::gloo::transport::Device> _device;

  std::unordered_map<DataOperation, std::shared_ptr<context_type>> _shared_contexts;
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
                           input_bytes, output_bytes, THDReduceMIN);
  }

  static GlooCache::value_type create(
    const DataChannel::Group& group, DataChannelGloo::store_type& store,
    std::size_t input_bytes, std::size_t output_bytes, std::size_t count
  ) {
    auto& cache = GlooCache::get();
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
                           input_bytes, input_bytes, op);
  }

  static GlooCache::value_type create(
    const DataChannel::Group& group, DataChannelGloo::store_type& store,
    std::size_t input_bytes, std::size_t count, THDReduceOp op
  ) {
    auto& cache = GlooCache::get();
    auto context = cache.createContext(group, store);
    auto input_buffer = cache.createBuffer(input_bytes);
    return std::make_tuple(
      std::make_shared<::gloo::AllreduceRing<T>>(
        context,
        std::initializer_list<T*>{reinterpret_cast<T*>(input_buffer.get())},
        count, THDToGlooReduceOp<T>(op)),
      input_buffer,
      input_buffer // we get the result in same buffer
    );
  }
};

template<typename T>
struct algorithm_spec<DataOperation::BROADCAST, T> {
  static GlooCache::key_type key(
    THDGroup group_id, std::size_t input_bytes,
    std::size_t unused_count, rank_type unused_src_rank
  ) {
    return std::make_tuple(DataOperation::BROADCAST, group_id,
                           input_bytes, input_bytes, THDReduceMIN);
  }

  static GlooCache::value_type create(
    const DataChannel::Group& group, DataChannelGloo::store_type& store,
    std::size_t input_bytes, std::size_t count, rank_type src_rank
  ) {
    auto& cache = GlooCache::get();
    auto context = cache.createContext(group, store);
    auto input_buffer = cache.createBuffer(input_bytes);
    return std::make_tuple(
      std::make_shared<::gloo::BroadcastOneToAll<T>>(
        context,
        std::initializer_list<T*>{reinterpret_cast<T*>(input_buffer.get())},
        count, src_rank),
      input_buffer,
      input_buffer // we get the result in same buffer
    );
  }
};

template<typename T> // unused
struct algorithm_spec<DataOperation::BARRIER, T> {
  static GlooCache::key_type key(THDGroup group_id) {
    return std::make_tuple(DataOperation::BARRIER, group_id, 0, 0, THDReduceMIN);
  }

  static GlooCache::value_type create(
    const DataChannel::Group& group,
    DataChannelGloo::store_type& store
  ) {
    auto context = GlooCache::get().createContext(group, store);
    return std::make_tuple(
      std::make_shared<::gloo::BarrierAllToAll>(context),
      nullptr,
      nullptr
    );
  }
};

} // namespace thd

