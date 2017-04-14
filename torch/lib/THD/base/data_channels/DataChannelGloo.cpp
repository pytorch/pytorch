#include "DataChannelGloo.hpp"
#include "DataChannelUtils.hpp"
#include "Store.hpp"

#include "gloo/transport/tcp/device.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>



#define RETURN_IF_NOT_IN_GROUP(name)                                          \
  rank_type name;                                                             \
  {                                                                           \
    bool exists;                                                              \
    std::tie(name, exists) = _groups.at(group_id).getGroupRank(_rank);        \
    if (!exists) return;                                                      \
  }


#define GENERATE_ALL_TYPES(type, func, args...)                               \
  switch (type) {                                                             \
    case ::thpp::Type::CHAR: func<char>(args); break;                         \
    case ::thpp::Type::UCHAR: func<unsigned char>(args); break;               \
    case ::thpp::Type::FLOAT: func<float>(args); break;                       \
    case ::thpp::Type::DOUBLE: func<double>(args); break;                     \
    /* case ::thpp::Type::HALF: func<float>(args); break; */                  \
    case ::thpp::Type::SHORT: func<short>(args); break;                       \
    case ::thpp::Type::USHORT: func<unsigned short>(args); break;             \
    case ::thpp::Type::INT: func<int>(args); break;                           \
    case ::thpp::Type::UINT: func<unsigned int>(args); break;                 \
    case ::thpp::Type::LONG: func<long>(args); break;                         \
    case ::thpp::Type::ULONG: func<unsigned long>(args); break;               \
    case ::thpp::Type::LONG_LONG: func<long long>(args); break;               \
    case ::thpp::Type::ULONG_LONG: func<unsigned long long>(args); break;     \
    default:                                                                  \
      throw std::runtime_error("Invalid " + std::string(#func) + " function type"); \
  }


namespace thd {

DataChannelGloo::RequestGloo::RequestGloo(QueueWorker::Request&& request)
  : _request(std::move(request)) {
}


DataChannelGloo::RequestGloo::~RequestGloo() {}


bool DataChannelGloo::RequestGloo::isCompleted() {
  return _request.isCompleted();
}


void DataChannelGloo::RequestGloo::wait() {
  _request.wait();
}


DataChannelGloo::DataChannelGloo()
  : _rank(load_rank_env())
{
  _store = std::unique_ptr<::gloo::rendezvous::Store>(new thd::Store());

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

  ::gloo::transport::tcp::attr attr; // default options listen on all interfaces
  _device = ::gloo::transport::tcp::CreateDevice(attr);

  GlooCache::get().setRank(_rank);
  GlooCache::get().setDevice(_device);
}


DataChannelGloo::~DataChannelGloo() {
}


bool DataChannelGloo::init() {
  std::vector<rank_type> ranks;
  ranks.reserve(_num_processes);
  for (rank_type rank = 0; rank < _num_processes; ++rank)
    ranks.push_back(rank);

  _groups.insert({
    THDGroupWORLD,
    DataChannel::Group(ranks, _num_processes - 1)
  });

  return true;
}


template<DataOperation op, typename... Args>
DataChannelGloo::store_type
DataChannelGloo::getStore(THDGroup group_id, Args... args) {
  std::string unique_prefix = std::to_string(static_cast<uint8_t>(op)) + "-" +
      std::to_string(group_id);
  std::vector<std::string> v = {std::to_string(args)...};
  for (auto it = v.begin(); it != v.end(); ++it) {
    unique_prefix += "-" + *it;
  }

  return ::gloo::rendezvous::PrefixStore(unique_prefix, *_store);
}


rank_type DataChannelGloo::getRank() {
  return _rank;
}


rank_type DataChannelGloo::getNumProcesses() {
  return _num_processes;
}


template<typename T>
void DataChannelGloo::allGatherT(std::vector<thpp::Tensor*>& output,
                                 thpp::Tensor& input, THDGroup group_id) {
  std::uint64_t tensor_bytes = input.elementSize() * input.numel();
  std::uint64_t all_tensor_bytes = tensor_bytes * output.size();
  auto ret = GlooCache::get().getAlgorithm<DataOperation::ALL_GATHER, T>(
    group_id, _groups.at(group_id), getStore<DataOperation::ALL_GATHER>(group_id),
    tensor_bytes, all_tensor_bytes, input.numel());

  std::memcpy(std::get<1>(ret).get(), input.data(), tensor_bytes);
  std::get<0>(ret)->run();

  for (std::size_t i = 0; i < output.size(); i++) {
    std::memcpy(output.at(i)->data(),
                std::get<2>(ret).get() + (i * tensor_bytes),
                tensor_bytes);
  }
}

void DataChannelGloo::allGather(std::vector<thpp::Tensor*>& output,
                                thpp::Tensor& input, THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP(_)
  GENERATE_ALL_TYPES(input.type(), allGatherT, output, input, group_id)
}


// XXX: `gather` is not supported by Gloo yet.
void DataChannelGloo::gather(std::vector<thpp::Tensor*>& output,
                             thpp::Tensor& input, rank_type dst_rank,
                             THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support gather");
}


// XXX: `scatter` is not supported by Gloo yet.
void DataChannelGloo::scatter(std::vector<thpp::Tensor*>& input,
                              thpp::Tensor& output,
                              rank_type src_rank, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support scatter");
}


template<typename T>
void DataChannelGloo::allReduceT(thpp::Tensor& t, THDReduceOp operation,
                                 THDGroup group_id) {
  std::uint64_t tensor_bytes = t.elementSize() * t.numel();
  auto ret = GlooCache::get().getAlgorithm<DataOperation::ALL_REDUCE, T>(
    group_id, _groups.at(group_id), getStore<DataOperation::ALL_REDUCE>(group_id),
    tensor_bytes, t.numel(), operation);

  std::memcpy(std::get<1>(ret).get(), t.data(), tensor_bytes);
  std::get<0>(ret)->run();
  std::memcpy(t.data(), std::get<2>(ret).get(), tensor_bytes);
}

void DataChannelGloo::allReduce(thpp::Tensor& data, THDReduceOp operation,
                                THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP(_)
  GENERATE_ALL_TYPES(data.type(), allReduceT, data, operation, group_id)
}


// XXX: `reduce` is not supported by Gloo yet.
void DataChannelGloo::reduce(thpp::Tensor& data, THDReduceOp operation,
                             rank_type dst_rank, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support reduce");
}


template<typename T>
void DataChannelGloo::broadcastT(thpp::Tensor& data, rank_type src_rank,
                                THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP(group_rank)

  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  auto ret = GlooCache::get().getAlgorithm<DataOperation::BROADCAST, T>(
    group_id, _groups.at(group_id), getStore<DataOperation::BROADCAST>(group_id),
    tensor_bytes, data.numel(), src_rank);

  if (group_rank == src_rank)
    std::memcpy(std::get<1>(ret).get(), data.data(), tensor_bytes);

  std::get<0>(ret)->run();

  if (group_rank != src_rank)
    std::memcpy(data.data(), std::get<2>(ret).get(), tensor_bytes);
}


void DataChannelGloo::broadcast(thpp::Tensor& data, rank_type src_rank,
                                THDGroup group_id) {
  GENERATE_ALL_TYPES(data.type(), broadcastT, data, src_rank, group_id)
}


void DataChannelGloo::send(const Scalar& data, rank_type dst_rank) {
  auto request = _send_worker.push([this, &data, dst_rank]{
    this->_send(data, dst_rank);
  });
  request.wait();
}


void DataChannelGloo::send(thpp::Tensor& data, rank_type dst_rank) {
  auto request = _send_worker.push([this, &data, dst_rank]{
    this->_send(data, dst_rank);
  });
  request.wait();
}


void DataChannelGloo::receive(Scalar& data, rank_type src_rank) {
  auto request = _receive_worker.push([this, &data, src_rank]{
    this->_receive(data, src_rank);
  });
  request.wait();
}


void DataChannelGloo::receive(thpp::Tensor& data) {
  throw std::runtime_error("DataChannelGloo doesn't support anonymous receive");
}


void DataChannelGloo::receive(thpp::Tensor& data, rank_type src_rank) {
  auto request = _receive_worker.push([this, &data, src_rank]{
    this->_receive(data, src_rank);
  });
  request.wait();
}


auto DataChannelGloo::isend(thpp::Tensor& data, rank_type dst_rank) -> RequestGloo* {
  std::shared_ptr<thpp::Tensor> copy_tensor(data.clone_shallow());
  auto request = _send_worker.push([this, copy_tensor, dst_rank]{
    this->_send(*copy_tensor, dst_rank);
  });
  return new DataChannelGloo::RequestGloo(std::move(request));
}


auto DataChannelGloo::ireceive(thpp::Tensor& data, rank_type src_rank) -> RequestGloo* {
  std::shared_ptr<thpp::Tensor> copy_tensor(data.clone_shallow());
  auto request = _receive_worker.push([this, copy_tensor, src_rank]{
    this->_receive(*copy_tensor, src_rank);
  });
  return new DataChannelGloo::RequestGloo(std::move(request));
}


void DataChannelGloo::barrier(THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP(_)

  auto ret = GlooCache::get().getAlgorithm<DataOperation::BARRIER, void>(
    group_id, _groups.at(group_id), getStore<DataOperation::BARRIER>(group_id));
  std::get<0>(ret)->run();
}


THDGroup DataChannelGloo::newGroup(const std::vector<rank_type>& ranks) {
  auto new_group = DataChannel::Group(ranks, _num_processes - 1);
  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());

  _groups.insert({new_group_id, new_group});
}


void DataChannelGloo::_send(const Scalar& data, rank_type dst_rank) {
  std::unique_ptr<Scalar> data_copy(data.clone());
  auto ctx = GlooCache::get().getSharedContext<DataOperation::SEND>(
    _groups.at(THDGroupWORLD),
    getStore<DataOperation::SEND>(THDGroupWORLD)
  );
  auto& pair = ctx->getPair(dst_rank);
  pair->createSendBuffer(ctx->nextSlot(), data_copy->data(), data_copy->elementSize())->waitSend();
}


void DataChannelGloo::_send(thpp::Tensor& data, rank_type dst_rank) {
  auto ctx = GlooCache::get().getSharedContext<DataOperation::SEND>(
    _groups.at(THDGroupWORLD),
    getStore<DataOperation::SEND>(THDGroupWORLD)
  );
  auto& pair = ctx->getPair(dst_rank);
  uint64_t tensor_bytes = data.elementSize() * data.numel();
  pair->createSendBuffer(ctx->nextSlot(), data.data(), tensor_bytes)->waitSend();
}


void DataChannelGloo::_receive(Scalar& data, rank_type src_rank) {
  auto ctx = GlooCache::get().getSharedContext<DataOperation::SEND>(
    _groups.at(THDGroupWORLD),
    getStore<DataOperation::SEND>(THDGroupWORLD)
  );
  auto& pair = ctx->getPair(src_rank);
  pair->createRecvBuffer(ctx->nextSlot(), data.data(), data.elementSize())->waitRecv();
}


void DataChannelGloo::_receive(thpp::Tensor& data, rank_type src_rank) {
  auto ctx = GlooCache::get().getSharedContext<DataOperation::SEND>(
    _groups.at(THDGroupWORLD),
    getStore<DataOperation::SEND>(THDGroupWORLD)
  );
  auto& pair = ctx->getPair(src_rank);
  uint64_t tensor_bytes = data.elementSize() * data.numel();
  pair->createRecvBuffer(ctx->nextSlot(), data.data(), tensor_bytes)->waitRecv();
}

} // namespace thd

