#include "DataChannelGloo.hpp"
#include "DataChannelUtils.hpp"
#include "GlooCache.hpp"
#include "Store.hpp"

#include "gloo/transport/tcp/device.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>



#define RETURN_IF_NOT_IN_GROUP                                                \
  {                                                                           \
    bool exists;                                                              \
    std::tie(std::ignore, exists) = _groups.at(group_id).getGroupRank(_rank); \
    if (!exists) return;                                                      \
  }


// TODO: gloo uses stdint types for integral values and there's some weird template
// magic going on that mangles names so that they don't always match the types
// below. Only float and double are left enabled for now, because they're most
// useful and unambiguous.
#define GENERATE_ALL_TYPES(type, func, args...)                               \
  switch (type) {                                                             \
    case ::thpp::Type::FLOAT: func<float>(args); break;                       \
    case ::thpp::Type::DOUBLE: func<double>(args); break;                     \
    /* case ::thpp::Type::CHAR: func<char>(args); break; */                   \
    /* case ::thpp::Type::UCHAR: func<unsigned char>(args); break; */         \
    /* case ::thpp::Type::HALF: func<float>(args); break; */                  \
    /* case ::thpp::Type::SHORT: func<short>(args); break; */                 \
    /* case ::thpp::Type::USHORT: func<unsigned short>(args); break; */       \
    /* case ::thpp::Type::INT: func<int>(args); break; */                     \
    /* case ::thpp::Type::UINT: func<unsigned int>(args); break; */           \
    /* case ::thpp::Type::LONG: func<long>(args); break; */                   \
    /* case ::thpp::Type::ULONG: func<unsigned long>(args); break; */         \
    /* case ::thpp::Type::LONG_LONG: func<long long>(args); break; */         \
    /* case ::thpp::Type::ULONG_LONG: func<unsigned long long>(args); break; */ \
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


DataChannelGloo::DataChannelGloo(InitMethod::Config config)
  : _rank(config.rank)
  , _listen_socket(-1)
  , _store(nullptr)
  , _cache(nullptr)
{
  if (_rank == 0) {
    _num_processes = config.master.world_size;
  }

  // Default options listen on this host's name.
  // NOTE: when hostname has bad configuration in `/etc/hosts` processes
  // will not connect to each other.
  ::gloo::transport::tcp::attr attr;
  _device = ::gloo::transport::tcp::CreateDevice(attr);

  if (_rank == 0) {
    _addr = "localhost";
    _port = config.master.listen_port;
    _listen_socket = config.master.listen_socket;
  } else {
    _addr = config.worker.address;
    _port = config.worker.listen_port;
  }
}


DataChannelGloo::~DataChannelGloo() {}


bool DataChannelGloo::init() {
  _store = std::unique_ptr<::gloo::rendezvous::Store>(
    new Store(_rank, _listen_socket, _addr, _port, _num_processes)
  );
  _cache = std::unique_ptr<GlooCache>(new GlooCache(_rank, _device, _store));

  if (_rank == 0) {
    auto num_proc_str = std::to_string(_num_processes);
    _store->set("world_size",
            std::vector<char>(num_proc_str.begin(), num_proc_str.end()));
  } else {
    auto world_size = _store->get("world_size");
    _num_processes = std::atoll(
            std::string(world_size.begin(), world_size.end()).c_str());
  }

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


rank_type DataChannelGloo::getRank() {
  return _rank;
}


rank_type DataChannelGloo::getNumProcesses() {
  return _num_processes;
}


template<typename T>
void DataChannelGloo::allGatherT(std::vector<thpp::Tensor*>& output,
                                 thpp::Tensor& input, THDGroup group_id) {
  auto input_device = getDeviceType(input);
  for (auto& out : output) {
    if (input_device != getDeviceType(*out)) {
      throw std::runtime_error("allGather got input and output on different devices");
    }
  }
  std::uint64_t tensor_bytes = input.elementSize() * input.numel();
  std::uint64_t all_tensor_bytes = tensor_bytes * output.size();
  auto ret = _cache->getAlgorithm<CollectiveType::ALL_GATHER, T>(
    group_id, _groups.at(group_id), input_device, tensor_bytes, all_tensor_bytes, input.numel());

  std::memcpy(GlooCache::input_buffer(ret).get(), input.data(), tensor_bytes);

  {
    std::lock_guard<std::mutex> lock(*GlooCache::mutex(ret));
    GlooCache::algorithm(ret)->run();
  }

  for (std::size_t i = 0; i < output.size(); i++) {
    std::memcpy(output.at(i)->data(),
                GlooCache::output_buffer(ret).get() + (i * tensor_bytes),
                tensor_bytes);
  }
}

void DataChannelGloo::allGather(std::vector<thpp::Tensor*>& output,
                                thpp::Tensor& input, THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP

  if (output.size() != _groups.at(group_id).size())
    throw std::logic_error("allGather: number of output tensors and group size does not match");

  for (auto out_tensor : output)
    assertSameSizeAndType(*out_tensor, input, "allGather");

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
  throw std::runtime_error("DataChannelGloo does not support scatter");
}


template<typename T>
void DataChannelGloo::allReduceT(thpp::Tensor& t, THDReduceOp operation,
                                 THDGroup group_id) {
  std::uint64_t tensor_bytes = t.elementSize() * t.numel();
  auto ret = _cache->getAlgorithm<CollectiveType::ALL_REDUCE, T>(
    group_id, _groups.at(group_id), getDeviceType(t), tensor_bytes, t.numel(), operation);

  GlooCache::memcpy_input(ret, t);
  {
    std::lock_guard<std::mutex> lock(*GlooCache::mutex(ret));
    GlooCache::algorithm(ret)->run();
  }
  GlooCache::memcpy_output(ret, t);
}

void DataChannelGloo::allReduce(thpp::Tensor& data, THDReduceOp operation,
                                THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP
  GENERATE_ALL_TYPES(data.type(), allReduceT, data, operation, group_id)
}


// XXX: `reduce` is not supported by Gloo yet.
void DataChannelGloo::reduce(thpp::Tensor& data, THDReduceOp operation,
                             rank_type dst_rank, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo does not support reduce");
}


template<typename T>
void DataChannelGloo::broadcastT(thpp::Tensor& data, rank_type src_rank,
                                 THDGroup group_id) {
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  auto ret = _cache->getAlgorithm<CollectiveType::BROADCAST, T>(
    group_id, _groups.at(group_id), getDeviceType(data), tensor_bytes, data.numel(),
    _groups.at(group_id).mustGetGroupRank(src_rank));

  if (_rank == src_rank) {
    GlooCache::memcpy_input(ret, data);
  }

  {
    std::lock_guard<std::mutex> lock(*GlooCache::mutex(ret));
    GlooCache::algorithm(ret)->run();
  }

  if (_rank != src_rank) {
    GlooCache::memcpy_output(ret, data);
  }
}


void DataChannelGloo::broadcast(thpp::Tensor& data, rank_type src_rank,
                                THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP
  GENERATE_ALL_TYPES(data.type(), broadcastT, data, src_rank, group_id)
}


void DataChannelGloo::send(const Scalar& data, rank_type dst_rank) {
  throw std::runtime_error("DataChannelGloo does not support send");
}


void DataChannelGloo::send(thpp::Tensor& data, rank_type dst_rank) {
  throw std::runtime_error("DataChannelGloo does not support send");
}


void DataChannelGloo::receive(Scalar& data, rank_type src_rank) {
  throw std::runtime_error("DataChannelGloo does not support receive");
}


void DataChannelGloo::receive(thpp::Tensor& data) {
  throw std::runtime_error("DataChannelGloo does not support receive from any source");
}


void DataChannelGloo::receive(thpp::Tensor& data, rank_type src_rank) {
  throw std::runtime_error("DataChannelGloo does not support receive");
}


auto DataChannelGloo::isend(thpp::Tensor& data, rank_type dst_rank) -> RequestGloo* {
  throw std::runtime_error("DataChannelGloo does not support isend");
}


auto DataChannelGloo::ireceive(thpp::Tensor& data, rank_type src_rank) -> RequestGloo* {
  throw std::runtime_error("DataChannelGloo does not support ireceive");
}


void DataChannelGloo::barrier(THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP
  auto ret = _cache->getAlgorithm<CollectiveType::BARRIER, void>(
    group_id, _groups.at(group_id));
  {
    std::lock_guard<std::mutex> lock(*GlooCache::mutex(ret));
    GlooCache::algorithm(ret)->run();
  }
}


THDGroup DataChannelGloo::newGroup(const std::vector<rank_type>& ranks) {
  auto new_group = DataChannel::Group(ranks, _num_processes - 1);
  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());

  _groups.insert({new_group_id, new_group});
  return new_group_id;
}

} // namespace thd

