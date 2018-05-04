#include "DataChannelGloo.hpp"
#include "DataChannelUtils.hpp"
#include "GlooCache.hpp"
#include "Store.hpp"

#if defined(WITH_GLOO_IBVERBS) && WITH_GLOO_IBVERBS
#include "gloo/transport/ibverbs/device.h"
#endif

#include "gloo/transport/tcp/device.h"

#include <algorithm>
#include <unistd.h>
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
    case ::at::ScalarType::Float: func<float>(args); break;                   \
    case ::at::ScalarType::Double: func<double>(args); break;                 \
    case ::at::ScalarType::Half: func<gloo::float16>(args); break;            \
    case ::at::ScalarType::Char: func<int8_t>(args); break;                   \
    case ::at::ScalarType::Byte: func<uint8_t>(args); break;                  \
    case ::at::ScalarType::Int: func<int32_t>(args); break;                   \
    case ::at::ScalarType::Long: func<int64_t>(args); break;                  \
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


DataChannelGloo::Group::Group(const std::string& addr,
                              port_type port,
                              std::vector<rank_type> ranks,
                              rank_type max_rank,
                              int store_socket)
  : DataChannel::Group(std::move(ranks), max_rank)
  , _store(new Store(addr, port, store_socket)) {}

DataChannelGloo::DataChannelGloo(InitMethod::Config config)
  : _rank(config.rank)
  , _listen_socket(-1)
  , _cache(nullptr)
{
  _num_processes = config.world_size;

#if defined(WITH_GLOO_IBVERBS) && WITH_GLOO_IBVERBS

  // This helper function automatically detects the IB device in the system
  auto ibDeviceNames = ::gloo::transport::ibverbs::getDeviceNames();

  // If there are IB devices, we will use IB
  if (!ibDeviceNames.empty()) {
    // Currently, gloo only supports a single IB device and will use the first
    auto ibDeviceToUse = ibDeviceNames[0];

    ::gloo::transport::ibverbs::attr attr = {
      .name = ibDeviceToUse,
      .port = 1,
      .index = 0,
    };

    _deviceList.push_back(::gloo::transport::ibverbs::CreateDevice(attr));

  // Otherwise, fallback to use TCP instead
  } else

#endif

  {
    // Default options listen on this host's name.
    // NOTE: when hostname has bad configuration in `/etc/hosts` processes
    // will not connect to each other.
    ::gloo::transport::tcp::attr attr(config.public_address.c_str());
    _deviceList.push_back(::gloo::transport::tcp::CreateDevice(attr));
  }

  if (_rank == 0) {
    _addr = "localhost";
    _port = config.master.listen_port;
    _listen_socket = config.master.listen_socket;
  } else {
    _addr = config.worker.master_addr;
    _port = config.worker.master_port;
  }
}


DataChannelGloo::~DataChannelGloo() {
  if (_listen_socket != -1) {
    ::close(_listen_socket);
  }
}

void DataChannelGloo::destroy() {}

bool DataChannelGloo::init() {
  _cache = std::unique_ptr<GlooCache>(new GlooCache(_rank, _deviceList));

  std::vector<rank_type> ranks;
  ranks.reserve(_num_processes);
  for (rank_type rank = 0; rank < _num_processes; ++rank)
    ranks.push_back(rank);

  _groups.insert({
    THDGroupWORLD,
    Group(_addr, _port, ranks, _num_processes - 1, _rank == 0 ? _listen_socket : Store::CLIENT_ONLY)
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
void DataChannelGloo::allGatherT(std::vector<at::Tensor>& output,
                                 at::Tensor& input, THDGroup group_id) {
  auto input_device = getDeviceType(input);
  for (auto& out : output) {
    if (input_device != getDeviceType(out)) {
      throw std::runtime_error("allGather got input and output on different devices");
    }
  }
  std::uint64_t tensor_bytes = input.type().elementSizeInBytes() * input.numel();
  std::uint64_t all_tensor_bytes = tensor_bytes * output.size();
  auto ret = _cache->getAlgorithm<CollectiveType::ALL_GATHER, T>(
    group_id, _groups.at(group_id), input_device, tensor_bytes, all_tensor_bytes, input.numel());


  {
    std::lock_guard<std::mutex> lock(*GlooCache::mutex(ret));
    std::memcpy(GlooCache::input_buffer(ret).get(), input.data_ptr(), tensor_bytes);
    GlooCache::algorithm(ret)->run();
    for (std::size_t i = 0; i < output.size(); i++) {
      std::memcpy(output.at(i).data_ptr(),
                  GlooCache::output_buffer(ret).get() + (i * tensor_bytes),
                  tensor_bytes);
    }
  }

}

void DataChannelGloo::allGather(std::vector<at::Tensor>& output,
                                at::Tensor& input, THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP

  if (output.size() != _groups.at(group_id).size())
    throw std::logic_error("allGather: number of output tensors and group size does not match");

  for (auto out_tensor : output)
    assertSameSizeAndType(out_tensor, input, "allGather");

  GENERATE_ALL_TYPES(input.type().scalarType(), allGatherT, output, input, group_id)
}


// XXX: `gather` is not supported by Gloo yet.
void DataChannelGloo::gather(std::vector<at::Tensor>& output,
                             at::Tensor& input, rank_type dst_rank,
                             THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo doesn't support gather");
}


// XXX: `scatter` is not supported by Gloo yet.
void DataChannelGloo::scatter(std::vector<at::Tensor>& input,
                              at::Tensor& output,
                              rank_type src_rank, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo does not support scatter");
}


template<typename T>
void DataChannelGloo::allReduceT(at::Tensor& t, THDReduceOp operation,
                                 THDGroup group_id) {
  std::uint64_t tensor_bytes = t.type().elementSizeInBytes() * t.numel();
  auto ret = _cache->getAlgorithm<CollectiveType::ALL_REDUCE, T>(
    group_id, _groups.at(group_id), getDeviceType(t), tensor_bytes, t.numel(), operation);

  {
    std::lock_guard<std::mutex> lock(*GlooCache::mutex(ret));
    GlooCache::memcpy_input(ret, t);
    GlooCache::algorithm(ret)->run();
    GlooCache::memcpy_output(ret, t);
  }
}

void DataChannelGloo::allReduce(at::Tensor& data, THDReduceOp operation,
                                THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP
  GENERATE_ALL_TYPES(data.type().scalarType(), allReduceT, data, operation, group_id)
}


// XXX: `reduce` is not supported by Gloo yet.
void DataChannelGloo::reduce(at::Tensor& data, THDReduceOp operation,
                             rank_type dst_rank, THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo does not support reduce");
}


template<typename T>
void DataChannelGloo::broadcastT(at::Tensor& data, rank_type src_rank,
                                 THDGroup group_id) {
  std::uint64_t tensor_bytes = data.type().elementSizeInBytes() * data.numel();
  auto ret = _cache->getAlgorithm<CollectiveType::BROADCAST, T>(
    group_id, _groups.at(group_id), getDeviceType(data), tensor_bytes, data.numel(),
    _groups.at(group_id).mustGetGroupRank(src_rank));

  {
    std::lock_guard<std::mutex> lock(*GlooCache::mutex(ret));
    if (_rank == src_rank) {
      GlooCache::memcpy_input(ret, data);
    }

    GlooCache::algorithm(ret)->run();

    if (_rank != src_rank) {
      GlooCache::memcpy_output(ret, data);
    }
  }

}


void DataChannelGloo::broadcast(at::Tensor& data, rank_type src_rank,
                                THDGroup group_id) {
  RETURN_IF_NOT_IN_GROUP
  GENERATE_ALL_TYPES(data.type().scalarType(), broadcastT, data, src_rank, group_id)
}


void DataChannelGloo::send(Scalar& data, rank_type dst_rank) {
  throw std::runtime_error("DataChannelGloo does not support send");
}


void DataChannelGloo::send(at::Tensor& data, rank_type dst_rank) {
  throw std::runtime_error("DataChannelGloo does not support send");
}


void DataChannelGloo::receive(Scalar& data, rank_type src_rank) {
  throw std::runtime_error("DataChannelGloo does not support receive");
}


rank_type DataChannelGloo::receive(at::Tensor& data) {
  throw std::runtime_error("DataChannelGloo does not support receive from any source");
}


void DataChannelGloo::receive(at::Tensor& data, rank_type src_rank) {
  throw std::runtime_error("DataChannelGloo does not support receive");
}


auto DataChannelGloo::isend(at::Tensor& data, rank_type dst_rank) -> RequestGloo* {
  throw std::runtime_error("DataChannelGloo does not support isend");
}


auto DataChannelGloo::ireceive(at::Tensor& data, rank_type src_rank) -> RequestGloo* {
  throw std::runtime_error("DataChannelGloo does not support ireceive");
}


void DataChannelGloo::allReduce(std::vector<at::Tensor>& data,
                                THDReduceOp operation,
                                THDGroup groupId) {

  throw std::runtime_error("DataChannelGloo does not support mult-GPU cross "
                           "node allreduce");
}


void DataChannelGloo::allGather(std::vector<at::Tensor>& output,
                                std::vector<at::Tensor>& input,
                                THDGroup groupId) {

  throw std::runtime_error("DataChannelGloo does not support mult-GPU cross "
                           "node allgather");
}


void DataChannelGloo::reduce(std::vector<at::Tensor>& data,
                             THDReduceOp operation,
                             rank_type dstRank,
                             THDGroup groupId) {

  throw std::runtime_error("DataChannelGloo does not support mult-GPU cross "
                           "node reduce");
}


void DataChannelGloo::broadcast(std::vector<at::Tensor>& data,
                                rank_type srcRank,
                                THDGroup groupId) {

  throw std::runtime_error("DataChannelGloo does not support mult-GPU cross "
                           "node broadcast");
}


void DataChannelGloo::clearGroupCache(THDGroup group_id) {
  throw std::runtime_error("DataChannelGloo does not support clear "
                           "group cache");
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
  auto new_group = DataChannelGloo::Group(_addr, _port, ranks, _num_processes - 1, Store::CLIENT_ONLY);
  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());

  _groups.insert({new_group_id, new_group});
  return new_group_id;
}

} // namespace thd

