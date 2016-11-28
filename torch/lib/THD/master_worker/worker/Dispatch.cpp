#include <TH/THStorage.h>
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../../process_group/General.hpp"
#include "../../base/Tensor.hpp"
#include "../../base/Traits.hpp"
#include "../../base/storages/THStorage.hpp"
#include "../../base/tensors/THTensor.hpp"
#include "../common/Functions.hpp"
#include "../common/RPC.hpp"
#include "../master/Master.hpp"
#include "Worker.hpp"

namespace thd {
namespace worker {

namespace detail {

Tensor* unpackRetrieveTensor(rpc::RPCMessage& message) {
  return workerTensors.at(unpackTensor(message)).get();
}

Storage* unpackRetrieveStorage(rpc::RPCMessage& message) {
  return workerStorages.at(unpackStorage(message)).get();
}

static std::unique_ptr<Tensor> createTensor(Type type) {
  if (type == Type::UCHAR)
    return std::unique_ptr<Tensor>(new THTensor<unsigned char>());
  else if (type == Type::CHAR)
    return std::unique_ptr<Tensor>(new THTensor<char>());
  else if (type == Type::SHORT)
    return std::unique_ptr<Tensor>(new THTensor<short>());
  else if (type == Type::INT)
    return std::unique_ptr<Tensor>(new THTensor<int>());
  else if (type == Type::LONG)
    return std::unique_ptr<Tensor>(new THTensor<long>());
  else if (type == Type::FLOAT)
    return std::unique_ptr<Tensor>(new THTensor<float>());
  else if (type == Type::DOUBLE)
    return std::unique_ptr<Tensor>(new THTensor<double>());
  throw std::invalid_argument("passed characted doesn't represent a tensor type");
}

static std::unique_ptr<Storage> createStorage(Type type) {
  if (type == Type::UCHAR)
    return std::unique_ptr<Storage>(new THStorage<unsigned char>());
  else if (type == Type::CHAR)
    return std::unique_ptr<Storage>(new THStorage<char>());
  else if (type == Type::SHORT)
    return std::unique_ptr<Storage>(new THStorage<short>());
  else if (type == Type::INT)
    return std::unique_ptr<Storage>(new THStorage<int>());
  else if (type == Type::LONG)
    return std::unique_ptr<Storage>(new THStorage<long>());
  else if (type == Type::FLOAT)
    return std::unique_ptr<Storage>(new THStorage<float>());
  else if (type == Type::DOUBLE)
    return std::unique_ptr<Storage>(new THStorage<double>());
  throw std::invalid_argument("passed characted doesn't represent a storage type");
}

static std::unique_ptr<Storage> createStorage(Type type, std::size_t size) {
  std::unique_ptr<Storage> storage = createStorage(type);
  storage->resize(size);
  return storage;
}


static void finalize(rpc::RPCMessage& raw_message) {
  if (raw_message.remaining() > 0)
    throw std::invalid_argument("message is too long");
}

#include "dispatch/Storage.cpp"
#include "dispatch/Tensor.cpp"
#include "dispatch/Communication.cpp"

using dispatch_fn = void (*)(rpc::RPCMessage&);
using Functions = thd::Functions;


static const std::unordered_map<std::uint16_t, dispatch_fn> functions {
    {Functions::construct, construct},
    {Functions::constructWithSize, constructWithSize},
    {Functions::add, add},
    {Functions::free, free},
    {Functions::storageConstruct, storageConstruct},
    {Functions::storageConstructWithSize, storageConstructWithSize},
    {Functions::storageFree, storageFree},
    {Functions::sendTensor, sendTensor},
    {Functions::sendStorage, sendStorage}
};

} // namespace detail

std::string execute(std::unique_ptr<rpc::RPCMessage> raw_message_ptr) {
  try {
    // TODO: unify the function id type (it's in rpc:: now)
    auto &raw_message = *raw_message_ptr;
    uint16_t fid = rpc::unpackFunctionId(raw_message);
    auto iter = detail::functions.find(fid);
    if (iter != detail::functions.end())
      (*iter->second)(raw_message);
    else
      throw std::invalid_argument(std::string("invalid function id: ") + std::to_string(fid));
    return std::string();
  } catch(std::exception& e) {
    return std::string(e.what());
  }
}

} // namespace worker
} // namespace thd
