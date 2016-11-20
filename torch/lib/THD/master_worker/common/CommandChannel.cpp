#include "CommandChannel.hpp"

#include "../../base/ChannelEnvVars.hpp"
#include "ByteArray.hpp"
#include "RPC.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include <zmq.hpp>

namespace thd {

namespace {
////////////////////////////////////////////////////////////////////////////////

constexpr int MAX_ADDR_SIZE = 1024;

void sendMessage(std::unique_ptr<rpc::RPCMessage> msg, zmq::socket_t& socket) {
  socket.send(zmq::message_t(msg.get()->bytes().data(),
                             msg.get()->bytes().length(),
                             rpc::RPCMessage::freeMessage,
                             msg.release()));
}

std::unique_ptr<rpc::RPCMessage> recvMessage(zmq::socket_t& socket) {
  zmq::message_t zmsg;
  if (socket.recv(&zmsg, ZMQ_DONTWAIT) == false)
    return nullptr;
  else {
    // XXX: Excesive copying here! I'm not sure how to avoid it.
    return std::unique_ptr<rpc::RPCMessage>(new rpc::RPCMessage(
        rpc::ByteArray::fromData(zmsg.data<char>(), zmsg.size())));
  }
}

void connectWorker(zmq::socket_t& socket,
                   zmq::context_t& context,
                   zmq::socket_type type,
                   const std::string& endpoint) {
  socket = zmq::socket_t(context, type);
  socket.connect(endpoint);
}

void composeEndpoint(std::string& endpoint,
                     const std::string& peer_addr,
                     const std::string& port) {
  endpoint = std::string("tcp://" + peer_addr + ":" + port);
}

// Instead of specifying a port, let the opertating system assign a free port.
std::string bindPort(zmq::socket_t& socket, std::string& endpoint) {
  rpc::ByteArray addr(MAX_ADDR_SIZE);
  size_t size = addr.length() - 1;
  socket.bind("tcp://*:*");
  socket.getsockopt(ZMQ_LAST_ENDPOINT, addr.data(), &size);
  addr.data()[size] = '\0';
  endpoint = std::string(addr.data());
  return endpoint.substr(endpoint.rfind(':') + 1);
}

void unpackInitMessage(const zmq::message_t& msg,
                       int& rank,
                       std::string& pull_port,
                       std::string& push_port) {
    std::string data(msg.data<char>(), msg.size());
    size_t first_separator = data.find(';');
    size_t second_separator = data.find(';', first_separator + 1);
    rank = std::stoi(data.substr(0, first_separator));
    size_t length = second_separator - (first_separator + 1);
    pull_port = data.substr(first_separator + 1, length);
    push_port = data.substr(second_separator + 1);
}

////////////////////////////////////////////////////////////////////////////////
} // anonymous namespace

// TODO: Validate this environmental variable.
MasterCommandChannel::MasterCommandChannel()
  : _context()
  , _rank(0)
  , _world_size(std::stoi(std::getenv(WORLD_SIZE_ENV)))
  , _pull_sockets()
  , _push_sockets()
  , _pull_endpoints(_world_size)
  , _push_endpoints(_world_size)
{
  for (int i = 0; i < _world_size; ++i) {
    _pull_sockets.emplace_back(_context, zmq::socket_type::pull);
    _push_sockets.emplace_back(_context, zmq::socket_type::push);
  }
  const std::string initendpoint(std::string("tcp://0.0.0.0:") +
                                 std::getenv(MASTER_PORT_ENV));
  zmq::socket_t initsocket(_context, zmq::socket_type::pull);
  initsocket.bind(initendpoint);

  for (int wi = 1; wi < _world_size; ++wi) {
    zmq::message_t msg;
    initsocket.recv(&msg); // Blocking.

    int worker_rank;
    std::string pull_port;
    std::string push_port;
    unpackInitMessage(msg, worker_rank, pull_port, push_port);

    composeEndpoint(_pull_endpoints.at(worker_rank),
                    msg.gets("Peer-Address"),
                    pull_port);
    connectWorker(_pull_sockets.at(worker_rank),
                  _context,
                  zmq::socket_type::pull,
                  _pull_endpoints[worker_rank]);
    composeEndpoint(_push_endpoints.at(worker_rank),
                    msg.gets("Peer-Address"),
                    push_port);
    connectWorker(_push_sockets.at(worker_rank),
                  _context,
                  zmq::socket_type::push,
                  _push_endpoints[worker_rank]);
  }

  initsocket.unbind(initendpoint);
}

MasterCommandChannel::~MasterCommandChannel() {
  for (int wi = 1; wi < _world_size; ++wi) {
    _pull_sockets.at(wi).disconnect(_pull_endpoints.at(wi));
    _push_sockets.at(wi).disconnect(_push_endpoints.at(wi));
  }
}

void MasterCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg,
                                       int rank) {
  thd::sendMessage(std::move(msg), _push_sockets.at(rank));
}

std::unique_ptr<rpc::RPCMessage> MasterCommandChannel::recvMessage(int rank) {
  return thd::recvMessage(_pull_sockets.at(rank));
}

// TODO: Validate this environmental variable.
WorkerCommandChannel::WorkerCommandChannel(int rank)
  : _context()
  , _pull_socket(_context, zmq::socket_type::pull)
  , _push_socket(_context, zmq::socket_type::push)
  , _rank(rank)
{
  const std::string initendpoint(std::string("tcp://") +
                                 std::getenv(MASTER_ADDR_ENV));
  zmq::socket_t initsocket(_context, zmq::socket_type::push);
  initsocket.connect(initendpoint);

  std::string pull_port = bindPort(_pull_socket, _pull_endpoint);
  std::string push_port = bindPort(_push_socket, _push_endpoint);

  rpc::ByteArray arr;
  std::string str_rank = std::to_string(_rank);
  arr.append(str_rank.c_str(), str_rank.size());
  arr.append(";", 1);
  arr.append(push_port.c_str(), push_port.size());
  arr.append(";", 1);
  arr.append(pull_port.c_str(), pull_port.size());

  thd::sendMessage(std::unique_ptr<rpc::RPCMessage>(new rpc::RPCMessage(arr)),
                   initsocket);

  initsocket.disconnect(initendpoint);
}

WorkerCommandChannel::~WorkerCommandChannel() {
  _pull_socket.unbind(_pull_endpoint);
  _push_socket.unbind(_push_endpoint);
}

void WorkerCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg) {
  thd::sendMessage(std::move(msg), _push_socket);
}

std::unique_ptr<rpc::RPCMessage> WorkerCommandChannel::recvMessage() {
  return thd::recvMessage(_pull_socket);
}

} // namespace thd
