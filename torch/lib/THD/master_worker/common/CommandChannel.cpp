#include "CommandChannel.hpp"

#include "../../base/ChannelEnvVars.hpp"
#include "ByteArray.hpp"
#include "RPC.hpp"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

#include <asio.hpp>

namespace thd {

namespace {
////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<rpc::ByteArray> packInitMessage(
    const std::string& rank,
    const std::string& port
) {
  std::unique_ptr<rpc::ByteArray> arr_ptr(new rpc::ByteArray);

  arr_ptr.get()->append(std::string(rank + ";" + port).c_str(),
          rank.size() + 1 + port.size());

  return std::move(arr_ptr);
}

void unpackInitMessage(
    const char* msg,
    size_t length,
    int& rank,
    std::string& port
) {
  std::string data(msg, length);
  std::size_t separator = data.find(';');
  rank = std::stoi(data.substr(0, separator));
  port = data.substr(separator + 1, length);
}

void acceptInitConnection(
    asio::ip::tcp::acceptor& acceptor,
    asio::ip::tcp::socket& socket,
    asio::ip::tcp::endpoint& endpoint
) {
  asio::error_code ec;
  acceptor.accept(socket, endpoint, ec);
  asio::detail::throw_error(ec, "master failed to accept an initial "
      "connection");
}

asio::ip::basic_resolver_iterator<asio::ip::tcp> receiveInitMessage(
    asio::ip::tcp::resolver& resolver,
    asio::ip::tcp::socket& socket,
    const asio::ip::tcp::endpoint& endpoint,
    int& worker_rank
) {
  rpc::RPCMessage::size_type length;
  // Read the length of the incoming message.
  asio::read(socket, asio::buffer(&length, sizeof(length)));

  char* data = new char[length];
  // Read the incoming message.
  asio::read(socket, asio::buffer(data, length));

  std::string port;
  // Unpack the rank and the port to connec to from the message.
  unpackInitMessage(data, length, worker_rank, port);

  // Return an iterator to a list of endpoints to connect to.
  return resolver.resolve({endpoint.address().to_string(), port});
}

void sendMessage(
    std::unique_ptr<rpc::RPCMessage> msg,
    asio::ip::tcp::socket& socket
) {
  size_t length = msg.get()->bytes().length();
  asio::write(socket, asio::buffer(&length, sizeof(length)));
  asio::write(socket, asio::buffer(msg.get()->bytes().data(), length));
}

std::unique_ptr<rpc::RPCMessage> recvMessage(
    asio::ip::tcp::socket& socket
) {
  rpc::RPCMessage::size_type length;
  asio::read(socket, asio::buffer(&length, sizeof(length)));
  char* data = new char[length];
  asio::read(socket, asio::buffer(data, length));

  return std::unique_ptr<rpc::RPCMessage>(new rpc::RPCMessage(data, length));
}

const char* get_env(const char* env) {
  const char* value = std::getenv(env);
  if (value == nullptr) {
    throw std::logic_error(std::string("") + "failed to read the " + env +
        " environmental variable; maybe you forgot to set it properly?");
  }
  return value;
}

////////////////////////////////////////////////////////////////////////////////
} // anonymous namespace

MasterCommandChannel::MasterCommandChannel()
  : _io()
  , _rank(0)
{
  unsigned short init_port;
  _load_env(init_port, _world_size);

  _sockets.reserve(_world_size);
  for (int i = 0; i < _world_size; ++i)
    _sockets.emplace_back(_io);

  asio::ip::tcp::acceptor init_acceptor(_io,
      asio::ip::tcp::endpoint(asio::ip::tcp::v6(), init_port));
  asio::ip::tcp::resolver init_resolver(_io);
  asio::ip::tcp::socket init_socket(_io);

  for (int i = 1; i < _world_size; ++i) {
    asio::ip::tcp::endpoint worker_endpoint;
    acceptInitConnection(init_acceptor, init_socket, worker_endpoint);

    int worker_rank;
    auto final_endpoint = receiveInitMessage(init_resolver, init_socket,
        worker_endpoint, worker_rank);
    try {
      asio::connect(_sockets.at(worker_rank), final_endpoint);
    } catch (const asio::system_error& e) {
      throw std::runtime_error(std::string("") + "master failed to " +
          "establish the connection with worker " +
          std::to_string(worker_rank) + "; specifically: " + e.what());
    }
    init_socket.close();
  }
}

MasterCommandChannel::~MasterCommandChannel() {}

void MasterCommandChannel::sendMessage(
    std::unique_ptr<rpc::RPCMessage> msg,
    int rank
) {
  thd::sendMessage(std::move(msg), _sockets.at(rank));
}

std::unique_ptr<rpc::RPCMessage> MasterCommandChannel::recvMessage(int rank) {
  return thd::recvMessage(_sockets.at(rank));
}

void MasterCommandChannel::_load_env(unsigned short& port, int& world_size) {
  try {
    unsigned long value = std::stoul(get_env(MASTER_PORT_ENV));
    if (value > USHRT_MAX) {
      throw std::logic_error("the number representing the port is out of"
          "range");
    }
    port = value;
  } catch (const std::exception& e) {
    throw std::logic_error(std::string("") + "failed to convert the " +
        MASTER_PORT_ENV + " environmental variable to unsigned short (port "
        "number); specifically: " + e.what());
  }

  try {
    world_size = std::stoi(get_env(WORLD_SIZE_ENV));
  } catch (const std::exception& e) {
    throw std::logic_error(std::string("") + "failed to convert the " +
        WORLD_SIZE_ENV + " environmental variable to int: " + e.what());
  }
}

// XXX: asio::ip::address::from_string has been deprecated in favour of
//      asio::ip::address::make_address in v1.11.0. The version downloadable
//      with brew on macOS is v1.10.8.
WorkerCommandChannel::WorkerCommandChannel(int rank)
  : _rank(rank)
  , _io()
  , _socket(_io)
{
  asio::error_code ec;
  std::string master_ip_addr;
  unsigned short master_port;

  _load_env(master_ip_addr, master_port);

  asio::ip::tcp::acceptor init_acceptor(_io, asio::ip::tcp::endpoint(
        asio::ip::tcp::v6(), 0));

  asio::ip::tcp::resolver init_resolver(_io);
  asio::ip::tcp::resolver::query init_query(master_ip_addr,
      std::to_string(master_port),
      asio::ip::resolver_query_base::flags::v4_mapped |
      asio::ip::resolver_query_base::flags::numeric_service);
  auto init_endpoint = init_resolver.resolve(init_query);

  // Connect to the master.
  asio::ip::tcp::socket init_socket(_io);
  do {
    asio::connect(init_socket, init_endpoint, ec);
  } while (ec == asio::error::basic_errors::connection_refused);
	asio::detail::throw_error(ec, "failed to connect to master");

  // Get a local port.
  auto local_port = init_acceptor.local_endpoint(ec).port();
  asio::detail::throw_error(ec, "worker failed to get a local port");

  // Prepare the init message.
  auto arr = packInitMessage(std::to_string(_rank), std::to_string(local_port));

  // Send the init message.
  try {
    auto msg_length = arr.get()->length();
    asio::write(init_socket, asio::buffer(&msg_length, sizeof(msg_length)));
    asio::write(init_socket, asio::buffer(arr.get()->data(), msg_length));

    asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v6(), local_port);
    init_acceptor.accept(_socket, endpoint, ec);
    asio::detail::throw_error(ec, "worker failed to accept the connection from "
        "master");
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string() + "unable to establish the " +
        "connection with master: " + e.what());
  }
}

WorkerCommandChannel::~WorkerCommandChannel() {}

void WorkerCommandChannel::sendMessage(std::unique_ptr<rpc::RPCMessage> msg) {
  thd::sendMessage(std::move(msg), _socket);
}

std::unique_ptr<rpc::RPCMessage> WorkerCommandChannel::recvMessage() {
  return thd::recvMessage(_socket);
}

void WorkerCommandChannel::_load_env(
    std::string& ip_addr,
    unsigned short& port
) {
  const std::string addr(get_env(MASTER_ADDR_ENV));
  auto separator_pos = addr.rfind(':');
  if (separator_pos == std::string::npos) {
    throw std::logic_error(std::string() + "failed to find the colon " +
        "separator (:) in the " + MASTER_ADDR_ENV + "environmental variable; " +
        "maybe you forgot to provide the port at all? " + MASTER_ADDR_ENV +
        "should be set to something like '127.0.0.1:12345'");
  }

  ip_addr = addr.substr(0, separator_pos);

  try {
    unsigned long value = std::stoul(addr.substr(separator_pos + 1));
    if (value > USHRT_MAX) {
      throw std::logic_error("the number representing the port is out of"
          "range");
    }
    port = value;
  } catch (const std::exception& e) {
    throw std::logic_error(std::string("") + "failed to convert the " +
        MASTER_PORT_ENV + " environmental variable to unsigned short (port "
        "number); specifically: " + e.what());
  }
}

} // namespace thd
