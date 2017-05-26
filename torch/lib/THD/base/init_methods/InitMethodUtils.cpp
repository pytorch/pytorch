#include "InitMethodUtils.hpp"

#include <unistd.h>

#include <tuple>

namespace thd {

void discoverWorkers(int listen_socket, rank_type world_size) {
  // accept connections from workers so they can know our address
  std::vector<int> sockets(world_size - 1);
  for (rank_type i = 0; i < world_size - 1; ++i) {
    std::tie(sockets[i], std::ignore) = accept(listen_socket);
  }

  for (auto socket : sockets) {
    send_value<uint8_t>(socket, 1);
    ::close(socket);
  }
}

std::string discoverMaster(std::vector<std::string> addresses, port_type port) {
  // try to connect to address via any of the addresses
  std::string master_address = "";
  int socket;
  for (const auto& address : addresses) {
    try {
      socket = connect(address, port);
      master_address = address;
      break;
    } catch (...) {} // when connection fails just try different address
  }

  if (master_address == "") {
    throw std::runtime_error("could not establish connection with other processes");
  }

  // wait until master sends us confirmation byte
  uint8_t confirmation_byte;
  try {
    recv_bytes<uint8_t>(socket, &confirmation_byte, 1);
  } catch (...) {
    /*
     * recv_bytes can throw error because master is immediately closing socket
     * after sending message. This causes recv to read 0 bytes what causes error.
     */
  }
  ::close(socket);

  return master_address;
}

}
