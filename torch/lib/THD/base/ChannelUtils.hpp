#pragma once

#include <sys/socket.h>
#include <sys/types.h>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <tuple>

namespace thd {

using rank_type = std::uint32_t;
using port_type = std::uint16_t;

#define SYSCHECK(expr) { \
  errno = 0; (expr);     \
  if (errno != 0) throw std::system_error(errno, std::system_category()); \
}

template<typename T>
void send_bytes(int socket, const T* buffer, std::size_t length, bool more_data = false)
{
  std::size_t bytes_to_send = sizeof(T) * length;
  if (bytes_to_send == 0)
    return;

  auto bytes = reinterpret_cast<const std::uint8_t*>(buffer);
  std::uint8_t *current_bytes = const_cast<std::uint8_t*>(bytes);

  int flags = 0;
#ifdef MSG_MORE
  if (more_data) { // there is more data to send
    flags |= MSG_MORE;
  }
#endif

  while (bytes_to_send > 0) {
    ssize_t bytes_sent;
    SYSCHECK(bytes_sent = ::send(socket, current_bytes, bytes_to_send, flags))
    if (bytes_sent == 0)
      throw std::system_error(EBADMSG, std::system_category());

    bytes_to_send -= bytes_sent;
    current_bytes += bytes_sent;
  }
}


template<typename T>
void recv_bytes(int socket, T* buffer, std::size_t length)
{
  std::size_t bytes_to_receive = sizeof(T) * length;
  if (bytes_to_receive == 0)
    return;

  auto bytes = reinterpret_cast<std::uint8_t*>(buffer);
  std::uint8_t *current_bytes = bytes;

  while (bytes_to_receive > 0) {
    ssize_t bytes_received;
    SYSCHECK(bytes_received = ::recv(socket, current_bytes, bytes_to_receive, 0))
    if (bytes_received == 0)
      throw std::system_error(EBADMSG, std::system_category());

    bytes_to_receive -= bytes_received;
    current_bytes += bytes_received;
  }
}

inline port_type convertToPort(long port) {
  if ((port < 0) || (port >= std::numeric_limits<port_type>::max()))
    throw std::domain_error("invalid port (value out of range)");

  return static_cast<port_type>(port);
}

inline rank_type convertToRank(long rank, long min = 0) {
  if ((rank < min) || (rank >= std::numeric_limits<rank_type>::max()))
    throw std::domain_error("invalid rank (value out of range)");

  return static_cast<rank_type>(rank);
}

std::tuple<int, port_type> listen(port_type port = 0);
int connect(const std::string& address, port_type port, bool wait = true);
std::tuple<int, std::string> accept(int listen_socket, int timeout = -1);

std::tuple<port_type, rank_type> load_master_env();
std::tuple<std::string, port_type> load_worker_env();
rank_type load_rank_env();

} // namespace thd
