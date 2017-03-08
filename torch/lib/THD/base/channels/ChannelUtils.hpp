#pragma once

#include <sys/socket.h>
#include <sys/types.h>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <system_error>
#include <tuple>

namespace thd {

#define SYSCHECK(expr) { \
  errno = 0; (expr);     \
  if (errno != 0) throw std::system_error(errno, std::system_category()); \
}

template<typename T>
void send_bytes(int socket, const T* buffer, std::size_t length)
{
  std::size_t bytes_to_send = sizeof(T) * length;
  if (bytes_to_send == 0)
    return;

  auto bytes = reinterpret_cast<const std::uint8_t*>(buffer);
  std::uint8_t *current_bytes = const_cast<std::uint8_t*>(bytes);

  while (bytes_to_send > 0) {
    ssize_t bytes_sent;
    SYSCHECK(bytes_sent = ::send(socket, current_bytes, bytes_to_send, 0))
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

inline std::uint16_t convertToPort(long port) {
  if ((port < 0) || (port >= UINT16_MAX))
    throw std::domain_error("invalid port");

  return static_cast<std::uint16_t>(port);
}

inline std::uint32_t convertToRank(long rank, long min = 0) {
  if ((rank < min) || (rank >= UINT32_MAX))
    throw std::domain_error("invalid rank");

  return static_cast<std::uint32_t>(rank);
}

std::tuple<int, std::uint16_t> listen(std::uint16_t port = 0);
int connect(const std::string& address, std::uint16_t port, int wait = true);
std::tuple<int, std::string> accept(int listen_socket, int timeout = -1);

std::tuple<std::uint16_t, std::uint32_t> load_master_env();
std::tuple<std::string, std::uint16_t> load_worker_env();
std::uint32_t load_rank_env();

} // namespace thd
