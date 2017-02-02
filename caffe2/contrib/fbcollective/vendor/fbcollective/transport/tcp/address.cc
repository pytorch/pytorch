#include "fbcollective/transport/tcp/address.h"

#include <arpa/inet.h>
#include <string.h>

#include "fbcollective/common/logging.h"

namespace fbcollective {
namespace transport {
namespace tcp {

Address::Address(const struct sockaddr_storage& ss) {
  ss_ = ss;
}

Address::Address(const std::vector<char>& bytes) {
  FBC_ENFORCE_EQ(sizeof(ss_), bytes.size());
  memcpy(&ss_, bytes.data(), sizeof(ss_));
}

std::vector<char> Address::bytes() const {
  std::vector<char> bytes(sizeof(ss_));
  memcpy(bytes.data(), &ss_, sizeof(ss_));
  return bytes;
}

std::string Address::str() const {
  char str[INET6_ADDRSTRLEN + 8];
  int port = 0;

  str[0] = '[';
  if (ss_.ss_family == AF_INET) {
    struct sockaddr_in* in = (struct sockaddr_in*)&ss_;
    inet_ntop(AF_INET, &in->sin_addr, str + 1, sizeof(str) - 1);
    port = in->sin_port;
  } else if (ss_.ss_family == AF_INET6) {
    struct sockaddr_in6* in6 = (struct sockaddr_in6*)&ss_;
    inet_ntop(AF_INET6, &in6->sin6_addr, str + 1, sizeof(str) - 1);
    port = in6->sin6_port;
  } else {
    snprintf(str + 1, sizeof(str) - 1, "none");
  }

  auto len = strlen(str);
  snprintf(str + len, sizeof(str) - len, "]:%d", port);

  return str;
}

Address Address::fromSockName(int fd) {
  struct sockaddr_storage ss;
  socklen_t addrlen = sizeof(ss);
  int rv;

  rv = getsockname(fd, (struct sockaddr*)&ss, &addrlen);
  FBC_ENFORCE_NE(rv, -1, "getsockname: ", strerror(errno));
  return Address(ss);
}

Address Address::fromPeerName(int fd) {
  struct sockaddr_storage ss;
  socklen_t addrlen = sizeof(ss);
  int rv;

  rv = getpeername(fd, (struct sockaddr*)&ss, &addrlen);
  FBC_ENFORCE_NE(rv, -1, "getpeername: ", strerror(errno));
  return Address(ss);
}

} // namespace tcp
} // namespace transport
} // namespace fbcollective
