/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/tcp/address.h"

#include <arpa/inet.h>
#include <string.h>

#include "gloo/common/logging.h"

namespace gloo {
namespace transport {
namespace tcp {

Address::Address(const struct sockaddr_storage& ss) {
  ss_ = ss;
}

Address::Address(const struct sockaddr* addr, size_t addrlen) {
  memcpy(&ss_, addr, addrlen);
}

Address::Address(const std::vector<char>& bytes) {
  GLOO_ENFORCE_EQ(sizeof(ss_), bytes.size());
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
  if (port > 0) {
    snprintf(str + len, sizeof(str) - len, "]:%d", port);
  } else {
    snprintf(str + len, sizeof(str) - len, "]");
  }

  return str;
}

Address Address::fromSockName(int fd) {
  struct sockaddr_storage ss;
  socklen_t addrlen = sizeof(ss);
  int rv;

  rv = getsockname(fd, (struct sockaddr*)&ss, &addrlen);
  GLOO_ENFORCE_NE(rv, -1, "getsockname: ", strerror(errno));
  return Address(ss);
}

Address Address::fromPeerName(int fd) {
  struct sockaddr_storage ss;
  socklen_t addrlen = sizeof(ss);
  int rv;

  rv = getpeername(fd, (struct sockaddr*)&ss, &addrlen);
  GLOO_ENFORCE_NE(rv, -1, "getpeername: ", strerror(errno));
  return Address(ss);
}

} // namespace tcp
} // namespace transport
} // namespace gloo
