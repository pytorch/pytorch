#pragma once

#include <string>

#include <sys/socket.h>

#include "fbcollective/transport/address.h"

namespace fbcollective {
namespace transport {
namespace tcp {

// Forward declaration
class Pair;

class Address : public ::fbcollective::transport::Address {
 public:
  Address() {}
  explicit Address(const struct sockaddr_storage&);
  explicit Address(const std::vector<char>&);
  virtual ~Address() {}

  virtual std::string str() const override;
  virtual std::vector<char> bytes() const override;

  static Address fromSockName(int fd);
  static Address fromPeerName(int fd);

 protected:
  struct sockaddr_storage ss_;

  // Pair can access ss_ directly
  friend class Pair;
};

} // namespace tcp
} // namespace transport
} // namespace fbcollective
