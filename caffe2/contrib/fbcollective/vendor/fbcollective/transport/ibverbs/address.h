#pragma once

#include <string>

#include <infiniband/verbs.h>

#include "fbcollective/transport/address.h"

namespace fbcollective {
namespace transport {
namespace ibverbs {

// Forward declaration
class Pair;

class Address : public ::fbcollective::transport::Address {
 public:
  Address();
  explicit Address(const std::vector<char>&);
  virtual ~Address() {}

  virtual std::vector<char> bytes() const override;
  virtual std::string str() const override;

 protected:
  struct {
    uint32_t lid;
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid ibv_gid;
  } addr_;

  // Pair can access addr_ directly
  friend class Pair;
};

} // namespace ibverbs
} // namespace transport
} // namespace fbcollective
