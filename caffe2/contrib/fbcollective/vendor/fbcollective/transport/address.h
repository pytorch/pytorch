#pragma once

#include <string>
#include <vector>

namespace fbcollective {
namespace transport {

class Address {
 public:
  virtual ~Address() = 0;

  virtual std::string str() const = 0;
  virtual std::vector<char> bytes() const = 0;
};

} // namespace transport
} // namespace fbcollective
