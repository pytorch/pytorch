#pragma once

#include <string>
#include <vector>

namespace fbcollective {
namespace rendezvous {

class Store {
 public:
  virtual ~Store();

  virtual void set(const std::string& key, const std::vector<char>& data) = 0;

  virtual std::vector<char> get(const std::string& key) = 0;

  virtual void wait(const std::vector<std::string>& keys) = 0;
};

} // namespace rendezvous
} // namespace fbcollective
