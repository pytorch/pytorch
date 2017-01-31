#pragma once

#include "fbcollective/rendezvous/store.h"

#include <condition_variable>
#include <mutex>
#include <unordered_map>

namespace fbcollective {
namespace rendezvous {

class HashStore : public Store {
 public:
  virtual ~HashStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(const std::vector<std::string>& keys) override;

 protected:
  std::unordered_map<std::string, std::vector<char>> map_;
  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace rendezvous
} // namespace fbcollective
