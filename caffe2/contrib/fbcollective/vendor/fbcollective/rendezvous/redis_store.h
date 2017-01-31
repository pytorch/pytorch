#pragma once

#include <string>
#include <vector>

#include <hiredis/hiredis.h>

#include "fbcollective/rendezvous/store.h"

namespace fbcollective {
namespace rendezvous {

class RedisStore : public Store {
 public:
  RedisStore(const std::string& host, int port);
  virtual ~RedisStore();

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  bool check(const std::vector<std::string>& keys);

  virtual void wait(const std::vector<std::string>& keys) override;

 protected:
  redisContext* redis_;
};

} // namespace rendezvous
} // namespace fbcollective
