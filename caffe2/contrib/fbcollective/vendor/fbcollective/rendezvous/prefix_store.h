#pragma once

#include "store.h"

#include <memory>

namespace fbcollective {
namespace rendezvous {

class PrefixStore : public Store {
 public:
  PrefixStore(const std::string& prefix, std::unique_ptr<Store>& store);

  virtual ~PrefixStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(const std::vector<std::string>& keys) override;

 protected:
  const std::string prefix_;
  std::unique_ptr<Store> store_;

  std::string joinKey(const std::string& key);
};

} // namespace rendezvous
} // namespace fbcollective
