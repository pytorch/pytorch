#pragma once

#include "caffe2/distributed/store_handler.h"

#include "fbcollective/rendezvous/store.h"

namespace caffe2 {
namespace fbcollective {

class StoreHandlerWrapper : public ::fbcollective::rendezvous::Store {
 public:
  explicit StoreHandlerWrapper(StoreHandler& handler) : handler_(handler) {}

  virtual ~StoreHandlerWrapper() {}

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(const std::vector<std::string>& keys) override;

 protected:
  StoreHandler& handler_;
};

} // namespace fbcollective
} // namespace caffe2
