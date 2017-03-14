#pragma once

#include "caffe2/distributed/store_handler.h"

#include "gloo/rendezvous/store.h"

namespace caffe2 {
namespace gloo {

class StoreHandlerWrapper : public ::gloo::rendezvous::Store {
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

} // namespace gloo
} // namespace caffe2
