#pragma once

#include "caffe2/core/common.h"
#include "caffe2/distributed/store_handler.h"

#include <gloo/rendezvous/store.h>

namespace caffe2 {
namespace gloo {

class TORCH_API StoreHandlerWrapper : public ::gloo::rendezvous::Store {
 public:
  explicit StoreHandlerWrapper(StoreHandler& handler) : handler_(handler) {}

  virtual ~StoreHandlerWrapper() override {}

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

   std::vector<char> get(const std::string& key) override;

   void wait(const std::vector<std::string>& keys) override {
    wait(keys, ::gloo::rendezvous::Store::kDefaultTimeout);
  }

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

 protected:
  StoreHandler& handler_;
};

} // namespace gloo
} // namespace caffe2
