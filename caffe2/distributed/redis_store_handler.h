#pragma once

#include <caffe2/distributed/store_handler.h>

extern "C" {
#include <hiredis/hiredis.h>
}

#include <string>

namespace caffe2 {

class CAFFE2_API RedisStoreHandler : public StoreHandler {
 public:
  explicit RedisStoreHandler(std::string& host, int port, std::string& prefix);
  virtual ~RedisStoreHandler();

  virtual void set(const std::string& name, const std::string& data) override;

  virtual std::string get(
      const std::string& name,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

  virtual int64_t add(const std::string& name, int64_t value) override;

  virtual bool check(const std::vector<std::string>& names) override;

  virtual void wait(
      const std::vector<std::string>& names,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

 private:
  std::string host_;
  int port_;
  std::string prefix_;

  redisContext* redis_;

  std::string compoundKey(const std::string& name);
};

} // namespace caffe2
