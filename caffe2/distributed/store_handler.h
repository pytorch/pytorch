#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace caffe2 {

class StoreHandler {
 public:
  virtual ~StoreHandler();

  virtual void set(const std::string& name, const std::string& data) = 0;

  virtual std::string get(const std::string& name) = 0;

  virtual int64_t add(const std::string& name, int64_t value) = 0;

  virtual bool check(const std::vector<std::string>& names) = 0;

  virtual void wait(const std::vector<std::string>& names) = 0;
};
}
