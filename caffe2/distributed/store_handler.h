#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace caffe2 {

class StoreHandler {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(30);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  virtual ~StoreHandler();

  virtual void set(const std::string& name, const std::string& data) = 0;

  virtual std::string get(const std::string& name) = 0;

  virtual int64_t add(const std::string& name, int64_t value) = 0;

  virtual bool check(const std::vector<std::string>& names) = 0;

  virtual void wait(
      const std::vector<std::string>& names,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) = 0;
};

struct StoreHandlerTimeoutException : public std::runtime_error {
  StoreHandlerTimeoutException() = default;
  explicit StoreHandlerTimeoutException(const std::string& msg)
      : std::runtime_error(msg) {}
};

#define STORE_HANDLER_TIMEOUT(...)              \
  throw ::caffe2::StoreHandlerTimeoutException( \
      ::caffe2::MakeString("[", __FILE__, ":", __LINE__, "] ", __VA_ARGS__));
}
