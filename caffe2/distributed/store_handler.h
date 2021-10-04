#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "caffe2/core/common.h"

namespace caffe2 {

class TORCH_API StoreHandler {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(30);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  virtual ~StoreHandler();

  /*
   * Set data for the key if it doesn't exist.
   * If the key exists the data should be the same as the existing key.
   */
  virtual void set(const std::string& name, const std::string& data) = 0;

  /*
   * Get the data for the key.
   * The call should wait until the key is stored with specified timeout
   * and return data if set else fail.
   */
  virtual std::string get(
      const std::string& name,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) = 0;

  /*
   * Does an atomic add operation on the key and returns the latest updated
   * value.
   * Note: To access the current value for this counter call with value = 0
   */
  virtual int64_t add(const std::string& name, int64_t value) = 0;

  /*
   * Returns the number of keys in this store.
   */
  virtual int64_t getNumKeys() = 0;

  /*
   * Removes the specified key from the store.
   */
  virtual bool deleteKey(const std::string& key) = 0;

  /*
   * Check if a keys exist in the store.
   */
  virtual bool check(const std::vector<std::string>& names) = 0;

  /*
   * Wait for Keys to be stored.
   */
  virtual void wait(
      const std::vector<std::string>& names,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) = 0;
};

/*
 * The backing store is no longer available. It may have been deleted.
 */
struct TORCH_API StoreHandlerNotAvailableException
    : public std::runtime_error {
  explicit StoreHandlerNotAvailableException(const std::string& msg)
      : std::runtime_error(msg) {}
};

#define STORE_HANDLER_NOT_AVAILABLE(...)             \
  throw ::caffe2::StoreHandlerNotAvailableException( \
      ::c10::str("[", __FILE__, ":", __LINE__, "] ", __VA_ARGS__));

/*
 * Timeout accessing the store.
 */
struct TORCH_API StoreHandlerTimeoutException : public std::runtime_error {
  explicit StoreHandlerTimeoutException(const std::string& msg)
      : std::runtime_error(msg) {}
};

#define STORE_HANDLER_TIMEOUT(...)              \
  throw ::caffe2::StoreHandlerTimeoutException( \
      ::c10::str("[", __FILE__, ":", __LINE__, "] ", __VA_ARGS__));
} // namespace caffe2
