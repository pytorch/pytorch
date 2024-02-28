#include "store_handler.h"

namespace caffe2 {
namespace gloo {

void StoreHandlerWrapper::set(
    const std::string& key,
    const std::vector<char>& data) {
  std::string stringValue(data.data(), data.size());
  handler_.set(key, stringValue);
}

std::vector<char> StoreHandlerWrapper::get(const std::string& key) {
  std::string str = handler_.get(key);
  return std::vector<char>(str.begin(), str.end());
}

void StoreHandlerWrapper::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  handler_.wait(keys, timeout);
}

} // namespace gloo
} // namespace caffe2
