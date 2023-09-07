#include <ATen/core/op_registration/pyimports.h>

namespace c10 {
namespace impl {

namespace {

std::vector<std::string>& required_pyimports() {
  static std::vector<std::string> _singleton;
  return _singleton;
}

std::mutex& required_pyimports_mutex() {
  static std::mutex _singleton;
  return _singleton;
}

} // namespace

void register_required_pyimport(std::string str) {
  std::lock_guard<std::mutex> lock(required_pyimports_mutex());
  required_pyimports().emplace_back(std::move(str));
}

const std::vector<std::string>& unsafe_get_required_pyimports() {
  return required_pyimports();
}

void clear_required_pyimports() {
  std::lock_guard<std::mutex> lock(required_pyimports_mutex());
  required_pyimports().clear();
}

}
}
