#include <torch/csrc/distributed/c10d/CallbackWork.hpp>

namespace c10d {
CallbackWork::~CallbackWork() {
  py::gil_scoped_acquire ag;
  callback_.dec_ref();
  callback_.ptr() = nullptr;
}

} // namespace c10d
