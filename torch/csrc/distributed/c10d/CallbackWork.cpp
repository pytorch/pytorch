#include <torch/csrc/distributed/c10d/CallbackWork.hpp>

namespace c10d {
CallbackWork::~CallbackWork() {
  py::gil_scoped_acquire ag;
  cb_.dec_ref();
  cb_.ptr() = nullptr;
}

} // namespace c10d
