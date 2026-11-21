#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

namespace torch::dynamo {

// Temporarily disable frame evaluation on this thread, including cached code.
// May be constructed without the GIL; when Dynamo is off it does not acquire
// it.
class TORCH_PYTHON_API DisableDynamoGuard {
 public:
  DisableDynamoGuard();
  ~DisableDynamoGuard();
  DisableDynamoGuard(const DisableDynamoGuard&) = delete;
  DisableDynamoGuard& operator=(const DisableDynamoGuard&) = delete;
  DisableDynamoGuard(DisableDynamoGuard&&) = delete;
  DisableDynamoGuard& operator=(DisableDynamoGuard&&) = delete;

 private:
  PyObject* callback_ = nullptr;
  PyObject* module_ = nullptr;
};

} // namespace torch::dynamo
