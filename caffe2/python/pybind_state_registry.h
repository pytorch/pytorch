#pragma once

#include <pybind11/pybind11.h>
#include "c10/util/Registry.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

struct PybindAddition {
  PybindAddition() {}
  PybindAddition(py::module&) {}
  virtual ~PybindAddition(){};
};

C10_DECLARE_REGISTRY(PybindAdditionRegistry, PybindAddition, py::module&);

#define REGISTER_PYBIND_ADDITION(funcname)                                    \
  namespace {                                                                 \
  struct funcname##Impl : public PybindAddition {                             \
    funcname##Impl(py::module& m) {                                           \
      funcname(m);                                                            \
    }                                                                         \
  };                                                                          \
  C10_REGISTER_CLASS(PybindAdditionRegistry, funcname##Impl, funcname##Impl); \
  }

} // namespace python
} // namespace caffe2
