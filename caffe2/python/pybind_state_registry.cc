#include "caffe2/python/pybind_state_registry.h"

C10_DEFINE_REGISTRY(PybindAdditionRegistry, caffe2::PybindAddition, pybind11::module&);
