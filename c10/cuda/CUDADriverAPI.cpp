#pragma once

#include <c10/cuda/CUDADriverAPI.h>

#ifndef C10_MOBILE
namespace c10 {
namespace cuda {
const std::shared_ptr<CUDADriverAPI> c10_get_driver_api() {
  static std::shared_ptr<CUDADriverAPI> _get_driver_api;
  if (!_get_driver_api) {
    _get_driver_api = std::make_shared<CUDADriverAPI>();
  }
  return _get_driver_api;
}
} // namespace cuda
} // namespace c10
#endif // C10_MOBILE