#pragma once

#include <chrono>

namespace at {
namespace native {

template<typename F>
class AutotuneTimer {
  public:
    AutotuneTimer() = delete;
    AutotuneTimer(F&& callback, const bool is_cuda=false, const bool profile=false)
       :  _callback(std::move(callback)),
          _is_cuda(is_cuda),
          _profile(profile)
    {
      if (_profile) {
        if (_is_cuda) {
           at::detail::getCUDAHooks().deviceSynchronize(at::detail::getCUDAHooks().current_device());
        }
        // cannot use initializer list as we may need to synchronize
        _start = std::chrono::high_resolution_clock::now();
      }
    }
    ~AutotuneTimer() {
      if (_profile) {
        if (_is_cuda) {
          at::detail::getCUDAHooks().deviceSynchronize(at::detail::getCUDAHooks().current_device());
        }
        auto dur = std::chrono::high_resolution_clock::now() - _start;
        _callback(dur.count());
      }
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
    F&& _callback;
    bool _is_cuda = false;
    bool _profile = false;
  };

}}
