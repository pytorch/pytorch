#pragma once

#include <zendnn.h>
#include <zendnn.hpp>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "ATen/native/zendnn/Allocators.hpp"

namespace zendnn {

#ifdef _WIN32
#define ZENDNN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define ZENDNN_EXPORT __attribute__((__visibility__("default")))
#else
#define ZENDNN_EXPORT
#endif

using format_tag = memory::format_tag;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using kind = zendnn::primitive::kind;
using batch_normalization_flag = zendnn::normalization_flags;
using scale_t = std::vector<float>;
using exec_args = std::unordered_map<int, memory>;

#ifndef NDEBUG
#define ZENDNN_ENFORCE(condition, message)                                   \
  do {                                                                       \
    error::wrap_c_api(                                                       \
        (condition) ? zendnn_success : zendnn_invalid_arguments, (message)); \
  } while (false)
#else
#define ZENDNN_ENFORCE(condition, message)
#endif

const scale_t ZENDNN_DEF_SCALE{1.0f};

namespace utils {
/// cpu execution engine only.
struct engine : public zendnn::engine {
  // friend class tensor;

  /// Singleton CPU engine for all primitives
  static ZENDNN_EXPORT engine& cpu_engine();

  engine(kind akind = kind::cpu, size_t index = 0)
      : zendnn::engine(akind, index),
        malloc(utils::allocator::malloc),
        free(utils::allocator::free) {}

  void set_allocator(
      const std::function<void*(size_t)>& malloc,
      const std::function<void(void*)>& free) {
    this->malloc = malloc;
    this->free = free;
  }

  //  private:
  std::function<void*(size_t)> malloc;
  std::function<void(void*)> free;
};

/// A default stream
struct stream : public zendnn::stream {
  static zendnn::stream& default_stream() {
    static zendnn::stream s(engine::cpu_engine());
    return s;
  }
};
} // namespace utils
} // namespace zendnn
