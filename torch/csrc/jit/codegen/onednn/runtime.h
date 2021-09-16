#pragma once

#include <oneapi/dnnl/dnnl_graph.hpp>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct Engine {
  // CPU engine singleton
  static dnnl::graph::engine& getEngine();
  Engine(const Engine&) = delete;
  void operator=(const Engine&) = delete;
};

struct Stream {
  // CPU stream singleton
  static dnnl::graph::stream& getStream();
  Stream(const Stream&) = delete;
  void operator=(const Stream&) = delete;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch