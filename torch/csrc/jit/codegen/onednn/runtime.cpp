#include <torch/csrc/jit/codegen/onednn/runtime.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using namespace dnnl::graph;

engine& Engine::getEngine() {
  static engine cpu_engine(dnnl::graph::engine::kind::cpu, 0);
  return cpu_engine;
}

stream& Stream::getStream() {
  static stream cpu_stream{Engine::getEngine(), nullptr};
  return cpu_stream;
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch