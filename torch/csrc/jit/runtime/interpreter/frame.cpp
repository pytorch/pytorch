#include <torch/csrc/jit/runtime/interpreter/frame.h>

namespace torch {
namespace jit {
namespace interpreter {
std::atomic<size_t> Frame::num_frames;
}
}
}
