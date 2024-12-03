#include <torch/csrc/jit/runtime/profiling_record.h>

namespace torch::jit::fuser::onednn {

static bool canFuseNode(const Node* node) {
  switch (node->kind()) {
    case aten::conv2d:
    case aten::_convolution:
    case aten::batch_norm:
    case aten::layer_norm:
    case aten::add:
    case aten::mul:
    case aten::tanh:
    case aten::relu:
    case aten::elu:
    case aten::sigmoid:
    case aten::gelu:
    case aten::sqrt:
    case aten::abs:
    case aten::square:
    case aten::hardtanh:
    case aten::relu6:
    case aten::softmax:
    case aten::max_pool2d:
    case aten::avg_pool2d:
    case aten::matmul:
    case aten::mm:
    case aten::linear:
    case aten::addmm:
      return true;

    default:
      return false;
  }
}

namespace {
class RegisterInterface {
 public:
  RegisterInterface() {
    RegisterProfilingNode(canFuseNode);
  }
};

static RegisterInterface register_interface_;
} // namespace

} // namespace torch::jit::fuser::onednn
