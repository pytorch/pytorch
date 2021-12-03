#include "lazy_tensor_core/csrc/ops/constant.h"

#include <algorithm>
#include <sstream>
#include <csignal>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Constant::Constant(lazy_tensors::Literal value)
    : TsNode(OpKind(at::prim::Constant), value.shape(), /*num_outputs=*/1,
           value.Hash()),
      value_(std::move(value)) {
        static const auto THROW_ON_CONSTANT = std::getenv("LTC_THROW_ON_CONSTANT"); 
        if (THROW_ON_CONSTANT) {
          raise(SIGINT);
          //TORCH_CHECK(false);
        }
      }

std::string Constant::ToString() const {
  // The Literal to string conversion produces \n separated content, which we do
  // not want. It can also produce giant strings, but that's a different issue.
  std::string value_as_string = value_.ToStringWithoutShape();
  std::replace(value_as_string.begin(), value_as_string.end(), '\n', ';');
  std::stringstream ss;
  ss << TsNode::ToString() << ", value=" << value_as_string;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
