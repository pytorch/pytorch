#include <torch/csrc/jit/script/type_parser.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/tree_views.h>
#include <torch/csrc/jit/script/sugared_value.h>

namespace torch {
namespace jit {
namespace script {

struct NoneValue : SugaredValue {
  NoneValue() = default;
  std::string kind() const override {
    return "None";
  }
};

std::shared_ptr<SugaredValue> PrintValue::call(
  const SourceRange& loc,
  Method & m,
  at::ArrayRef<NamedValue> inputs,
  at::ArrayRef<NamedValue> attributes,
  size_t n_binders) {
    auto& g = *m.graph();
    if (!attributes.empty())
      throw ErrorReport(loc) << "print doesn't accept any keyword arguments";

    //temporary hack to allow print statements to work in python 2, where
    //print(a, b) is treated as a (a, b) tuple input.

    std::vector<Value*> lowered_inputs = toValues(*m.graph(), inputs);
    if(lowered_inputs.size() == 1 && lowered_inputs.at(0)->node()->kind() == prim::TupleConstruct) {
      auto input = lowered_inputs[0];
      for(size_t j = 0; j < input->node()->inputs().size(); ++j) {
        lowered_inputs.insert(lowered_inputs.begin() + 1 + j, input->node()->inputs().at(j));
      }
      lowered_inputs.erase(lowered_inputs.begin());
    }
    g.insertNode(g.create(prim::Print, lowered_inputs, 0)
                     ->setSourceLocation(std::make_shared<SourceRange>(loc)));
    return std::make_shared<NoneValue>();
}

} // namespace script
} // namespace jit
} // namespace torch
