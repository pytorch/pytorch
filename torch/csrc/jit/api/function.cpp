#include <torch/csrc/jit/api/function.h>

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

namespace {
c10::FunctionSchema defaultSchemaFor(const Function& function) {
  std::vector<c10::Argument> args;
  std::vector<c10::Argument> returns;
  Graph& g = *function.graph();
  size_t num_inputs = function.num_inputs();
  for (size_t i = 0; i < num_inputs; ++i) {
    const Value* v = g.inputs().at(i);
    std::string name = v->hasDebugName() ? v->debugNameBase()
                                         : ("argument_" + c10::to_string(i));
    args.emplace_back(std::move(name), unshapedType(g.inputs()[i]->type()));
  }
  for (size_t i = 0; i < g.outputs().size(); ++i) {
    returns.emplace_back("", unshapedType(g.outputs()[i]->type()));
  }
  return {function.name(), "", std::move(args), std::move(returns)};
}
} // namespace

Function& Function::setSchema(c10::FunctionSchema schema) {
  schema_ = std::make_unique<c10::FunctionSchema>(std::move(schema));
  return *this;
}

const c10::FunctionSchema& Function::getSchema() const {
  if (schema_ == nullptr) {
    schema_ = std::make_unique<c10::FunctionSchema>(defaultSchemaFor(*this));
  }
  return *schema_;
}

void Function::check_single_output() {
  TORCH_CHECK(
      graph()->outputs().size() == 1,
      "Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs");
}

size_t Function::num_inputs() const {
  return graph()->inputs().size();
}

std::string Function::pretty_print_schema() const {
  AT_ASSERT(schema_);
  std::stringstream ss;
  ss << *schema_;
  return ss.str();
}

} // namespace jit
} // namespace torch
