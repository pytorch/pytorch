#include <jit/caffe2_operator.h>
#include <caffe2/core/operator.h>

namespace torch {
namespace jit {

Operator createOperatorFromCaffe2(const std::string& name) {
  auto symbolic_name = c10::Symbol::fromQualString("caffe2::" + name);
  auto fn_wrap = caffe2::FunctionSchemaRegistry()->Create(symbolic_name.toUnqualString());
  CAFFE_ENFORCE(
      fn_wrap,
      "Operator not registered with FunctionSchema constructor:",
      name);
  auto fn = fn_wrap->getSchema();

  return Operator(fn, [symbolic_name, fn](Stack& stack) {
      const auto input_size = fn.arguments().size();
      const auto output_size = fn.returns().size();
      std::vector<c10::IValue> inputs;
      for (size_t i = 0; i < input_size; ++i) {
        auto input = pop(stack);
        // Tensors come in as variables but need to be unwrapped
        if (input.isTensor()) {
          input = torch::autograd::Variable(input.toTensor()).data();
        }
        inputs.emplace(inputs.begin(), std::move(input));
      }

      // We use a temporary stack for arguments passed into RunOperator
      std::list<c10::IValue> outputs_real;
      std::vector<c10::IValue*> outputs;
      for (size_t i = 0; i < output_size; ++i) {
        if (DimensionedTensorType::get() == fn.returns()[i].type()) {
          outputs_real.emplace_back(c10::IValue(at::Tensor()));
        } else {
          outputs_real.emplace_back(c10::IValue());
        }
        outputs.emplace_back(&outputs_real.back());
      }

      caffe2::RunOperator(symbolic_name, inputs, outputs);

      // We need to convert tensors back into variables
      for (auto& t : outputs_real) {
        if (t.isTensor()) {
            push(stack, c10::IValue(torch::autograd::make_variable(t.toTensor())));
        } else {
            push(stack, std::move(t));
        }
      }

      return 0;
  });
}

}} // torch::jit
