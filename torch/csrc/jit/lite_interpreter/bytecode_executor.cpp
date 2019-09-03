#include <torch/csrc/lite_interpreter/instruction_executor.h>
#include <aten/core/dispatch/Dispatcher.h>

namespace torch {
namespace jit {

namespace {
template <typename dtype> // int64_t, bool, double
int listConstruct(Stack& stack, int64_t num_inputs) {
  auto inputs = peekSlice(stack, 0, num_inputs, num_inputs);
  std::vector<dtype> vals =
      fmap(inputs, [](const IValue& v) { return v.to<dtype>(); });
  drop(stack, num_inputs);
  push(stack, std::move(vals));
  return 0;
}
}

InstructionExecutor::InstructionExecutor(std::shared_ptr<GenericInstructionList> ins_list)
    : ins_list(ins_list) {
  size_t register_size = 0;
  for (const auto& ins : ins_list->instructions) {
    for (const auto& input : ins.inputs) {
      register_size = std::max(register_size, input.unique_id);
    }
  }
  registers.resize(register_size + 1, 0);
}

void InstructionExecutor::loadTensorsFromRegisters(const std::vector<Variable>& inputs, Stack& stack) {
  for (const Variable& input : inputs) {
    int reg = input.unique_id;
    // std::cout << "push reg[" << reg << "];\n" << registers[reg] << "\n\n";
    if (input.free_flag) {
      stack.push_back(std::move(registers[reg]));
    } else {
      stack.push_back(registers[reg]);
    }
  }
}

IValue InstructionExecutor::run(Stack& stack) {
  auto& instructions = ins_list->instructions;
  size_t last = instructions.size();

  while (pc < last) {
    auto& inst = instructions[pc];
    std::cout << "executing " << pc << ": " << inst.name << "(";
    for (int i = 0; i < inst.inputs.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << inst.inputs[i].unique_id;
      if(inst.inputs[i].free_flag)
        std::cout << "!";
    }
    std::cout << ") -> ";
    for (int i = 0; i < inst.outputs.size(); ++i) {
      if (i > 0)
        std::cout << ", ";
      std::cout << inst.outputs[i].unique_id;
    }
    std::cout << std::endl;

    loadTensorsFromRegisters(inst.inputs, stack);

//    std::cout << "stack:" << std::endl;
//    for (auto val : stack) {
//      std::cout << val << ", " << (int)val.isTensor() << std::endl;
//    }

  // Currently we cannot pass constants to c10 kernels. One work-around
    // is to register an operator for each constant. It may also make sense to
    // directly push constants to stack, without goint through the operator
    // registration route.
    if (inst.name == "prim::Load___") {
      for (const auto& attr : inst.attributes) {
        torch::jit::push(stack, attr);
      }
    }
    else if (inst.name == "prim::Constant___") {
      if (inst.attributes.empty()) { // type None
        stack.emplace_back();
      }
      else {
        auto val = inst.attributes[0];
        if (val.isIntList() || val.isBoolList() || val.isDoubleList()) {
          auto v = val.toIntList()->elements();
          torch::jit::push(stack, v);
        } else {
          torch::jit::push(stack, inst.attributes[0]);
        }
      }
    }
    else if (inst.name == "prim::Drop___") {
      drop(stack, inst.inputs.size());
    }
    else if (inst.name == "prim::ListConstruct___") {
      const auto num_inputs = inst.inputs.size();
      AT_ASSERT(inst.outputs.size() == 1);
      size_t output_reg = inst.outputs[0].unique_id;
      auto& output = registers[output_reg];

      if (output.isInt()) {
        listConstruct<int64_t>(stack, num_inputs);
      } else if (output.isDouble()) {
        return listConstruct<double>(stack, num_inputs);
      } else if (output.isBool()) {
        return listConstruct<bool>(stack, num_inputs);
      } else if (output.isTensor()) {
        const size_t stack_size = stack.size();
        std::vector<at::Tensor> vals;
        vals.reserve(num_inputs);
        for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
          vals.emplace_back(std::move(stack[i]).toTensor());
        }
        drop(stack, num_inputs);
        push(stack, std::move(vals));
      } else {
        const size_t stack_size = stack.size();
        std::vector<IValue> vals;
        vals.reserve(num_inputs);
        for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
          vals.emplace_back(std::move(stack[i]));
        }
        drop(stack, num_inputs);
        push(stack, std::move(vals));
      }
    }
    else {
      auto fc = c10::Dispatcher::singleton().findSchema(inst.name.c_str(), inst.overload_name.c_str());
      assert(fc.has_value());
      auto kernel = c10::Dispatcher::singleton().lookup(fc.value(), &stack);
      kernel.call(&stack);
    }

    for (int i = inst.outputs.size() - 1; i >= 0; --i) {
      int reg = inst.outputs[i].unique_id;
      registers[reg] = pop(stack);
//      std::cout << "pop reg[" << reg << "];\n" << registers[reg] << "\n";
    }
    ++pc;
  }

  return stack.front();
}

}
}
