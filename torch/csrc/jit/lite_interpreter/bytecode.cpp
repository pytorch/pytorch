#include "bytecode.h"
#include <aten/src/ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/jit/script/jit_exception.h>
#include <iostream>

namespace torch {
namespace jit {

std::ostream& operator<<(std::ostream& out, Instruction inst);
template <typename dtype> // int64_t, bool, double
void ListConstructFunc(int64_t num_inputs, Stack& stack) {
  auto inputs = peekSlice(stack, 0, num_inputs, num_inputs);
  c10::List<dtype> vals =
      c10::impl::toList(fmap(inputs, [](const IValue& v) {
        return v.to<dtype>(); }));
  drop(stack, num_inputs);
  push(stack, std::move(vals));
}

namespace mobile {

const std::string& Method::name() const {
  return name_;
}

void Method::set_name(const std::string& name) {
  name_ = name;
}

void Method::append_instruction(OpCode op, int N, int X) {
  instructions_.emplace_back(op, N, X);
}

void Method::append_opname(const std::string& name,
                           const std::string& overload_name) {
  op_names_.emplace_back(name, overload_name);
}

void Method::append_constant(const c10::IValue& constant) {
  constants_.push_back(constant);
}

void Method::resize_registers(int size) {
  registers_.resize(size);
}

IValue& Method::reg(size_t reg) {
  return *(registers_.end() - reg);
}

bool Method::run(Stack& stack) {
  size_t pc = 0;
  while (true) {
    std::cout << "RUNNING " << pc << " " << instructions_[pc];
    std::cout << std::endl;
    for (auto val : stack) {
      if (val.isTensor()) {
        std::cout << val.toTensor().sizes() << std::endl;
      } else {
        std::cout << val << std::endl;
      }
    }
    if (pc == 125)
      int debugint = 0;
    Instruction inst = instructions_[pc];
    switch (inst.op) {
      case OP: {
        auto opname = op_names_[inst.X];
        auto fc = c10::Dispatcher::singleton().findSchema(opname);
        assert(fc.has_value());
        auto kernel = c10::Dispatcher::singleton().lookup(fc.value(), &stack);
        kernel.call(&stack);
        ++pc;
      } break;
      case LOAD:
        stack.emplace_back(reg(inst.X));
        ++pc;
        break;
      case MOVE:
        stack.emplace_back(std::move(reg(inst.X)));
        ++pc;
        break;
      case STORE:
        reg(inst.X) = pop(stack);
        ++pc;
        break;
      case STOREN:
        for (size_t i = inst.N; i > 0; --i) {
          reg(inst.X + i - 1) = pop(stack);
        }
        ++pc;
        break;
      case DROP:
        pop(stack);
        ++pc;
        break;
      case DROPR:
        reg(inst.X) = IValue();
        ++pc;
        break;
      case LOADC:
        stack.emplace_back(constants_[inst.X]);
        ++pc;
        break;
      case GET_ATTR: {
        auto userObj = pop(stack).toObject();
        auto value = userObj->getSlot(inst.X);
        if (value.isObject()) {
          auto obj = value.toObject();
          std::cout << "obj : " << obj->name() << ", "
                    << obj->slots().size() << " slots."
                    << std::endl;
        } else if (value.isTensor()) {
          auto tensor = value.toTensor();
          std::cout << "tensor with dim " << tensor.dim() << std::endl;
        }
        push(stack, std::move(value));
        ++pc;
      } break;
      case SET_ATTR: {
        auto v = pop(stack);
        auto userObj = pop(stack).toObject();
        userObj->setSlot(inst.X, std::move(v));
        ++pc;
      } break;
      case LIST_CONSTRUCT: {
        if (inst.N == 1) {
          ListConstructFunc<int64_t>(inst.X, stack);
        } else if (inst.N == 2) {
          ListConstructFunc<double>(inst.X, stack);
        } else if (inst.N == 3) {
          ListConstructFunc<bool>(inst.X, stack);
        } else if (inst.N == 4) {
          const size_t stack_size = stack.size();
          c10::List<at::Tensor> vals;
          vals.reserve(inst.X);
          for (size_t i = stack_size - inst.X; i < stack_size; ++i) {
            vals.emplace_back(std::move(stack[i]).toTensor());
          }
          drop(stack, inst.X);
          push(stack, std::move(vals));
        } else {
          const size_t stack_size = stack.size();
          auto vals = c10::impl::GenericList(c10::impl::deprecatedUntypedList());
          vals.reserve(inst.X);
          for (size_t i = stack_size - inst.X; i < stack_size; ++i) {
            vals.emplace_back(std::move(stack[i]));
          }
          drop(stack, inst.X);
          push(stack, std::move(vals));
        }
        ++pc;
      } break;
      case JF:
        pc += (pop(stack).toBool()) ? 1 : inst.X;
        break;
      case JMP:
        pc += inst.X;
        break;
      case LOOP: {
        // stack: iteration_count, max_iter, cond, loop_carried_deps...
        auto frame = stack.end() - (inst.N + 1);
        int64_t trip_count = frame[0].toInt();
        int64_t max_trip_count = frame[1].toInt();
        bool cond = frame[2].toBool();
        if (trip_count < max_trip_count && cond) {
          frame[2] = trip_count;
          frame[0] = trip_count + 1;
          ++pc;
        } else {
          size_t n_loop_carried = inst.N - 2;
          for (size_t i = 0; i < n_loop_carried; ++i) {
            frame[i] = std::move(frame[i + 3]);
          }
          drop(stack, 3); // iteration_count, max_iter, cond
          pc += inst.X;
        }
      } break;
      case CALL: {
        AT_ERROR("Instruction CALL is not supported in mobile.");
      } break;
      case INTERFACE_CALL: {
        AT_ERROR("Instruction INTERFACE_CALL is not supported in mobile.");
      } break;
      case RET:
        return false;
      case WAIT: {
        AT_ERROR("Instruction WAIT is not supported in mobile.");
      } break;
      case GUARD: {
        AT_ERROR("Instruction GUARD is not supported in mobile.");
      } break;
      case TAIL_CALL: {
        AT_ERROR("Instruction TAIL_CALL is not supported in mobile.");
      } break;
    }
  }
  return false;
}

void Bytecode::append_method(const Method& method) {
  methods_.push_back(method);
}

IValue Bytecode::run_method(const std::string& method_name, Stack& stack) {
  auto m = find_method(method_name);
  stack.insert(stack.begin(), object_);
  m.run(stack);
  return stack.front();
}

void Bytecode::set_object(const c10::intrusive_ptr<c10::ivalue::Object>& object) {
  object_ = object;
}

Method Bytecode::find_method(const std::string& name) {
  for (auto m : methods_) {
    if (m.name() == name) return m;
  }
  AT_ERROR("Method '", name, "' is not defined.");
}

} // namespace mobile
} // namespace torch
} // namespace jit
