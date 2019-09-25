#include "interpreter.h"
#include <torch/csrc/jit/mobile/function.h>
#include <aten/src/ATen/core/operator_name.h>

namespace torch{
namespace jit{
char const * toString(OpCode op);
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
InterpreterState::InterpreterState(std::shared_ptr<Code> code) : code_(code) {
  registers_.resize(code_->register_size_);
}

//InterpreterState::InterpreterState(Function* function)
//    : function_(function) {
//  registers_.resize(function->register_size());
//}

bool InterpreterState::run(Stack& stack) {
  size_t pc = 0;
  while (true) {
    std::cout << "RUNNING " << pc << " " << code_->instructions_[pc];
    std::cout << std::endl;
    for (auto val : stack) {
      if (val.isTensor()) {
        std::cout << val.toTensor().sizes() << std::endl;
      } else {
        std::cout << val << std::endl;
      }
    }
    Instruction inst = code_->instructions_[pc];
    TORCH_CHECK(isOpSupportedInMobile(inst.op), toString(inst.op),
                " is not supported in mobile module.");
    switch (inst.op) {
      case OP: {
        c10::Dispatcher::singleton().callBoxed(*code_->operators_[inst.X], &stack);
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
        stack.emplace_back(code_->constants_[inst.X]);
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
          auto vals = c10::impl::GenericList(c10::AnyType::get());
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
      case RET:
        return false;
      default:
        AT_ERROR(toString(inst.op), " is invalid.");
    }
  }
  return false;
}

IValue& InterpreterState::reg(size_t reg) {
  return *(registers_.end() - reg);
}

} // namespace mobile
} // namespace torch
} // namespace jit
