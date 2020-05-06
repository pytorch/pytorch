#include <torch/csrc/jit/mobile/interpreter.h>
#include <ATen/core/function.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#if defined(PYTORCH_MOBILE_OPERATOR_OBSERVER)
#include <ATen/record_function.h>
#include <torch/csrc/jit/mobile/observer.h>
#endif

namespace torch {
namespace jit {
char const* toString(OpCode op);
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {
InterpreterState::InterpreterState(std::shared_ptr<Code> code)
    : code_(std::move(code)) {
  registers_.resize(code_->register_size_);
}

bool InterpreterState::run(Stack& stack) {
  size_t pc = 0;
  while (true) {
    Instruction inst = code_->instructions_[pc];

    //    std::cout << "RUNNING " << pc << " " << code_->instructions_[pc];
    //    if (inst.op == OP) {
    //      std::cout << ", " << code_->op_names_[inst.X].name;
    //      if (!code_->op_names_[inst.X].overload_name.empty()) {
    //        std::cout << "." << code_->op_names_[inst.X].overload_name;
    //      }
    //    }
    //    std::cout << std::endl;
    switch (inst.op) {
      case OP: {
#if defined(PYTORCH_MOBILE_OPERATOR_OBSERVER)
        if (auto debug_info = c10::ThreadLocalDebugInfo::get(
                c10::DebugInfoKind::MOBILE_RUNTIME_INFO)) {
          if (auto* mobile_debug_info =
                  dynamic_cast<MobileDebugInfo*>(debug_info.get())) {
            mobile_debug_info->setOpIdx(pc);
          }
        }
        RECORD_FUNCTION(code_->op_names_[inst.X].name, stack);
#endif
        code_->operators_[inst.X](stack);
        ++pc;
      } break;
      case OPN: {
        stack.push_back(inst.N);
        code_->operators_[inst.X](stack);
        ++pc;
      } break;
      case INTERFACE_CALL: {
        torch::jit::Function& method =
            peek(stack, 0, inst.N)
                .toObject()
                ->type()
                ->getMethod(code_->constants_[inst.X].toStringRef());
        method.run(stack);
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
        push(stack, std::move(value));
        ++pc;
      } break;
      case SET_ATTR: {
        auto v = pop(stack);
        auto userObj = pop(stack).toObject();
        // Mobile only: since the number of slots is not known, resize the
        // numAttributes before setSlot.
        while (userObj->type()->numAttributes() <= inst.X) {
          std::stringstream ss;
          ss << userObj->type()->numAttributes();
          userObj->type()->addAttribute(ss.str(), c10::NoneType::create());
        }
        userObj->setSlot(inst.X, std::move(v));
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
      case LIST_CONSTRUCT: {
        auto type = code_->types_[inst.X]->expect<at::ListType>();
        listConstruct(stack, type, inst.N);
        ++pc;
      } break;
      case LIST_UNPACK: {
        listUnpack(stack, inst.X);
        ++pc;
      } break;
      case TUPLE_CONSTRUCT: {
        tupleConstruct(stack, inst.X);
        ++pc;
      } break;
      case TUPLE_SLICE: {
        tupleSlice(stack, inst.X, inst.X + inst.N);
        ++pc;
      } break;
      case DICT_CONSTRUCT: {
        auto type = code_->types_[inst.X]->expect<at::DictType>();
        dictConstruct(stack, type, inst.N);
        ++pc;
      } break;
      case NAMED_TUPLE_CONSTRUCT: {
        auto type = code_->types_[inst.X]->expect<at::TupleType>();
        namedTupleConstruct(stack, type, inst.N);
        ++pc;
      } break;
      case WARN: {
        drop(stack, 1);
        TORCH_WARN(pop(stack).toStringRef());
        ++pc;
      } break;
      default:
        AT_ERROR(toString(inst.op), " is invalid.");
    }
    //  for (auto val : stack) {
    //    if (val.isTensor()) {
    //      std::cout << val.toTensor().sizes() << std::endl;
    //    } else {
    //      std::cout << val << std::endl;
    //    }
    //  }
  }
  return false;
}

IValue& InterpreterState::reg(size_t reg) {
  return *(registers_.end() - reg);
}

} // namespace mobile
} // namespace jit
} // namespace torch
