#include <torch/csrc/jit/mobile/interpreter.h>

#include <ATen/core/function.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#include <ATen/record_function.h>
#include <torch/csrc/jit/mobile/observer.h>

namespace torch {
namespace jit {
char const* toString(OpCode op);
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {
InterpreterState::InterpreterState(std::shared_ptr<Code> code)
    : code_(std::move(code)) {
  registers_.resize(code_->register_size_);
}

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local int64_t exception_pc_{-1};
void createObject(Stack& stack, const at::ClassTypePtr& type) {
  auto userObj = c10::ivalue::Object::create(
      c10::StrongTypePtr(type->compilation_unit(), type),
      type->numAttributes());
  push(stack, std::move(userObj));
}

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types) {
  at::TypePtr ty = pop(stack).type();
  for (const at::TypePtr& candidate : types) {
    if (ty->isSubtypeOf(candidate)) {
      push(stack, true);
      return;
    }
  }
  push(stack, false);
}
} // namespace

using namespace at;

int64_t getInterpretersExceptionPC() {
  return exception_pc_;
}

bool InterpreterState::run(Stack& stack) {
  size_t pc = 0;
  while (true) {
    try {
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
          if (at::hasGlobalCallbacks()) {
            if (auto* mobile_debug_info = static_cast<MobileDebugInfo*>(
                    c10::ThreadLocalDebugInfo::get(
                        c10::DebugInfoKind::MOBILE_RUNTIME_INFO))) {
              mobile_debug_info->setOpIdx(pc);
            }
          }

          // TODO(iliacher): remove the workaround after RecordFunction is in
          // Dispatcher
          bool prev_value = isRecordFunctionEnabled();
          if (!prev_value) {
            // enable only for the RecordFunction
            enableRecordFunction(true);
          }
          RECORD_USER_SCOPE_WITH_INPUTS(code_->op_names_[inst.X].name, stack);
          if (!prev_value) {
            enableRecordFunction(false);
          }
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
          // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
          while (userObj->type()->numAttributes() <= inst.X) {
            std::stringstream ss;
            ss << userObj->type()->numAttributes();
            userObj->type()->addAttribute(ss.str(), c10::NoneType::get());
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
          const auto& type = code_->types_[inst.X]->expectRef<at::ListType>();
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
          const auto& type = code_->types_[inst.X]->expectRef<at::DictType>();
          dictConstruct(stack, type, inst.N);
          ++pc;
        } break;
        case NAMED_TUPLE_CONSTRUCT: {
          namedTupleConstruct(
              stack, code_->types_[inst.X]->expect<at::TupleType>(), inst.N);
          ++pc;
        } break;
        case CREATE_OBJECT: {
          auto type = code_->types_[inst.X]->expect<c10::ClassType>();
          createObject(stack, type);
          ++pc;
        } break;
        case ISINSTANCE: {
          at::ArrayRef<TypePtr> types(
              &(code_->types_[inst.X]), &(code_->types_[inst.X + inst.N]));
          isinstance(stack, types);
          ++pc;
        } break;
        case WARN: {
          drop(stack, 1);
          // Note: Please don't move the pop(stack) code below into the
          // TORCH_WARN macro since TORCH_WARN fails to evaluate its arguments
          // when STRIP_ERROR_MESSAGES is defined (which happens for production
          // mobile builds). This will cause the stack to be in an inconsistent
          // state. It has previously resulted in a SEV (S22350).
          const auto& sref = stack.back().toStringRef();
          TORCH_WARN(sref);
          stack.pop_back();
          ++pc;
        } break;
        default:
          AT_ERROR(toString(inst.op), " is invalid.");
      }
    } catch (c10::Error& error) {
      // Reason for catching and rethrowing the error is so that we can
      // set the exception pc that is queried later
      exception_pc_ = pc;
      TORCH_RETHROW(error);
    } catch (...) {
      exception_pc_ = pc;
      throw;
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
