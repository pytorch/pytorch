#include <torch/csrc/jit/mobile/interpreter.h>

#include <ATen/core/function.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/operator_name.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/backends/backend_exception.h>
#include <torch/csrc/jit/mobile/executor.h>
#include <torch/csrc/jit/mobile/observer.h>

namespace torch {
namespace jit {
char const* toString(OpCode op);
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {
InterpreterState::InterpreterState(const Code& code) {
  enterFrame(code);
}

namespace {
static thread_local DebugHandle exception_debug_handle_{-1};
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

int64_t getInterpretersExceptionDebugHandle() {
  return exception_debug_handle_;
}

void InterpreterState::enterFrame(const Code& code) {
  frames_.emplace_back(code);
  registers_.resize(registers_.size() + code.register_size_);
}

void InterpreterState::leaveFrame() {
  registers_.resize(
      registers_.size() - frames_.back().getCode().register_size_);
  frames_.pop_back();
}

void InterpreterState::saveExceptionDebugHandle() {
  const auto& frame = frames_.back();
  exception_debug_handle_ =
      frame.getCode().instructions_with_handles_.at(frame.getPC()).debug_handle;
}

bool InterpreterState::run(Stack& stack) {
  while (true) {
    try {
      auto& frame = frames_.back();
      const auto& code = frame.getCode();
      const auto pc = frame.getPC();
      auto inst_with_handle = code.instructions_with_handles_.at(pc);
      Instruction inst = inst_with_handle.instruction;
      DebugHandle debug_handle = inst_with_handle.debug_handle;
      // If no valid debug handle found then just log pc.
      // This is possible when we did not save debug handles
      debug_handle = debug_handle == -1 ? pc : debug_handle;

      // std::cout << "RUNNING " << pc << " "
      //           << code_->instructions_with_handles_[pc].instruction;
      // if (inst.op == OP) {
      //   std::cout << ", " << code_->op_names_[inst.X].name;
      //   if (!code_->op_names_[inst.X].overload_name.empty()) {
      //     std::cout << "." << code_->op_names_[inst.X].overload_name;
      //   }
      // }
      // std::cout << std::endl;

      // TODO(iliacher): remove the workaround after RecordFunction is in
      // Dispatcher
      // Check with iliacher if has been done.
      // Plus this is not safe as if you throw exception record function will be
      // left enabled. That is a TODO
      bool prev_value = isRecordFunctionEnabled();
      if (!prev_value) {
        // enable only for the RecordFunction
        enableRecordFunction(true);
      }
      switch (inst.op) {
        case OP: {
          if (at::hasGlobalCallbacks()) {
            if (auto* mobile_debug_info = static_cast<MobileDebugInfo*>(
                    c10::ThreadLocalDebugInfo::get(
                        c10::DebugInfoKind::MOBILE_RUNTIME_INFO))) {
              mobile_debug_info->setOpIdx(pc);
            }
          }

          RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS(
              code.op_names_[inst.X].name, debug_handle, stack);
          code.operators_[inst.X](stack);
          frame.step();
        } break;
        case OPN: {
          stack.push_back(inst.N);
          RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS(
              code.op_names_[inst.X].name, debug_handle, stack);
          code.operators_[inst.X](stack);
          frame.step();
        } break;
        case INTERFACE_CALL: {
          torch::jit::Function& method =
              peek(stack, 0, inst.N)
                  .toObject()
                  ->type()
                  ->getMethod(code.constants_[inst.X].toStringRef());
          RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS(
              method.name(), debug_handle, stack);

          if (method.hasExecutor()) {
            auto& plan = method.get_executor().getPlanFor(stack, 0);
            frame.step();
            enterFrame(toEdgeExecutionPlan(plan).getCode());
            continue;
          }

          method.run(stack);
          frame.step();
        } break;
        case LOAD:
          stack.emplace_back(reg(inst.X));
          frame.step();
          break;
        case MOVE:
          stack.emplace_back(std::move(reg(inst.X)));
          frame.step();
          break;
        case STORE:
          reg(inst.X) = pop(stack);
          frame.step();
          break;
        case STOREN:
          for (size_t i = inst.N; i > 0; --i) {
            reg(inst.X + i - 1) = pop(stack);
          }
          frame.step();
          break;
        case DROP:
          pop(stack);
          frame.step();
          break;
        case DROPR:
          reg(inst.X) = IValue();
          frame.step();
          break;
        case LOADC:
          stack.emplace_back(code.constants_[inst.X]);
          frame.step();
          break;
        case GET_ATTR: {
          auto userObj = pop(stack).toObject();
          auto value = userObj->getSlot(inst.X);
          push(stack, std::move(value));
          frame.step();
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
          frame.step();
        } break;
        case JF:
          frame.jump(pop(stack).toBool() ? 1 : inst.X);
          break;
        case JMP:
          frame.jump(inst.X);
          break;
        case LOOP: {
          // stack: iteration_count, max_iter, cond, loop_carried_deps...
          auto sframe = stack.end() - (inst.N + 1);
          int64_t trip_count = sframe[0].toInt();
          int64_t max_trip_count = sframe[1].toInt();
          bool cond = sframe[2].toBool();
          if (trip_count < max_trip_count && cond) {
            sframe[2] = trip_count;
            sframe[0] = trip_count + 1;
            frame.step();
          } else {
            size_t n_loop_carried = inst.N - 2;
            for (const auto i : c10::irange(n_loop_carried)) {
              sframe[i] = std::move(sframe[i + 3]);
            }
            drop(stack, 3); // iteration_count, max_iter, cond
            frame.jump(inst.X);
          }
        } break;
        case RET:
          leaveFrame();
          if (frames_.size() > 0) {
            continue;
          }
          return false;
        case LIST_CONSTRUCT: {
          const auto& type = code.types_[inst.X]->expectRef<at::ListType>();
          listConstruct(stack, type, inst.N);
          frame.step();
        } break;
        case LIST_UNPACK: {
          listUnpack(stack, inst.X);
          frame.step();
        } break;
        case TUPLE_CONSTRUCT: {
          tupleConstruct(stack, inst.X);
          frame.step();
        } break;
        case TUPLE_SLICE: {
          tupleSlice(stack, inst.X, inst.X + inst.N);
          frame.step();
        } break;
        case DICT_CONSTRUCT: {
          const auto& type = code.types_[inst.X]->expectRef<at::DictType>();
          dictConstruct(stack, type, inst.N);
          frame.step();
        } break;
        case NAMED_TUPLE_CONSTRUCT: {
          namedTupleConstruct(
              stack, code.types_[inst.X]->expect<at::TupleType>(), inst.N);
          frame.step();
        } break;
        case CREATE_OBJECT: {
          auto type = code.types_[inst.X]->expect<c10::ClassType>();
          createObject(stack, type);
          frame.step();
        } break;
        case ISINSTANCE: {
          at::ArrayRef<TypePtr> types(
              &(code.types_[inst.X]), &(code.types_[inst.X + inst.N]));
          isinstance(stack, types);
          frame.step();
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
          frame.step();
        } break;
        default:
          AT_ERROR(toString(inst.op), " is invalid.");
      }

      if (!prev_value) {
        enableRecordFunction(false);
      }
      // This exception must be caught first as it derived from c10::Error
    } catch (c10::BackendRuntimeException& e) {
      saveExceptionDebugHandle();
      TORCH_RETHROW(e);
    } catch (c10::Error& error) {
      // Reason for catching and rethrowing the error is so that we can
      // set the exception pc that is queried later
      saveExceptionDebugHandle();
      TORCH_RETHROW(error);
    } catch (...) {
      saveExceptionDebugHandle();
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
