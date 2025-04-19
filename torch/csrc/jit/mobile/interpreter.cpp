#include <torch/csrc/jit/mobile/interpreter.h>

#include <ATen/core/class_type.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/function.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/operator_name.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/backends/backend_exception.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch::jit {
char const* toString(OpCode op);
std::ostream& operator<<(std::ostream& out, Instruction inst);
namespace mobile {
InterpreterState::InterpreterState(const Code& code) {
  enterFrame(code);
}

namespace {
static thread_local std::vector<DebugHandle> exception_debug_handles_;
void createObject(Stack& stack, const at::ClassTypePtr& type) {
  auto userObj = c10::ivalue::Object::create(
      c10::StrongTypePtr(type->compilation_unit(), type),
      type->numAttributes());
  push(stack, std::move(userObj));
}

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types) {
  at::TypePtr ty = pop(stack).type<c10::DynamicType>();
  for (const at::TypePtr& candidate : types) {
    if (ty->isSubtypeOf(*candidate)) {
      push(stack, true);
      return;
    }
  }
  push(stack, false);
}
} // namespace

using namespace at;

const std::vector<DebugHandle>& getInterpretersExceptionDebugHandles() {
  return exception_debug_handles_;
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

void InterpreterState::saveExceptionDebugHandles() {
  std::vector<DebugHandle> exception_debug_handles;
  for (auto frame = frames_.crbegin(); frame != frames_.crend(); frame++) {
    size_t pc = frame->getPC() - (frame != frames_.crbegin() ? 1 : 0);
    if (auto handle = frame->getDebugHandle(pc)) {
      exception_debug_handles.push_back(*handle);
    } else {
      exception_debug_handles.push_back(-1);
    }
  }
  exception_debug_handles_ = std::move(exception_debug_handles);
}

void InterpreterState::callFunction(torch::jit::Function& f, Stack& stack) {
  bool newFrame =
      f.call(stack, [&](const mobile::Code& code) { enterFrame(code); });
  (frames_.rbegin() + (newFrame ? 1 : 0))->step();
}

bool InterpreterState::run(Stack& stack) {
  while (true) {
    try {
      auto& frame = frames_.back();
      const auto& code = frame.getCode();
      const auto pc = frame.getPC();
      auto inst = frame.getInstruction();
      // If no valid debug handle found then just log pc.
      // This is possible when we did not save debug handles

      DebugHandle debug_handle = pc;
      if (auto handle = frame.getDebugHandle()) {
        debug_handle = *handle;
      }

      // std::cout << "RUNNING " << pc << " " << code.instructions_[pc];
      // if (inst.op == OP) {
      //   std::cout << ", " << code.op_names_[inst.X].name;
      //   if (!code.op_names_[inst.X].overload_name.empty()) {
      //     std::cout << "." << code.op_names_[inst.X].overload_name;
      //   }
      // }
      // std::cout << std::endl;

      // TODO(iliacher): remove the workaround after RecordFunction is in
      // Dispatcher
      // Check with iliacher if has been done.
      // Plus this is not safe as if you throw exception record function will be
      // left enabled. That is a TODO
      // NOTE: this recordFunction logic takes up ~2-3% of cpu cycles in some
      // workflows. do we need it and/or can we opt-out of
      // isRecordFunctionEnabled with a macro? if we delete it, things appear to
      // work just fine.
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
          if (inst.X < 0 ||
              static_cast<size_t>(inst.X) >= code.op_names_.size() ||
              static_cast<size_t>(inst.X) >= code.operators_.size()) {
            TORCH_CHECK(false, "Can't load op with index: ", inst.X);
          }
          RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS(
              code.op_names_[inst.X].name, debug_handle, stack);
          code.operators_[inst.X](stack);
          frame.step();
        } break;
        case OPN: {
          if (inst.X < 0 ||
              static_cast<size_t>(inst.X) >= code.op_names_.size() ||
              static_cast<size_t>(inst.X) >= code.operators_.size()) {
            TORCH_CHECK(false, "Can't load op with index: ", inst.X);
          }
          stack.emplace_back(inst.N);
          RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS(
              code.op_names_[inst.X].name, debug_handle, stack);
          code.operators_[inst.X](stack);
          frame.step();
        } break;
        case CALL: {
          auto& function = *frame.getCode().functions_.at(inst.X);
          callFunction(function, stack);
        } break;
        case INTERFACE_CALL: {
          if (inst.X < 0 ||
              static_cast<size_t>(inst.X) >= code.constants_.size()) {
            TORCH_CHECK(false, "Can't load constant with index: ", inst.X);
          }
          if (inst.N == 0 || inst.N > stack.size()) {
            TORCH_CHECK(
                false,
                "INTERFACE_CALL N=",
                inst.N,
                " not in range [1, ",
                stack.size(),
                "]");
          }
          torch::jit::Function& method =
              peek(stack, 0, inst.N)
                  .toObject()
                  ->type()
                  ->getMethod(code.constants_[inst.X].toStringRef());
          RECORD_EDGE_SCOPE_WITH_DEBUG_HANDLE_AND_INPUTS(
              method.name(), debug_handle, stack);
          callFunction(method, stack);
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
          if (inst.X < 0 ||
              static_cast<size_t>(inst.X) >= code.constants_.size()) {
            TORCH_CHECK(false, "Can't load constant with index: ", inst.X);
          }
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
          while (static_cast<int>(userObj->type()->numAttributes()) <= inst.X) {
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
          if (!frames_.empty()) {
            continue;
          }
          return false;
        case LIST_CONSTRUCT: {
          listConstruct(stack, *code.types_.at(inst.X), inst.N);
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
        case TUPLE_INDEX: {
          tupleIndex(stack);
          frame.step();
        } break;
        case RAISE_EXCEPTION: {
          raiseExceptionWithMessage(stack);
          frame.step();
        } break;
        case __IS__: {
          is(stack);
          frame.step();
        } break;
        case UN_INITIALIZED: {
          unInitialized(stack);
          frame.step();
        } break;
        case __ISNOT__: {
          isNot(stack);
          frame.step();
        } break;
        case FORMAT: {
          format(stack, inst.X);
          frame.step();
        } break;
        case DEVICE: {
          device(stack);
          frame.step();
        } break;
        case DTYPE: {
          dtype(stack);
          frame.step();
        } break;
        case DIM: {
          dim(stack);
          frame.step();
        } break;
        case __NOT__: {
          _not(stack);
          frame.step();
        } break;
        case DICT_INDEX: {
          dictIndex(stack);
          frame.step();
        } break;
        case TO_LIST: {
          toList(stack);
          frame.step();
        } break;
        case NUM_TO_TENSOR: {
          numToTensorScalar(stack);
          frame.step();
        } break;
        case IS_CUDA: {
          isCuda(stack);
          frame.step();
        } break;
        case DICT_CONSTRUCT: {
          dictConstruct(stack, *code.types_.at(inst.X), inst.N);
          frame.step();
        } break;
        case NAMED_TUPLE_CONSTRUCT: {
          namedTupleConstruct(stack, code.types_.at(inst.X), inst.N);
          frame.step();
        } break;
        case CREATE_OBJECT: {
          auto type = code.types_.at(inst.X)->expect<c10::ClassType>();
          createObject(stack, type);
          frame.step();
        } break;
        case ISINSTANCE: {
          at::ArrayRef<TypePtr> types(&code.types_.at(inst.X), inst.N);
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
          TORCH_WARN(stack.back().toStringRef());
          stack.pop_back();
          frame.step();
        } break;
        default:
          TORCH_CHECK(false, toString(inst.op), " is invalid.");
      }

      if (!prev_value) {
        enableRecordFunction(false);
      }
      // This exception must be caught first as it derived from c10::Error
    } catch (c10::BackendRuntimeException& e) {
      saveExceptionDebugHandles();
      TORCH_RETHROW(e);
    } catch (c10::Error& error) {
      // Reason for catching and rethrowing the error is so that we can
      // set the exception pc that is queried later
      saveExceptionDebugHandles();
      TORCH_RETHROW(error);
    } catch (...) {
      saveExceptionDebugHandles();
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
  TORCH_CHECK(
      reg > 0 && reg <= registers_.size(), "Invalid register index: ", reg);
  return *(registers_.end() - reg);
}

} // namespace mobile
} // namespace torch::jit
