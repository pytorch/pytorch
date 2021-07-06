#include <torch/csrc/jit/runtime/interpreter.h>

#include <ATen/Parallel.h>
#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>
#include <c10/core/thread_pool.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/runtime/interpreter/code_impl.h>
#include <torch/csrc/jit/runtime/interpreter/frame.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/profiling_record.h>
#include <torch/csrc/jit/runtime/script_profile.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#ifdef USE_RPC
#include <torch/csrc/distributed/autograd/context/container.h>
using torch::distributed::autograd::DistAutogradContainer;
#endif

#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

using CodeImpl = interpreter::CodeImpl;

// Before we translate to intepreter instructions, we do
// some preprocessing of the graph to turn it into a form that is closer
// to what the instructions will look like.
// In particular we:
// *  Computes whether a input to a node is the last use, so we can issue MOVE
//    rather than LOAD instructions.
// *  Drop nodes are inserted for any node that is unused to create a dummy use
//    that will cause the interpreter to free the node.
//    A drop node just pops its input off the stack to  ensure the interpreter
//    releases references to nodes that are never used. Drop nodes are also
//    inserted when the last use of a node is in some conditionally run control
//    flow (e.g. one side of an If) and the interpreter must free the node only
//    after the control flow has reconverged
// Outputs are:
// * graph - the post processed copy of g
// * move_flags[n] - a list of booleans, one for each input,
//   indicating whether this is the last use of the value. The interpreter
//   should generate a move rather than a copy in this case.

TensorTypePtr tensorTypeInCurrentExecutionContext(const at::Tensor& t) {
  if (!t.defined()) {
    return TensorType::get()->withUndefined();
  }
  auto r = TensorType::create(t);
  if (!at::GradMode::is_enabled()) {
    return r->withRequiresGrad(false);
  }
  return r;
}

namespace {
inline int64_t getDistAutogradContextId() {
#ifdef USE_RPC
  return DistAutogradContainer::currentContextId();
#else
  return 0;
#endif
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local InterpreterStateImpl* tls_int_state_ptr_ = nullptr;
struct TLSCurrentInterpreterGuard {
  TLSCurrentInterpreterGuard(InterpreterStateImpl* state) {
    prev_state_ = tls_int_state_ptr_;
    tls_int_state_ptr_ = state;
  }

  ~TLSCurrentInterpreterGuard() {
    tls_int_state_ptr_ = prev_state_;
  }

 private:
  InterpreterStateImpl* prev_state_;
};

// InterpreterState state that and used to compute a Code
struct InterpreterStateImpl : c10::intrusive_ptr_target {
  InterpreterStateImpl(const Code& code, TaskLauncher taskLauncher)
      : taskLauncher_(std::move(taskLauncher)) {
    enterFrame(code, 0);
  }

 private:
  using Frame = torch::jit::interpreter::Frame;
  struct WarnedNodes {
   public:
    // Inserts idx into warned_nodes_, returns a boolean indicates whether
    // insertion actually happened (idx wasn't originally in the set).
    bool insert(int32_t idx) {
      std::unique_lock<std::mutex> lock(mutex_);
      return warned_nodes_.insert(idx).second;
    }

   private:
    std::mutex mutex_;
    std::unordered_set<int32_t> warned_nodes_;
  };

  WarnedNodes warned_nodes_;

  // if we need to suspend, where do we reset the stack?
  // answer: to where it was when we were called, not
  // including any inputs to this function
  int64_t stack_start_ = -1;
  c10::intrusive_ptr<Future> future_;
  TaskLauncher taskLauncher_;

  // this holds all the tensors for this interpreter run
  // we don't bother minimizing the size of this vector, since the extra
  // memory used by the pointers in this will be small
  // instead we are very aggresive about releasing tensors when they become dead
  // to make sure memory management happens efficiently.
  // We optimize for the case where derivatives are run with retain_graph=False
  // in the case where it is true, then the interpreter and this array get
  // copied if this every becomes a bottleneck then we _should_ consider
  // minimizing the total number or register
  std::vector<IValue> registers;

  // A stack of objects that have been __enter__'d.
  std::vector<IValue> entered_objects;

  std::vector<Frame> frames;

  c10::intrusive_ptr<InterpreterStateImpl> intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this);
    return c10::intrusive_ptr<InterpreterStateImpl>::reclaim(this);
  }

  void enterFrame(const Code& code, size_t base_pointer) {
    frames.emplace_back(Frame{code.pImpl, 0, base_pointer, c10::nullopt});
    registers.resize(registers.size() + code.pImpl->register_size_);
  }

  void leaveFrame() {
    registers.resize(registers.size() - frames.back().function->register_size_);
    frames.pop_back();
  }

  // relative to the end of the register list so that when we call
  // functions we are referring to the registers of the currenly executing
  // function.
  IValue& reg(size_t reg) {
    return *(registers.end() - reg);
  }

  void dump(std::ostream& out, const Stack& stack) const {
    out << "Stack:\n";
    for (const auto& val : stack) {
      out << val;
      out << "\n";
    }
  }

  void runBuiltinFunction(Stack& stack, Function* fn) {
    // BuiltinOpFunction directly invokes a void(Stack&) to implement
    // custom C++ classes. Call run() here with the stack, and we will
    // get the results from that C++ method back in the stack. Advance
    // the PC by 1 without adding any new frame.
    fn->run(stack);
    ++frames.back().pc;
  }

  void runGraphFunction(Stack& stack, Function* fn) {
    const Code& code =
        // consider passing
        // `frames.back().function->remaining_bailout_depth_` into
        // `get_executor().getPlanFor()` to propagate caller's depth
        // restrictions onto children while this strategy has a
        // potential to reduce the number of compilations for too
        // dynamic callers we might miss opportunities where a caller is
        // dynamic but a callee gets stable arguments
        fn->get_executor()
            .getPlanFor(stack, GraphExecutor::getDefaultNumBailOuts())
            .code;
    ++frames.back().pc;
    enterFrame(code, stack.size() - code.num_inputs());
    checkAndStartRecordFunction(frames.back(), stack);
  }

#if defined(__GNUC__) || defined(__clang__)
#define JIT_USE_COMPUTED_GOTO
#endif
// Primitives for making interpreter internal state transitions.
// We maintain two local variables as the internal interpreter state:
// `frame` will be the current frame that the interpreter operatos on.
// `inst` will the current instruction pointed to by program counter.
//
// Instruction blocks should be always declared through `INST` macro and
// the instruction body should always start with a `INST_GUARD` declaration.
// Also blocks should be ended properly with either `INST_NEXT` (for going
// to the next instruction), or `INST_DISPATCH` (for jumping to a computed
// position using `INST_FETCH`).
#define INST_FETCH(X) (frame.function->instructions_[frame.pc += (X)])
#define INST_GUARD                                   \
  profiling::InstructionSpan span {                  \
    *frame.function->instructions_source()[frame.pc] \
  }
#if defined(JIT_USE_COMPUTED_GOTO)
#define INST(NAME) \
  NAME:            \
  label_##NAME
#define INST_DISPATCH goto* dispatch_table[inst.op]
#else
#define INST(NAME) NAME
#define INST_DISPATCH break
#endif
#define INST_NEXT       \
  inst = INST_FETCH(1); \
  INST_DISPATCH

  bool runImpl(Stack& stack) {
    // if we have never run before, then we might have to return the
    // stack when we suspend, record where it starts so we return the right
    // stack
    if (stack_start_ == -1) {
      TORCH_INTERNAL_ASSERT(stack.size() >= frames.back().function->n_inputs);
      stack_start_ = stack.size() - frames.back().function->n_inputs;
    } else {
      // during restarts, all of the stack is always our own, so we leave
      // nothing
      stack_start_ = 0;
    }

    TLSCurrentInterpreterGuard g(this);
    if (frames.back().pc == 0 && stack_start_ == 0) {
      checkAndStartRecordFunction(frames.back(), stack);
    }

#if defined(JIT_USE_COMPUTED_GOTO)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    static void* dispatch_table[] = {
#define DISPATCH_TABLE_ENTRY(op, _) &&label_##op,
        FORALL_OPCODES(DISPATCH_TABLE_ENTRY)
#undef DISPATCH_TABLE_ENTRY
    };
#endif

    try {
      while (true) {
        Frame& frame = frames.back();
        Instruction inst = INST_FETCH(0);
        switch (inst.op) {
          case INST(ENTER): {
            INST_GUARD;
            const auto& obj = peek(stack, 0, 1);
            TORCH_INTERNAL_ASSERT(obj.isObject());
            entered_objects.push_back(obj);
          }
            INST_NEXT;
          case INST(EXIT): {
            INST_GUARD;
            auto obj = entered_objects.back().toObject();
            auto& f = obj->type()->getMethod("__exit__");
            push(stack, std::move(obj));
            entered_objects.pop_back();
            push(stack, IValue());
            push(stack, IValue());
            push(stack, IValue());
            runGraphFunction(stack, &f);
            continue;
          }
          case INST(OP): {
            INST_GUARD;
            frame.function->operator_table_[inst.X](&stack);
          }
            INST_NEXT;
          case INST(OPN): {
            INST_GUARD;
            stack.push_back(inst.N);
            frame.function->operator_table_[inst.X](&stack);
          }
            INST_NEXT;
          case INST(LOAD): {
            INST_GUARD;
            stack.emplace_back(reg(inst.X));
          }
            INST_NEXT;
          case INST(MOVE): {
            INST_GUARD;
            stack.emplace_back(std::move(reg(inst.X)));
          }
            INST_NEXT;
          case INST(STORE): {
            INST_GUARD;
            reg(inst.X) = pop(stack);
          }
            INST_NEXT;
          case INST(STOREN): {
            INST_GUARD;
            for (size_t i = inst.N; i > 0; --i) {
              reg(inst.X + i - 1) = pop(stack);
            }
          }
            INST_NEXT;
          case INST(DROP): {
            INST_GUARD;
            pop(stack);
          }
            INST_NEXT;
          case INST(DROPR): {
            INST_GUARD;
            reg(inst.X) = IValue();
          }
            INST_NEXT;
          case INST(LOADC): {
            INST_GUARD;
            stack.emplace_back(frame.function->constant_table_[inst.X]);
          }
            INST_NEXT;
          case INST(GET_ATTR): {
            INST_GUARD;
            auto userObj = pop(stack).toObject();
            auto value = userObj->getSlot(inst.X);
            push(stack, std::move(value));
          }
            INST_NEXT;
          case INST(SET_ATTR): {
            INST_GUARD;
            auto v = pop(stack);
            auto userObj = pop(stack).toObject();
            userObj->setSlot(inst.X, std::move(v));
          }
            INST_NEXT;
          case INST(JF): {
            INST_GUARD;
            if (pop(stack).toBool()) {
              inst = INST_FETCH(1);
            } else {
              inst = INST_FETCH(inst.X);
            }
          }
            INST_DISPATCH;
          case INST(JMP): {
            INST_GUARD;
            inst = INST_FETCH(inst.X);
          }
            INST_DISPATCH;
          case INST(LOOP): {
            INST_GUARD;
            // stack: iteration_count, max_iter, cond, loop_carried_deps...
            auto fr = stack.end() - (inst.N + 1);
            int64_t trip_count = fr[0].toInt();
            int64_t max_trip_count = fr[1].toInt();
            bool cond = fr[2].toBool();
            if (trip_count < max_trip_count && cond) {
              fr[2] = trip_count;
              fr[0] = trip_count + 1;
              inst = INST_FETCH(1);
            } else {
              size_t n_loop_carried = inst.N - 2;
              for (size_t i = 0; i < n_loop_carried; ++i) {
                fr[i] = std::move(fr[i + 3]);
              }
              drop(stack, 3); // iteration_count, max_iter, cond
              inst = INST_FETCH(inst.X);
            }
          }
            INST_DISPATCH;
          case INST(CALL): {
            INST_GUARD;
            Function* fn = frame.function->function_table_[inst.X];
            if (!fn->isGraphFunction()) {
              runBuiltinFunction(stack, fn);
            } else {
              runGraphFunction(stack, fn);
            }
            continue;
          }
          case INST(INTERFACE_CALL): {
            INST_GUARD;
            // note the hash table lookup to find the function
            // this can be more optimized if necessary, caching parts
            // of the hashing computation or storing the offset when
            // the object is turned into an interface

            // consider passing
            // `frames.back().function->remaining_bailout_depth_` into
            // `get_executor().getPlanFor()` to propagate caller's depth
            // restrictions onto children while this strategy has a potential to
            // reduce the number of compilations for too dynamic callers we
            // might miss opportunities where a caller is dynamic but a callee
            // gets stable arguments
            Function& function =
                peek(stack, 0, inst.N)
                    .toObject()
                    ->type()
                    ->getMethod(
                        frame.function->constant_table_[inst.X].toStringRef());
            if (!function.isGraphFunction()) {
              runBuiltinFunction(stack, &function);
            } else {
              runGraphFunction(stack, &function);
            }
            continue;
          }
          case INST(RET): {
            if (frames.size() > 1) {
              leaveFrame();
              continue;
            }
            if (future_) {
              auto num_outputs = frames.back().function->n_outputs;
              if (num_outputs == 1) {
                future_->markCompleted(stack.back());
              } else {
                future_->markCompleted(c10::ivalue::Tuple::create(
                    jit::last(stack, num_outputs).vec()));
              }
            }
            // destroy the last frame and call RecordFunction's end callbacks
            leaveFrame();
            return false;
          }
          case INST(WAIT): {
            INST_GUARD;
            auto future = stack.back().toFuture();
            if (!future->completed()) {
              getOrCreateFuture();

              // callback needs to be a struct rather than a lambda so that
              // we can move the stack to the other thread
              struct Callback {
                Callback(
                    c10::intrusive_ptr<InterpreterStateImpl> state,
                    Stack stack)
                    : stateImpl_(std::move(state)),
                      state_(stateImpl_),
                      stack_(std::move(stack)) {
                  dist_autograd_context_id_ = getDistAutogradContextId();
                  state_ = InterpreterState(stateImpl_);
                }
                void operator()(c10::ivalue::Future& /* unused */) {
                  stateImpl_->taskLauncher_(InterpreterContinuation(
                      state_,
                      std::move(stack_),
                      dist_autograd_context_id_,
                      std::move(tls_state_)));
                }

               private:
                c10::intrusive_ptr<InterpreterStateImpl> stateImpl_;
                InterpreterState state_;
                Stack stack_;
                int64_t dist_autograd_context_id_;
                // preserve the original ThreadLocalState
                at::ThreadLocalState tls_state_;
              };

              // we are suspending, so we need to reset the stack to where we
              // started if it started empty, except for the inputs we can avoid
              // a true copy by swapping, which leaves the original stack empty.
              Stack copied;
              if (stack_start_ == 0) {
                copied.swap(stack);
              } else {
                copied.insert(
                    copied.begin(),
                    std::make_move_iterator(stack.begin() + stack_start_),
                    std::make_move_iterator(stack.end()));
                stack.resize(stack_start_);
              }
              // save pc into the frame so we continue here when restored
              future->addCallback(
                  Callback(intrusive_from_this(), std::move(copied)));

              return true;
            }
            stack.pop_back();
            stack.emplace_back(future->value());
          }
            INST_NEXT;
          case INST(PROFILE_OP): {
            INST_GUARD;
            auto& frame_id_ref = frame.id;
            if (!frame_id_ref.has_value()) {
              frame_id_ref = Frame::genId();
            }
            const auto& callback =
                frame.function->profile_function_table_[inst.X];
            push(stack, c10::IValue{static_cast<int64_t>(*frame_id_ref)});
            callback(stack);
          }
            INST_NEXT;
          case INST(FAIL_GUARD): {
            INST_GUARD;
            // patch FAIL_GUARD back to GUARD
            GRAPH_DEBUG(
                "Bailout ", inst.X, " triggered via bailout_requests_!");
            frame.function->instructions_[frame.pc].op = GUARD;
            push(stack, false);
          }
            INST_NEXT;
          case INST(TYPECHECK): {
            INST_GUARD;
            int num_inputs = inst.N, i = 0;
            // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
            TORCH_INTERNAL_ASSERT(stack.size() >= num_inputs && num_inputs > 0);
            // Check every input's shape against profiled (expected) shape.
            for (i = 0; i < num_inputs; i++) {
              auto& input = peek(stack, i, num_inputs);
              auto& t = input.toTensor();
              const TypePtr& expected = frame.function->type_table_[inst.X + i];
              auto* expected_type = expected->castRaw<TensorType>();
              if (t.defined() && !expected_type->matchTensor(t)) {
                push(stack, false);
                break;
              }
            }
            if (i == num_inputs) {
              push(stack, true);
            }
          }
            INST_NEXT;
          case INST(GUARD): {
            INST_GUARD;
            if (!stack.back().isTensor()) {
              // stack.back() is an Uninitialized IValue and this is a guard
              // on a block output. Uninitialized IValues are never used
              // so it's safe to pass this guard check
              push(stack, true);
            } else {
              auto& t = stack.back().toTensor();
              const TypePtr& expected = frame.function->type_table_[inst.X];
              auto* expected_type = expected->castRaw<TensorType>();
              if (t.defined() &&
                  !frames.back().symbols2dims.bindSymbolicShapes(
                      t.sizes(), expected_type->symbolic_sizes())) {
                push(stack, false);
              } else {
                push(stack, expected_type->matchTensor(t));
              }
            }
          }
            INST_NEXT;
          case INST(TAIL_CALL): {
            INST_GUARD;
            GRAPH_DEBUG("running TAIL_CALL for ", inst.X);
            frame.function->function_table_[inst.X]->ensure_defined();
            size_t remaining_bailout_depth =
                frame.function->remaining_bailout_depth_ > 0
                ? frame.function->remaining_bailout_depth_ - 1
                : 0;
            const Code& code = frame.function->function_table_[inst.X]
                                   ->get_executor()
                                   .getPlanFor(stack, remaining_bailout_depth)
                                   .code;
            size_t num_inputs = code.num_inputs();
            size_t base_pointer = frame.base_pointer;
            TORCH_INTERNAL_ASSERT(stack.size() >= num_inputs);
            size_t inputs_start = stack.size() - num_inputs;
            for (size_t i = 0; i < num_inputs; ++i) {
              stack.at(base_pointer + i) =
                  std::move(stack.at(inputs_start + i));
            }
            stack.resize(base_pointer + num_inputs);
            leaveFrame();
            enterFrame(code, base_pointer);
            checkAndStartRecordFunction(frames.back(), stack);
            continue;
          }
          case INST(LIST_UNPACK): {
            INST_GUARD;
            listUnpack(stack, inst.X);
          }
            INST_NEXT;
          case INST(TUPLE_CONSTRUCT): {
            INST_GUARD;
            tupleConstruct(stack, inst.X);
          }
            INST_NEXT;
          case INST(TUPLE_SLICE): {
            INST_GUARD;
            tupleSlice(stack, inst.X, inst.X + inst.N);
          }
            INST_NEXT;
          case INST(NAMED_TUPLE_CONSTRUCT): {
            INST_GUARD;
            namedTupleConstruct(
                stack,
                frame.function->type_table_[inst.X]->expect<TupleType>(),
                inst.N);
          }
            INST_NEXT;
          case INST(LIST_CONSTRUCT): {
            INST_GUARD;
            const auto& type =
                frame.function->type_table_[inst.X]->expectRef<ListType>();
            listConstruct(stack, type, inst.N);
          }
            INST_NEXT;
          case INST(DICT_CONSTRUCT): {
            INST_GUARD;
            const auto& type =
                frame.function->type_table_[inst.X]->expectRef<DictType>();
            dictConstruct(stack, type, inst.N);
          }
            INST_NEXT;
          case INST(CREATE_OBJECT): {
            INST_GUARD;
            auto type =
                frame.function->type_table_[inst.X]->expect<ClassType>();
            createObject(stack, type);
          }
            INST_NEXT;
          case INST(ISINSTANCE): {
            INST_GUARD;
            at::ArrayRef<TypePtr> types(
                &(frame.function->type_table_[inst.X]),
                &(frame.function->type_table_[inst.X + inst.N]));
            isinstance(stack, types);
          }
            INST_NEXT;
          case INST(FORK): {
            INST_GUARD;
            // Move inputs to a separate stack
            Function* forked_fn = frame.function->function_table_[inst.X];
            InterpreterState forked_interpreter(
                forked_fn->get_executor()
                    .getPlanFor(stack, GraphExecutor::getDefaultNumBailOuts())
                    .code,
                taskLauncher_);
            InterpreterContinuation continuation(
                forked_interpreter,
                Stack(stack.end() - inst.N, stack.end()),
                getDistAutogradContextId());
            drop(stack, inst.N);
            push(stack, forked_interpreter.getFuture());
            taskLauncher_(std::move(continuation));
          }
            INST_NEXT;
          case INST(WARN): {
            INST_GUARD;
            // Keeps track of which WARN instruction has been executed before,
            // we only want to execute each WARN once to match default Python
            // warning behavior.
            bool need_warn = true;
            if (inst.X != -1) {
              need_warn = warned_nodes_.insert(inst.X);
            }

            Node* node =
                frames.back().function->instructions_source_.at(frame.pc);
            auto range = node->sourceRange().source();
            if (range->filename()) {
              drop(stack, 1);
              const auto& msg = stack.back().toStringRef();
              if (need_warn) {
                auto line = range->starting_line_no() +
                    range->lineno_for_offset(node->sourceRange().start());
                c10::SourceLocation location{
                    "", range->filename()->c_str(), uint32_t(line)};
                // Sends the warning to the warning handler with the
                // "verbatim" flag. This flag ensures the warning handler
                // will print the exception as configured.
                c10::Warning::warn(location, msg, /*verbatim=*/true);
              }
              stack.pop_back();
            } else {
              const auto& msg = stack.back().toStringRef();
              if (need_warn) {
                TORCH_WARN(msg);
              }
              stack.pop_back();
            }
          }
            INST_NEXT;
        }
      }
    } catch (std::exception& e) {
      for (auto it = entered_objects.rbegin(), end = entered_objects.rend();
           it != end;
           ++it) {
        auto& f = it->toObject()->type()->getMethod("__exit__");
        Stack stack;
        push(stack, *it);
        push(stack, IValue());
        push(stack, IValue());
        push(stack, IValue());
        try {
          f.run(stack);
        } catch (std::exception& e) {
          std::ostringstream ss;
          ss << "The following operation failed in the TorchScript interpreter.\n";
          formatStackTrace(ss);
          ss << "RuntimeError: " << ExceptionMessage(e) << "\n";
        }
      }
      bool is_jit_exception = dynamic_cast<JITException*>(&e);
      // Janky af.  See https://github.com/pytorch/pytorch/issues/54612
      auto* not_implemented_error = dynamic_cast<c10::NotImplementedError*>(&e);
      handleError(ExceptionMessage(e), is_jit_exception, not_implemented_error);
      return false;
    }
  }

#undef INST_NEXT
#undef INST_DISPATCH
#undef INST
#undef INST_GUARD
#undef INST_FETCH
#undef JIT_USE_COMPUTED_GOTO

  void formatStackTrace(std::ostream& out) {
    format_stack_trace(out, callstack());
  }

  void handleError(
      const ExceptionMessage& msg,
      bool is_jit_exception,
      c10::NotImplementedError* not_implemented_error) {
    std::ostringstream ss;
    ss << "The following operation failed in the TorchScript interpreter.\n";
    formatStackTrace(ss);
    ss << "RuntimeError: " << msg << "\n";
    if (future_) {
      future_->setError(std::make_exception_ptr(Future::FutureError(ss.str())));
    } else if (is_jit_exception) {
      throw JITException(ss.str());
    } else if (not_implemented_error) {
      throw c10::NotImplementedError(
          ss.str(),
          not_implemented_error->backtrace(),
          not_implemented_error->caller());
    } else {
      throw std::runtime_error(ss.str());
    }
  }

  static void checkAndStartRecordFunction(Frame& frame, Stack& stack) {
    bool pre_sampled = false;
    if (!frame.record_function && at::hasCallbacks() &&
        at::shouldRunRecordFunction(&pre_sampled)) {
      auto rec_fn = std::make_unique<at::RecordFunction>(
          at::RecordScope::TORCHSCRIPT_FUNCTION, pre_sampled);
      if (rec_fn->isActive()) {
        if (rec_fn->needsInputs()) {
          rec_fn->before(
              frame.function->function_name_,
              last(stack, frame.function->n_inputs));
        } else {
          rec_fn->before(frame.function->function_name_);
        }
        frame.record_function = std::move(rec_fn);
      }
    }
  }

 public:
  std::vector<StackEntry> callstack() const {
    std::vector<StackEntry> entries;
    for (size_t i = 0; i < frames.size(); ++i) {
      const Frame& frame = frames[i];
      std::string previous_fn_name = frame.function->function_name_;
      size_t pc = frame.pc;
      // CALL nodes have already advanced the pc, so
      // undo that to report the call node
      if (i + 1 < frames.size()) {
        --pc;
      }

      Node* node = frame.function->instructions_source_[pc];
      if (node->callstack()) {
        for (const auto& p : (*node->callstack())->vec()) {
          entries.emplace_back(StackEntry{previous_fn_name, std::get<1>(p)});
          previous_fn_name = std::get<0>(p)->name();
        }
      }
      entries.emplace_back(StackEntry{previous_fn_name, node->sourceRange()});
    }
    return entries;
  }

  c10::intrusive_ptr<Future> getOrCreateFuture() {
    if (!future_) {
      future_ =
          c10::make_intrusive<Future>(frames.front().function->return_type_);
    }
    return future_;
  }

  c10::intrusive_ptr<Future> runAsync(Stack& stack) {
    getOrCreateFuture();
    runImpl(stack);
    return future_;
  }

  void run(Stack& stack) {
    if (runImpl(stack)) {
      future_->wait();

      auto num_outputs = frames.front().function->n_outputs;
      if (num_outputs == 1) {
        push(stack, future_->value());
      } else {
        auto tuple = future_->value().toTuple();
        for (const IValue& value : tuple->elements()) {
          push(stack, value);
        }
      }
    }
  }
};

std::vector<StackEntry> currentCallstack() {
  if (tls_int_state_ptr_) {
    auto cs = tls_int_state_ptr_->callstack();
    std::reverse(cs.begin(), cs.end());
    return cs;
  }
  return std::vector<StackEntry>();
}

std::ostream& operator<<(std::ostream& out, const Code& code) {
  out << *code.pImpl->graph_ << "\n";
  code.pImpl->dump(out);
  return out;
}

Code::Code(
    const std::shared_ptr<Graph>& graph,
    std::string function_name,
    size_t remaining_bailout_depth)
    : pImpl(new CodeImpl(
          graph,
          std::move(function_name),
          remaining_bailout_depth)) {}

Code::Code(CodeImpl* codeImpl) : pImpl(codeImpl) {}
Code::~Code() = default;

MobileCode::MobileCode(
    const std::shared_ptr<Graph>& graph,
    std::string function_name,
    bool emit_default_input_instructions,
    size_t remaining_bailout_depth)
    : Code(new interpreter::MobileCodeImpl(
          graph,
          std::move(function_name),
          emit_default_input_instructions,
          remaining_bailout_depth)) {}

MobileCode::~MobileCode() = default;

const std::vector<GraphExecutor*>& Code::grad_executors() {
  return pImpl->grad_executors();
}

const std::vector<GraphExecutor*>& Code::diff_graph_op_executors() {
  return pImpl->diff_graph_op_executors();
}

size_t Code::num_bailouts() const {
  return pImpl->type_table_.size();
}

void Code::request_bailout(size_t index) {
  pImpl->request_bailout(index);
}

size_t Code::num_inputs() const {
  return pImpl->n_inputs;
}

size_t Code::num_outputs() const {
  return pImpl->n_outputs;
}

const std::vector<c10::IValue>& Code::constant_table() const {
  return pImpl->constant_table();
}

const std::vector<Instruction>& Code::instructions() const {
  return pImpl->instructions();
}

const std::unordered_map<std::string, size_t>& Code::op_to_num_specified_args()
    const {
  return pImpl->op_to_num_specified_args();
}

const std::vector<Node*>& Code::instructions_source() const {
  return pImpl->instructions_source();
}

const std::vector<TypePtr>& Code::type_table() const {
  return pImpl->type_table_;
}

size_t Code::register_size() const {
  return pImpl->register_size_;
}

InterpreterState::InterpreterState(const Code& code, TaskLauncher taskLauncher)
    : pImpl(c10::make_intrusive<InterpreterStateImpl>(
          code,
          std::move(taskLauncher))) {}
InterpreterState::~InterpreterState() = default;

void InterpreterState::run(Stack& stack) {
  static_cast<InterpreterStateImpl*>(pImpl.get())->run(stack);
}

c10::intrusive_ptr<Future> InterpreterState::runAsync(Stack& stack) {
  return static_cast<InterpreterStateImpl*>(pImpl.get())->runAsync(stack);
}

c10::intrusive_ptr<Future> InterpreterState::getFuture() {
  return static_cast<InterpreterStateImpl*>(pImpl.get())->getOrCreateFuture();
}

InterpreterState::InterpreterState(
    c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl_)
    : pImpl(std::move(pImpl_)) {}

void InterpreterContinuation::operator()() {
#ifdef USE_RPC
  auto prev_dist_id = DistAutogradContainer::currentContextId();
  DistAutogradContainer::forceCurrentContextId(dist_autograd_context_id_);
#endif
  if (tls_state_ != c10::nullopt) {
    at::ThreadLocalStateGuard g(*tls_state_);
    state.runAsync(stack);
  } else {
    state.runAsync(stack);
  }
#ifdef USE_RPC
  DistAutogradContainer::forceCurrentContextId(prev_dist_id);
#endif
}

} // namespace jit
} // namespace torch
