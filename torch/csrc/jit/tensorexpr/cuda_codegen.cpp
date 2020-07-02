#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <torch/csrc/jit/tensorexpr/cuda_half_support.h>

#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/cuda_random.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/execution_counter.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

#define DEBUG_PRINT 0

namespace torch {
namespace jit {
namespace tensorexpr {

DEFINE_TRIGGER(cuda_codegen_created);
DEFINE_TRIGGER(cuda_codegen_executed);

// A RAII wrapper to manage a variable and name pair in the look-up table.
// TODO: move this to a more shared place.
class ScopedVarName {
 public:
  ScopedVarName(VarNameMap* mapping, const Var* var, const std::string& name)
      : mapping_(mapping), var_(var) {
    auto iter = mapping->find(var);
    if (iter != mapping->end()) {
      throw std::runtime_error("Duplicate var entry: " + var->name_hint());
    }
    mapping->insert(std::make_pair(var, name));
  }

  ScopedVarName(
      UniqueNameManager* manager,
      const Var* var,
      const std::string& name)
      : ScopedVarName(&manager->unique_name_mapping_, var, name) {}

  ScopedVarName(const ScopedVarName&) = delete;
  ScopedVarName& operator=(const ScopedVarName&) = delete;

  ~ScopedVarName() noexcept(false) {
    mapping_->erase(var_);
  }

 private:
  VarNameMap* mapping_ = nullptr;
  const Var* var_ = nullptr;
};

static int as_int(const Expr* expr) {
  auto v = dynamic_cast<const IntImm*>(expr);
  if (!v) {
    throw malformed_input(
        "cuda_codegen: non Int expr interpreted as int", expr);
  }

  return v->value();
}

static bool is_zero(const Expr* expr) {
  return as_int(expr) == 0;
}

static const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

static void getMajorMinor(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor) {
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;
  AT_CUDA_NVRTC_CHECK(
      nvrtc().nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  AT_ASSERT(nvrtc_version.first >= 6);

  CudaVersion dev_version = CudaVersion(prop->major, prop->minor);
  CudaVersion max_dev_version(dev_version);
  if (nvrtc_version.first <= 7) { // 7 supports 2-5.x
    max_dev_version = CudaVersion(5, 0);
  } else if (nvrtc_version.first <= 8) { // 8 supports 2-6.x
    max_dev_version = CudaVersion(6, 0);
  } else if (nvrtc_version.first <= 9) { // 9 supports 3-7.2
    max_dev_version = CudaVersion(7, 2);
  } else if (nvrtc_version.first <= 10) { // 10 supports 3-7.5
    max_dev_version = CudaVersion(7, 5);
  }
  if (dev_version > max_dev_version) {
    dev_version = max_dev_version;
  }
  major = dev_version.first;
  minor = dev_version.second;
}

void CudaPrinter::maybe_insert_sync() {
  if (need_sync_) {
    emitIndent();
    os() << "__syncthreads();" << std::endl;
    need_sync_ = false;
  }
}

std::string cudaDtypeCppString(const Dtype& dtype) {
  switch (dtype.scalar_type()) {
    case ScalarType::Half:
      return "half";
    case ScalarType::Char:
      return "char";
    case ScalarType::Byte:
      return "unsigned char";
    case ScalarType::Short:
      return "short";
    case ScalarType::Long:
      return "long";
    default:; /* nothing */
  }
  return dtype.ToCppString();
}

static void print_flat_alloc(std::ostream& os, const Allocate* alloc) {
  std::vector<const Expr*> dims = alloc->dims();
  // TODO: this should be merged with the storage flattener.
  int64_t flat_size = 1;
  for (auto dim : dims) {
    const IntImm* dim_i = dynamic_cast<const IntImm*>(dim);
    if (dim_i) {
      flat_size *= dim_i->value();
    } else {
      throw std::runtime_error("Only IntImm dimensions are supported for now");
    }
  }
  os << cudaDtypeCppString(alloc->dtype()) << " " << (*alloc->buffer_var())
     << "[" << flat_size << "];" << std::endl;
}

void CudaPrinter::visit(const Free* v) {
  Stmt* p = v->get_parent();
  while (p) {
    const For* for_v = dynamic_cast<const For*>(p);
    if (for_v &&
        (for_v->loop_options().is_gpu_block_index() ||
         for_v->loop_options().is_gpu_thread_index())) {
      return;
    }
    p = p->get_parent();
  }
  throw std::runtime_error("Global free not supported yet");
}

void CudaPrinter::visit(const Allocate* v) {
  Stmt* p = v->get_parent();
  while (p) {
    const For* for_v = dynamic_cast<const For*>(p);
    if (for_v) {
      if (for_v->loop_options().is_gpu_block_index()) {
        emitIndent();
        os() << "__shared__ ";
        print_flat_alloc(os(), v);
        return;
      } else if (for_v->loop_options().is_gpu_thread_index()) {
        emitIndent();
        print_flat_alloc(os(), v);
        thread_local_bufs_.insert(v->buffer_var());
        return;
      }
    }
    p = p->get_parent();
  }
  throw std::runtime_error("Global alloc not supported yet");
}

void CudaPrinter::visit(const For* v) {
  maybe_insert_sync();
  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    ScopedVarName var_name(
        name_manager(), v->var(), loop_options.gpu_block_index_str());
    emitIndent();
    v->body()->accept(this);
    os() << std::endl;
    int gpu_block_index = loop_options.gpu_block_index();
    if (gpu_block_extents_.size() <= gpu_block_index) {
      gpu_block_extents_.resize(gpu_block_index + 1);
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(v->start()));
    }
    gpu_block_extents_[gpu_block_index] = v->stop();
  } else if (loop_options.is_gpu_thread_index()) {
    ScopedVarName var_name(
        name_manager(), v->var(), loop_options.gpu_thread_index_str());
    emitIndent();
    v->body()->accept(this);
    os() << std::endl;
    int gpu_thread_index = loop_options.gpu_thread_index();
    if (gpu_thread_extents_.size() <= gpu_thread_index) {
      gpu_thread_extents_.resize(gpu_thread_index + 1);
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(v->start()));
    }
    // A conservative measure to insert thread-syncs between each thread-idx
    // change.
    // TODO: only apply this when a cross-thread dependency happens across this
    // point.
    // TODO: maybe move this to a dedicated IRNode, if the logic gets
    // sufficiently complicated.
    need_sync_ = true;
    if (gpu_thread_extents_[gpu_thread_index]) {
      if (immediateEquals(v->stop(), 1)) {
        // This is a trivial thread-idx
        return;
      }
    }
    gpu_thread_extents_[gpu_thread_index] = v->stop();
  } else {
    IRPrinter::visit(v);
  }
}

void CudaPrinter::visit(const Intrinsics* v) {
  if (v->op_type() == IntrinsicsOp::kRand) {
    os() << "Uint32ToFloat(" << *rand_func_ << "())";
    return;
  }

  std::string func_name = v->func_name();

  // get type of resulting expression.
  ScalarType returnType = v->param(0)->dtype().scalar_type();
  for (int i = 1; i < v->nparams(); ++i) {
    returnType = promoteTypes(returnType, v->param(i)->dtype().scalar_type());
  }

  if (returnType == ScalarType::Half || returnType == ScalarType::Float) {
    func_name = func_name + "f";
  }

  os() << func_name << "(";
  for (int i = 0; i < v->nparams(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void CudaPrinter::visit(const Load* v) {
  // TODO: find a better metric in using ldg or not. Support different dtypes.
  if (v->dtype().scalar_type() == ScalarType::Half) {
    os() << "__half2float(" << *v->base_handle() << "[" << *v->flat_index()
         << "])";
  } else {
    // Detects whether the load target is also a store target.
    // TODO: this is currently too wide. It detects whether a store-target
    // exists within the program. In fact, this check is only necessary within a
    // kernel.
    if (!cuda_analysis_->is_buf_store_target(v->buf())) {
      // Cuda __ldg can only be applied on read-only buffers.
      os() << "__ldg(" << *v->base_handle() << " + " << *v->flat_index() << ")";
    } else {
      os() << *v->base_handle() << "[" << *v->flat_index() << "]";
    }
  }
}

// TODO: maybe this should be a more shared location?
// TODO: investigate how "Expr*" can be implicitly converted to "ExprHandle" as
// a bool.
static bool CheckEqual(const Expr* lhs, const Expr* rhs) {
  // The fast path. Checks if the pointers are the same.
  if (lhs == rhs) {
    return true;
  }
  ExprHandle diff = Sub::make(ExprHandle(lhs), ExprHandle(rhs));
  ExprHandle diff_s = IRSimplifier::simplify(diff);
  return immediateEquals(diff_s.node(), 0);
}

// Identify the pattern: a[e1] = a[e1] + e2.
static bool isAtomicAdd(const Store* v, const Expr** atomic_add_value) {
  ScalarType dtype = v->value()->dtype().scalar_type();
  if (dtype != ScalarType::Float && dtype != ScalarType::Double) {
    return false;
  }
  const Add* add_v = dynamic_cast<const Add*>(v->value());
  if (!add_v) {
    return false;
  }
  const Load* load_v = dynamic_cast<const Load*>(add_v->lhs());
  if (!load_v) {
    return false;
  }
  if (v->base_handle() != load_v->base_handle()) {
    return false;
  }
  bool index_equal = CheckEqual(v->flat_index(), load_v->flat_index());
  if (index_equal) {
    *atomic_add_value = add_v->rhs();
  }
  return index_equal;
}

class AtomicAddFuser : public IRMutator {
  Stmt* mutate(const Store* v) override {
    const Buf* buf = v->buf();
    const std::vector<const Expr*>& indices = v->indices();
    const Expr* value = v->value();
    const Expr* atomic_add_value = nullptr;
    if (isAtomicAdd(v, &atomic_add_value)) {
      return new AtomicAdd(buf, indices, atomic_add_value);
    }
    return const_cast<Store*>(v); // NOLINT
  }
};

void CudaPrinter::visit(const Store* v) {
  emitIndent();
  os() << *v->base_handle() << "[" << *v->flat_index() << "] = ";
  if (v->value()->dtype().scalar_type() == ScalarType::Half) {
    os() << "__float2half(" << *v->value() << ");";
  } else {
    os() << *v->value() << ";";
  }
  os() << std::endl;
}

void CudaPrinter::visit(const AtomicAdd* v) {
  emitIndent();
  if (thread_local_bufs_.count(v->base_handle()) > 0) {
    // atomicAdd only works on global and shared memory
    os() << *v->base_handle() << "[" << *v->flat_index()
         << "] += " << *v->value() << ";";
  } else {
    os() << "atomicAdd(&" << *v->base_handle() << "[" << *v->flat_index() << "]"
         << ", " << *v->value() << ");";
  }
  os() << std::endl;
}

void CudaPrinter::visit(const Max* v) {
  auto dtype = v->dtype().scalar_type();
  switch (dtype) {
    case ScalarType::Half:
      // doing Half math in float.
    case ScalarType::Float:
      os() << "fmaxf";
      break;
    case ScalarType::Double:
      os() << "fmax";
      break;
    default:
      os() << "max";
      break;
  }
  os() << "(";
  v->lhs()->accept(this);
  os() << ",";
  v->rhs()->accept(this);
  os() << ")";
}

void CudaPrinter::visit(const Min* v) {
  auto dtype = v->dtype().scalar_type();
  switch (dtype) {
    case ScalarType::Half:
      // doing Half math in float.
    case ScalarType::Float:
      os() << "fminf";
      break;
    case ScalarType::Double:
      os() << "fmin";
      break;
    default:
      os() << "min";
      break;
  }
  os() << "(";
  v->lhs()->accept(this);
  os() << ",";
  v->rhs()->accept(this);
  os() << ")";
}

void CudaPrinter::visit(const LetStmt* v) {
  emitIndent();
  const Var* var = v->var();
  if (var->dtype().scalar_type() == ScalarType::Half) {
    // we do math in floats so use that.
    os() << "float";
  } else {
    os() << cudaDtypeCppString(var->dtype());
  }
  os() << " " << *var << " = " << *v->value() << "; " << std::endl;
  auto b = dynamic_cast<Block*>(v->body());
  if (b) {
    emitIndent();
  }
  v->body()->accept(this);
  if (b) {
    os() << std::endl;
  }
}

void CudaPrinter::visit(const IfThenElse* v) {
  os() << "((";
  v->condition()->accept(this);
  os() << ") ? ";
  v->true_value()->accept(this);
  os() << " : ";
  v->false_value()->accept(this);
  os() << ")";
}

class PrioritizeLoad : public IRMutator {
 public:
  const Expr* mutate(const Load* v) override {
    // Look at the declaration of this variable for more details.
    if (nested_if_then_else_ > 0) {
      return IRMutator::mutate(v);
    }
    if (thread_local_bufs_.count(v->base_handle()) > 0) {
      return IRMutator::mutate(v);
    }
    MemLoadList& load_list = load_stack_.back();
    const Var* load_new_var = new Var("v", v->dtype());
    const Expr* new_value = IRMutator::mutate(v);
    load_list.push_back(std::make_pair(load_new_var, new_value));
    return load_new_var;
  }

  // TODO: merge this with CudaPrinter into CudaAnalysis
  Stmt* mutate(const Allocate* v) override {
    Stmt* p = v->get_parent();
    while (p) {
      const For* for_v = dynamic_cast<const For*>(p);
      if (for_v) {
        if (for_v->loop_options().is_gpu_thread_index()) {
          thread_local_bufs_.insert(v->buffer_var());
          break;
        }
      }
      p = p->get_parent();
    }
    return (Stmt*)v;
  }

  // TODO: merge this with the IRMutator::mutate version.
  Stmt* mutate(const For* v) override {
    const Var* var = v->var();
    const Expr* start = v->start();
    const Expr* stop = v->stop();
    Stmt* body = v->body();
    LoopOptions loop_options = v->loop_options();
    const Var* var_new = dynamic_cast<const Var*>(var->accept_mutator(this));
    const Expr* start_new = start->accept_mutator(this);
    const Expr* stop_new = stop->accept_mutator(this);
    PushList();
    Stmt* body_new = body->accept_mutator(this);
    if (!body_new) {
      return nullptr;
    }
    Stmt* body_with_loads = AddMemLoadsFromList(body_new);
    PopList();
    if (var == var_new && start == start_new && stop == stop_new &&
        body == body_with_loads) {
      return (Stmt*)v;
    }
    return new For(var_new, start_new, stop_new, body_with_loads, loop_options);
  }

  Stmt* mutate(const LetStmt* v) override {
    const Var* var = v->var();
    const Expr* value = v->value();
    Stmt* body = v->body();
    const Var* var_new = dynamic_cast<const Var*>(var->accept_mutator(this));
    if (var_new == nullptr) {
      throw std::runtime_error("LetStmt var must be variable");
    }
    const Expr* value_new = value->accept_mutator(this);
    PushList();
    Stmt* body_new = body->accept_mutator(this);
    Stmt* body_with_loads = AddMemLoadsFromList(body_new);
    PopList();
    if (var == var_new && value == value_new && body == body_with_loads) {
      return (Stmt*)v;
    }
    return new LetStmt(var_new, value_new, body_with_loads);
  }

  Stmt* mutate(const Cond* v) override {
    const Expr* cond_old = v->condition();
    Stmt* true_old = v->true_stmt();
    Stmt* false_old = v->false_stmt();

    const Expr* cond_new = cond_old->accept_mutator(this);
    PushList();
    Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
    Stmt* true_with_loads = AddMemLoadsFromList(true_new);
    PopList();
    PushList();
    Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;
    Stmt* false_with_loads = AddMemLoadsFromList(false_new);
    PopList();

    if (cond_old == cond_new && true_old == true_with_loads &&
        false_old == false_with_loads) {
      return (Stmt*)v;
    }
    return new Cond(cond_new, true_with_loads, false_with_loads);
  }

  const Expr* mutate(const IfThenElse* v) override {
    nested_if_then_else_++;
    const Expr* new_v = IRMutator::mutate(v);
    nested_if_then_else_--;
    return new_v;
  }

  Stmt* Process(Stmt* stmt) {
    this->PushList();
    Stmt* stmt_v = stmt;
    Stmt* stmt_new = stmt_v->accept_mutator(this);
    Stmt* stmt_with_loads = AddMemLoadsFromList(stmt_new);
    this->PopList();
    return stmt_with_loads;
  }

 private:
  using MemLoadEntry = std::pair<const Var*, const Expr*>;
  using MemLoadList = std::vector<MemLoadEntry>;
  using MemoryLoadStack = std::vector<MemLoadList>;

  void PushList() {
    load_stack_.push_back(MemLoadList());
  }

  void PopList() {
    load_stack_.pop_back();
  }

  Stmt* AddMemLoadsFromList(Stmt* stmt) {
    MemLoadList& load_list = load_stack_.back();
    Stmt* stmt_v = stmt;
    for (auto iter = load_list.rbegin(); iter != load_list.rend(); iter++) {
      const MemLoadEntry& entry = *iter;
      const Var* var_ptr = entry.first;
      stmt_v = new LetStmt(var_ptr, entry.second, stmt_v);
    }
    return stmt_v;
  }

  MemoryLoadStack load_stack_;
  // TODO: For now, we are not moving the loads with the IfThenElse.
  // Eventually, we should switch to a more generic structure like:
  // int v2 = IfThenElse(cond, true_v, false_v) + 2 ->
  //
  // int v;
  // if (cond) {
  //   v = true_v;
  // } else {
  //   v = false_v;
  // }
  // int v2 = v + 2;
  int nested_if_then_else_ = 0;
  std::unordered_set<const Var*> thread_local_bufs_;
};

std::string CudaCodeGen::GetUniqueFuncName(const std::string& func_prefix) {
  // We are using a global counter here to make sure difference instances within
  // CudaCodeGen have different names.
  static int64_t counter = 0;
  ++counter;
  int64_t value = counter;
  return func_prefix + "_" + std::to_string(value);
}

// Find all the statements that are not covered by any thread-idx axes,
// and wrap them under a trivial thread idx.
class NoThreadIdxRewriter : public IRMutator {
 private:
  Stmt* rewrite(const std::vector<Stmt*>& stmts) {
    std::vector<Stmt*> cloned_stmts(stmts.size());
    for (size_t index = 0; index < stmts.size(); index++) {
      cloned_stmts[index] = Stmt::clone(stmts[index]);
    }
    Stmt* new_block = Block::make(cloned_stmts);
    // Wrap the new block under a trivial thread-idx
    //   for t in 0..1: // threadIdx
    //     if (t < 1):
    //       new_block
    // Note: the insertion of this for loop serves two purpose. First it is
    // turned into a mask; Second, it will make sure a sync point is inserted
    // when we switch to another thread-idx axis.
    VarHandle t("t", kInt);
    ExprHandle t_lt_1 = CompareSelect::make(t, 1, CompareSelectOperation::kLT);
    // TODO: move "if (t < 1)" to threadIdx generation
    Cond* masked_block = Cond::make(t_lt_1, new_block, nullptr);
    LoopOptions thread_idx_opt;
    // TODO: the added trivial threadIdx needs to match the kernel threadIdx
    // dimensions
    thread_idx_opt.set_gpu_thread_index(0);
    For* trivial_loop = For::make(t, 0, 1, masked_block, thread_idx_opt);
    return trivial_loop;
  }

  Stmt* mutate(const For* v) override {
    if (v->loop_options().is_gpu_block_index()) {
      gpu_blocks_.push_back(v);
      need_rewrite_ = false;
    } else if (v->loop_options().is_gpu_thread_index()) {
      gpu_threads_.push_back(v);
    }

    Stmt* new_for = IRMutator::mutate(v);

    if (v->loop_options().is_gpu_block_index()) {
      gpu_blocks_.pop_back();
      need_rewrite_ = false;
    } else if (v->loop_options().is_gpu_thread_index()) {
      gpu_threads_.pop_back();
    }

    return new_for;
  }

  Stmt* mutate(const Block* v) override {
    std::list<Stmt*> old_stmts(v->begin(), v->end());
    std::vector<bool> need_rewrites(old_stmts.size());
    std::vector<Stmt*> new_stmts(old_stmts.size());
    int index = 0;
    for (auto old_stmt : old_stmts) {
      need_rewrite_ = false;
      Stmt* new_stmt = old_stmt->accept_mutator(this);
      need_rewrites[index] = need_rewrite_;
      new_stmts[index] = new_stmt;
      index++;
    }

    bool any_need_fix = false;
    bool all_need_fix = need_rewrites.empty();
    for (auto need_fix : need_rewrites) {
      if (need_fix) {
        any_need_fix = true;
      } else {
        all_need_fix = false;
      }
    }

    need_rewrite_ = false;
    // If nothing needs fix, return as it is
    if (!any_need_fix) {
      return (Stmt*)v;
    }

    // If all needs fix, then we could have its parent statement to merge
    // further. Unless the parent is a block-indx axis, then we should handle
    // the rewrite now.
    if (all_need_fix) {
      Stmt* parent = v->get_parent();
      For* loop_parent = dynamic_cast<For*>(parent);
      if (loop_parent && loop_parent->loop_options().is_gpu_block_index()) {
        Stmt* new_block = rewrite(new_stmts);
        return new_block;
      }
      need_rewrite_ = true;
      return (Stmt*)v;
    }

    // if some needs fix, rewrites the consecutive parts
    int start = 0;
    int count = new_stmts.size();
    std::vector<Stmt*> rewrite_stmts;
    while (start < count) {
      while (start < count && !need_rewrites[start]) {
        rewrite_stmts.push_back(Stmt::clone(new_stmts[start]));
        start++;
      }
      if (start >= count) {
        break;
      }
      int stop = start + 1;
      while (stop < count && need_rewrites[stop]) {
        stop++;
      }

      // Rewrite the stmts from [start, stop)
      std::vector<Stmt*> stmts_to_rewrite(
          new_stmts.begin() + start, new_stmts.begin() + stop);
      Stmt* rewritten_stmt = rewrite(stmts_to_rewrite);
      rewrite_stmts.push_back(rewritten_stmt);

      start = stop;
    }
    Stmt* rewritten_block = Block::make(rewrite_stmts);
    return rewritten_block;
  }

  Stmt* mutate(const Store* v) override {
    need_rewrite_ = gpu_threads_.empty();
    return (Stmt*)v;
  }

  std::vector<const For*> gpu_blocks_;
  std::vector<const For*> gpu_threads_;
  bool need_rewrite_ = false;
};

void CudaCodeGen::Initialize() {
  // TODO: handle multiple kernels.
  // TODO: handle dynamic dimension.
  // TODO: call nvrtc.
  // TODO: merge HasRand with CudaAnalysis.
  HasRand has_rand_func(stmt());
  has_random_ = has_rand_func.has_rand();
  cuda_analysis_ = std::make_unique<CudaAnalysis>();
  printer_ =
      std::make_unique<CudaPrinter>(&oss_, cuda_analysis_.get(), has_random_);

  os() << "#define NAN __int_as_float(0x7fffffff)\n"
          "#define POS_INFINITY __int_as_float(0x7f800000)\n"
          "#define NEG_INFINITY __int_as_float(0xff800000)\n";
  if (has_random_) {
    os() << philox_random_string << std::endl;
  }

  // Check whether the statement uses the Half type, if so add the
  // half_support_literal.
  CudaHalfChecker halfChecker;
  stmt()->accept(&halfChecker);
  if (halfChecker.hasHalf()) {
    os() << fuser::cuda::half_support_literal << std::endl;
  }

  std::string func_name = GetUniqueFuncName("func");
  os() << "extern \"C\" __global__" << std::endl << "void " << func_name << "(";
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (size_t i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    const BufferArg& buffer_arg = buffer_args[i];
    const Var* var = buffer_arg.var();
    Dtype dtype = buffer_arg.dtype();

    os() << cudaDtypeCppString(dtype) << (buffer_arg.isVar() ? " " : "* ")
         << name_manager()->get_unique_name(var);
  }
  const Var* rand_seed;
  const Var* rand_offset;
  if (has_random_) {
    // TODO: switch to kUint64 when it is available.
    rand_seed = new Var("rand_seed", kInt);
    rand_offset = new Var("rand_offset", kInt);
    std::string uint64_str = "unsigned long long";
    os() << ", " << uint64_str << " " << *rand_seed << ", " << uint64_str << " "
         << *rand_offset;
  }
  os() << ") {";
  os() << std::endl;

  if (has_random_) {
    const Var* idx = new Var("idx", kInt);
    os() << "int " << *idx << " = blockIdx.x*blockDim.x + threadIdx.x;"
         << std::endl;
    const Var* rand_func = printer_->rand_func();
    os() << "Philox " << *rand_func << "(" << *rand_seed << ", " << *idx << ", "
         << *rand_offset << ");" << std::endl;
    os() << std::endl;
  }

  Stmt* stmt_v = stmt();
  NoThreadIdxRewriter no_thread_idx;
  stmt_v = stmt_v->accept_mutator(&no_thread_idx);
  AtomicAddFuser atomic_add_fuser;
  stmt_v = stmt_v->accept_mutator(&atomic_add_fuser);
  PrioritizeLoad prioritize_load;
  stmt_v = prioritize_load.Process(stmt_v);
  stmt_v->accept(cuda_analysis_.get());
  stmt_v->accept(printer_.get());
  os() << std::endl;
  os() << "}";

  // Check that all block extents had been set.
  const std::vector<const Expr*>& gpu_block_extents =
      printer_->gpu_block_extents();
  const std::vector<const Expr*>& gpu_thread_extents =
      printer_->gpu_thread_extents();
  for (size_t i = 0; i < gpu_block_extents.size(); i++) {
    if (!gpu_block_extents[i]) {
      throw std::runtime_error("Missing gpu_block_index: " + std::to_string(i));
    }
  }

#if DEBUG_PRINT
  std::cout << "stmt: " << std::endl;
  std::cout << oss_.str() << std::endl;
  std::cout << "block(";
  for (size_t i = 0; i < gpu_block_extents.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << *gpu_block_extents[i];
  }
  std::cout << "), thread(";
  for (size_t i = 0; i < gpu_thread_extents.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << *gpu_thread_extents[i];
  }
  std::cout << ")" << std::endl;
  ;
#endif

  CompileToNVRTC(oss_.str(), func_name);
  USE_TRIGGER(cuda_codegen_created);
}

void CudaCodeGen::call(const std::vector<CallArg>& args) {
  if (args.size() != buffer_args().size()) {
    throw malformed_input("cuda_codegen: wrong number of args in call");
  }

  // TODO: move as much of this into the constructors.
  const std::vector<const Expr*>& gpu_block_extents =
      printer_->gpu_block_extents();
  const std::vector<const Expr*>& gpu_thread_extents =
      printer_->gpu_thread_extents();
  if (gpu_block_extents.size() > 3 || gpu_thread_extents.size() > 3) {
    throw malformed_input(
        "cuda_codegen: block or thread extent greater than 3D");
  }

  std::vector<int> gpu_block_extents_v(3, 1);
  std::vector<int> gpu_thread_extents_v(3, 1);
  // evaluate all the block/thread extents into values
  // TODO: eventually, codegen these calculations and make them part of the
  // module.
  for (size_t i = 0; i < gpu_block_extents.size(); i++) {
    ExprEval<SimpleIREvaluator> eval(
        ExprHandle(gpu_block_extents[i]), buffer_args());
    gpu_block_extents_v[i] = eval.value<int>(args);
  }
  for (size_t i = 0; i < gpu_thread_extents.size(); i++) {
    ExprEval<SimpleIREvaluator> eval(
        ExprHandle(gpu_thread_extents[i]), buffer_args());
    gpu_thread_extents_v[i] = eval.value<int>(args);
  }

  // Skip launching the kernel if there are no elements to process.
  for (int extent : gpu_block_extents_v) {
    if (extent == 0) {
      return;
    }
  }

  // Bind the buffer addresses into arguments
  auto const& buffer_args = this->buffer_args();
  int ptr_count = buffer_args.size();
  if (has_random_) {
    ptr_count += 2;
  }
  std::vector<void*> args_data(buffer_args.size());
  std::vector<void*> ptr_to_args(ptr_count);
  uint64_t rand_seed = uint64_t(-1);
  uint64_t rand_offset = uint64_t(-1);
  for (size_t i = 0; i < buffer_args.size(); i++) {
    auto const& bufferArg = buffer_args[i];
    if (bufferArg.isVar()) {
      auto stype = bufferArg.dtype().scalar_type();
      switch (stype) {
#define TYPE_CASE(Type, Name)             \
  case ScalarType::Name:                  \
    ptr_to_args[i] = args[i].Name##Ptr(); \
    break;
        AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
        default:
          throw unsupported_dtype();
      }
    } else {
      args_data[i] = args[i].data();
      ptr_to_args[i] = &args_data[i];
    }
  }

  if (has_random_) {
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    // TODO: total hack. Switch to numel when it is available.
    int64_t total_elements_per_thread = (1LL << 28);
    {
      std::lock_guard<std::mutex> lock(gen.mutex());
      auto philox_engine_inputs =
          at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
              total_elements_per_thread);
      rand_seed = philox_engine_inputs.first;
      rand_offset = philox_engine_inputs.second;
    }
    ptr_to_args[buffer_args.size()] = &rand_seed;
    ptr_to_args[buffer_args.size() + 1] = &rand_offset;
  }

  // Launch the kernels
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      function_,
      gpu_block_extents_v[0],
      gpu_block_extents_v[1],
      gpu_block_extents_v[2],
      gpu_thread_extents_v[0],
      gpu_thread_extents_v[1],
      gpu_thread_extents_v[2],
      0,
      stream,
      ptr_to_args.data(),
      nullptr));
  USE_TRIGGER(cuda_codegen_executed);
}

void CudaSetContext(CUcontext pctx) {
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(0);
  }
}

void CudaCodeGen::CompileToNVRTC(
    const std::string& code,
    const std::string& func_name) {
  CUcontext pctx = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  // Note: hacked at::DeviceGuard since at::DeviceGuard was failing to work
  // properly in some scenarios
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(this->device().index());
  // cudaSetDevice does not have to really change the underlying device if it
  // doesn't have to, so calling cudaFree to force that change
  CudaSetContext(pctx);

  // Acquires device and NVRTC properties (for compile arch and occupancy
  // calculations)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop, major, minor);

#if DEBUG_PRINT
  std::cout << "major: " << major << ", "
            << "minor: " << minor << std::endl;
#endif

  // Creates the NVRTC program
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

#ifdef __HIP_PLATFORM_HCC__
  std::vector<const char*> args = {};
#else
  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};
#endif

  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLogSize(program, &logsize));
    std::vector<char> log(logsize);
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLog(program, log.data()));
    std::stringstream cu;
    cu << log.data() << std::endl;
    cu << "nvrtc compilation failed: " << std::endl;
    cu << code << std::endl;
    throw std::runtime_error(cu.str());
  }
  ResourceGuard holdProgram(
      [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx.data()));

  CUmodule module;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&module, ptx.data()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&function_, module, func_name.c_str()));
  at::cuda::set_device(prior_device);
}

CudaCodeGen::~CudaCodeGen() = default;

RegisterCodeGen<CudaCodeGen> cuda_codegen_reg("cuda_codegen");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
