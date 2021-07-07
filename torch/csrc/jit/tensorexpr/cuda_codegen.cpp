#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#include <torch/csrc/jit/tensorexpr/half_support.h>

#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/fuser/cuda/resource_strings.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/cuda_random.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/registerizer.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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

// query codegen output arch and target
static void codegenOutputQuery(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor,
    bool& compile_to_sass) {
  using CudaVersion = std::pair<int, int>;
  CudaVersion nvrtc_version;
  AT_CUDA_NVRTC_CHECK(
      nvrtc().nvrtcVersion(&nvrtc_version.first, &nvrtc_version.second));

  AT_ASSERT(nvrtc_version.first >= 6);

  CudaVersion dev_version = CudaVersion(prop->major, prop->minor);
  CudaVersion max_dev_version(dev_version);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (nvrtc_version.first <= 7) { // 7 supports 2-5.x
    max_dev_version = CudaVersion(5, 0);
  } else if (nvrtc_version.first <= 8) { // 8 supports 2-6.x
    max_dev_version = CudaVersion(6, 0);
  } else if (nvrtc_version.first <= 9) { // 9 supports 3-7.2
    max_dev_version = CudaVersion(7, 2);
  } else if (nvrtc_version.first <= 10) { // 10 supports 3-7.5
    max_dev_version = CudaVersion(7, 5);
  } else if (nvrtc_version.first == 11 && nvrtc_version.second == 0) {
    // 11.0 supports 3-8.0
    max_dev_version = CudaVersion(8, 0);
  }
  if (dev_version > max_dev_version) {
    dev_version = max_dev_version;
  }
  major = dev_version.first;
  minor = dev_version.second;

  // if we are clamping major/minor, sass is not compatible
  compile_to_sass = (major == prop->major) && (minor == prop->minor);
}

std::string CudaPrinter::dtypeToCppString(const Dtype& dtype) {
  switch (dtype.scalar_type()) {
    case ScalarType::Bool:
      return "bool";
    case ScalarType::Half:
      return "half";
    case ScalarType::Char:
      return "char";
    case ScalarType::Byte:
      return "unsigned char";
    case ScalarType::Short:
      return "short";
    case ScalarType::Long:
      return "long long";
    default:
      return dtype.ToCppString();
  }
}

void CudaAnalysis::visit(const Free* v) {
  if (thread_local_bufs_.count(v->buffer_var()) == 0 &&
      cross_block_bufs_.count(v->buffer_var()) == 0) {
    throw std::runtime_error("Global free not supported yet");
  }
}

void CudaAnalysis::visit(const Allocate* v) {
  Stmt* p = v->get_parent();
  while (p) {
    const For* for_v = dynamic_cast<const For*>(p);
    if (for_v) {
      // NOLINTNEXTLINE(bugprone-branch-clone)
      if (for_v->loop_options().is_gpu_block_index()) {
        // TODO: This isn't right if there's a thread index at a higher level
        // than this.
        cross_block_bufs_.insert(v->buffer_var());
        return;
      } else if (for_v->loop_options().is_gpu_thread_index()) {
        thread_local_bufs_.insert(v->buffer_var());
        return;
      }
    }
    p = p->get_parent();
  }
  throw std::runtime_error("Global alloc not supported yet");
}

void CudaAnalysis::visit(const For* v) {
  // Recurse first.
  v->body()->accept(this);

  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    int gpu_block_index = loop_options.gpu_block_index();
    if (gpu_block_index >= 3) {
      throw std::runtime_error("support only 3D gpu_block_index");
    }
    const Expr* prev = nullptr;
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (gpu_block_extents_.size() <= gpu_block_index) {
      gpu_block_extents_.resize(gpu_block_index + 1);
    } else {
      prev = gpu_block_extents_[gpu_block_index];
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(v->start()));
    }

    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (prev == nullptr) {
      gpu_block_extents_[gpu_block_index] = v->stop();
    } else if (prev->isConstant() && immediateEquals(prev, 1)) {
      // extents must be positive so if the current extent is 1 then even if the
      // stop is symbolic it's the max.
      gpu_block_extents_[gpu_block_index] = v->stop();
    } else {
      gpu_block_extents_[gpu_block_index] =
          IRSimplifier::simplify(new Max(prev, v->stop(), true));
    }
  } else if (loop_options.is_gpu_thread_index()) {
    int gpu_thread_index = loop_options.gpu_thread_index();
    if (gpu_thread_index >= 3) {
      throw std::runtime_error("support only 3D gpu_thread_index");
    }
    const Expr* prev = nullptr;
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (gpu_thread_extents_.size() <= gpu_thread_index) {
      gpu_thread_extents_.resize(gpu_thread_index + 1);
    } else {
      prev = gpu_thread_extents_[gpu_thread_index];
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_thread_index: " +
          std::to_string(v->start()));
    }

    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (prev == nullptr) {
      gpu_thread_extents_[gpu_thread_index] = v->stop();
    } else if (prev->isConstant() && immediateEquals(prev, 1)) {
      // extents must be positive so if the current extent is 1 then even if the
      // stop is symbolic it's the max.
      gpu_thread_extents_[gpu_thread_index] = v->stop();
    } else {
      gpu_thread_extents_[gpu_thread_index] =
          IRSimplifier::simplify(new Max(prev, v->stop(), true));
    }
  }
}

void CudaPrinter::print_flat_alloc(const Allocate* alloc) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
  os() << dtypeToCppString(alloc->dtype()) << " " << (*alloc->buffer_var())
       << "[" << flat_size << "];" << std::endl;
}

void CudaPrinter::visit(const Allocate* v) {
  // TODO: handle dynamic shapes here.
  if (cuda_analysis_->cross_block_bufs().count(v->buffer_var()) != 0) {
    emitIndent();
    os() << "__shared__ ";
    print_flat_alloc(v);
    return;
  }

  if (cuda_analysis_->thread_local_bufs().count(v->buffer_var()) != 0) {
    emitIndent();
    print_flat_alloc(v);
    return;
  }

  throw std::runtime_error("Encountered Alloc not local to block or thread");
}

void CudaPrinter::visit(const Free* v) {
  // do nothing
}

void CudaPrinter::visit(const For* v) {
  IRPrinter::visit(v);
}

void CudaPrinter::visit(const Cast* v) {
  if (v->dtype().scalar_type() == ScalarType::Half) {
    os() << "__float2half(";
    v->src_value()->accept(this);
    os() << ")";
    return;
  } else if (v->src_value()->dtype().scalar_type() == ScalarType::Half) {
    os() << "__half2float(";
    v->src_value()->accept(this);
    os() << ")";
    return;
  }

  os() << "(" << dtypeToCppString(v->dtype()) << ")";
  os() << "(";
  v->src_value()->accept(this);
  os() << ")";
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
  if (v->op_type() == IntrinsicsOp::kAbs &&
      !c10::isIntegralType(returnType, true)) {
    // since kAbs's func_name is `abs`, prefix `f` for floating point
    func_name = "f" + func_name;
  }
  if (v->op_type() == IntrinsicsOp::kIsNan) {
    func_name = "isnan";
  }

  os() << func_name << "(";
  for (const auto i : c10::irange(v->nparams())) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void CudaPrinter::visit(const ExternalCall* v) {
  throw unimplemented_lowering(v);
}

void CudaPrinter::visit(const Load* v) {
  // TODO: find a better metric in using ldg or not. Support different dtypes.
  // Detects whether the load target is also a store target.
  // TODO: this is currently too wide. It detects whether a store-target
  // exists within the program. In fact, this check is only necessary within a
  // kernel.
  if (v->indices().empty()) {
    os() << *v->base_handle();
    return;
  }
  if (v->dtype().scalar_type() == ScalarType::Bool ||
      v->dtype().scalar_type() == ScalarType::Half) {
    // There's no __ldg overload for bool or half.
    os() << *v->base_handle() << "[" << *v->flat_index() << "]";
    return;
  }
  if (cuda_analysis_->is_buf_store_target(v->buf())) {
    // Cuda __ldg can only be applied on read-only buffers.
    os() << *v->base_handle() << "[" << *v->flat_index() << "]";
    return;
  }
  os() << "__ldg(" << *v->base_handle() << " + " << *v->flat_index() << ")";
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

class AtomicAddFuser : public IRMutator {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AtomicAddFuser(
      const std::unordered_set<const Var*>& thread_local_bufs,
      const GPUMetaVarRewriter& metavars)
      : thread_local_bufs_(thread_local_bufs) {
    const std::vector<const Expr*>& block_extents =
        metavars.gpu_block_extents();
    const std::vector<const Var*>& block_vars = metavars.gpu_block_vars();
    for (size_t i = 0; i < block_extents.size(); ++i) {
      MetaVarExtent extent{block_extents[i], false};
      if (extent.expr->isConstant() && immediateEquals(extent.expr, 1)) {
        extent.trivial = true;
      } else {
        nontrivial_metavars_.insert(block_vars[i]);
      }
      metavars_[block_vars[i]] = extent;
    }

    const std::vector<const Expr*>& thread_extents =
        metavars.gpu_thread_extents();
    const std::vector<const Var*>& thread_vars = metavars.gpu_thread_vars();
    for (size_t i = 0; i < thread_extents.size(); ++i) {
      MetaVarExtent extent{thread_extents[i], false};
      if (extent.expr->isConstant() && immediateEquals(extent.expr, 1)) {
        extent.trivial = true;
      } else {
        nontrivial_metavars_.insert(thread_vars[i]);
      }
      metavars_[thread_vars[i]] = extent;
    }
  }

  Stmt* mutate(const Store* v) override {
    const Buf* buf = v->buf();
    Store* orig = const_cast<Store*>(v); // NOLINT

    // Thread locals never need to be atomic.
    if (thread_local_bufs_.count(buf->base_handle()) != 0) {
      return orig;
    }

    ScalarType dtype = v->value()->dtype().scalar_type();
    if (dtype != ScalarType::Float && dtype != ScalarType::Double) {
      return orig;
    }
    const Add* add_v = dynamic_cast<const Add*>(v->value());
    if (!add_v) {
      return orig;
    }
    const Load* load_v = dynamic_cast<const Load*>(add_v->lhs());
    if (!load_v) {
      return orig;
    }
    if (v->base_handle() != load_v->base_handle()) {
      return orig;
    }
    if (v->indices().empty() && load_v->indices().empty()) {
      return orig;
    }
    bool index_equal = CheckEqual(v->flat_index(), load_v->flat_index());
    if (!index_equal) {
      return orig;
    }

    // TODO: this checks that the metavars occur directly as an index, but this
    // is pessimistic, blockIdx.x + 1 is fine too if there is no overlapping.
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::unordered_set<const Var*> vars_to_find = nontrivial_metavars_;
    for (const Expr* e : v->indices()) {
      if (const Var* v = dynamic_cast<const Var*>(e)) {
        vars_to_find.erase(v);
      }
    }

    if (vars_to_find.empty()) {
      // All metavars accounted for.
      return orig;
    }

    return new AtomicAdd(buf, v->indices(), add_v->rhs());
  }

 private:
  const std::unordered_set<const Var*>& thread_local_bufs_;
  struct MetaVarExtent {
    const Expr* expr{nullptr};
    bool trivial{false};
  };
  std::unordered_map<const Var*, MetaVarExtent> metavars_;
  std::unordered_set<const Var*> nontrivial_metavars_;
};

void CudaPrinter::visit(const Store* v) {
  emitIndent();
  if (v->indices().empty()) {
    os() << *v->base_handle() << " = ";
  } else {
    os() << *v->base_handle() << "[" << *v->flat_index() << "] = ";
  }
  os() << *v->value() << ";";
  os() << std::endl;
}

void CudaPrinter::visit(const AtomicAdd* v) {
  emitIndent();
  if (cuda_analysis_->thread_local_bufs().count(v->base_handle()) > 0) {
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
  if (v->dtype().is_integral()) {
    os() << "max(";
  } else {
    os() << "maximum(";
  }
  v->lhs()->accept(this);
  os() << ",";
  v->rhs()->accept(this);
  os() << ")";
}

void CudaPrinter::visit(const Min* v) {
  if (v->dtype().is_integral()) {
    os() << "min(";
  } else {
    os() << "minimum(";
  }
  v->lhs()->accept(this);
  os() << ",";
  v->rhs()->accept(this);
  os() << ")";
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

void CudaPrinter::visit(const Block* v) {
  os() << "{" << std::endl;
  indent_++;

  for (Stmt* s : v->stmts()) {
    s->accept(this);
  }

  indent_--;
  emitIndent();
  os() << "}";
}

void CudaPrinter::visit(const Let* v) {
  emitIndent();
  os() << dtypeToCppString(v->dtype());
  os() << " " << *v->var() << " = ";
  v->value()->accept(this);
  os() << ";" << std::endl;
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class PrioritizeLoad : public IRMutator {
 public:
  const Expr* mutate(const Load* v) override {
    // Look at the declaration of this variable for more details.
    if (nested_if_then_else_ > 0) {
      return IRMutator::mutate(v);
    }
    if (nested_let_) {
      return IRMutator::mutate(v);
    }
    if (thread_local_bufs_.count(v->base_handle()) > 0) {
      return IRMutator::mutate(v);
    }
    if (v->indices().size() == 0) {
      return IRMutator::mutate(v);
    }
    if (nested_store_) {
      if (v->base_handle() == nested_store_->buf()->base_handle() &&
          v->indices().size() == nested_store_->indices().size()) {
        // also check indices
        bool same = true;
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        for (int i = 0; i < v->indices().size(); ++i) {
          if (!exprEquals(v->indices()[i], nested_store_->indices()[i])) {
            same = false;
            break;
          }
        }
        if (same) {
          return IRMutator::mutate(v);
        }
      } else if (nested_store_->indices().empty()) {
        return IRMutator::mutate(v);
      }
    }

    MemLoadList& load_list = load_stack_.back();
    const Var* load_new_var = new Var("v", v->dtype());
    const Expr* new_value = IRMutator::mutate(v);
    load_list.push_back(std::make_pair(load_new_var, new_value));

    return load_new_var;
  }

  const Expr* mutate(const Cast* v) override {
    const Load* src_load = dynamic_cast<const Load*>(v->src_value());
    const Expr* new_src = v->src_value()->accept_mutator(this);
    const Var* new_var = dynamic_cast<const Var*>(new_src);
    if (!src_load || !new_var) {
      return new Cast(v->dtype(), new_src);
    }

    // We just did the prioritize load, let's fold in the Cast.
    MemLoadList& load_list = load_stack_.back();
    assert(!load_list.empty());
    auto pair = load_list.back();
    assert(pair.first == new_var);
    load_list.pop_back();

    new_var = new Var("v", v->dtype());
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const Expr* new_value = new Cast(v->dtype(), pair.second);
    load_list.push_back(std::make_pair(new_var, new_value));
    return new_var;
  }

  Stmt* mutate(const Store* v) override {
    const Store* last = nested_store_;
    nested_store_ = v;
    Stmt* s = IRMutator::mutate(v);
    nested_store_ = last;
    return s;
  }

  Stmt* mutate(const Let* v) override {
    nested_let_ = true;
    Stmt* s = IRMutator::mutate(v);
    nested_let_ = false;
    return s;
  }

  Stmt* mutate(const Block* v) override {
    Block* v1 = const_cast<Block*>(v); // NOLINT
    assert(v1);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::list<Stmt*> stmts = v1->stmts();
    for (Stmt* stmt : stmts) {
      PushList();
      Stmt* stmt_new = stmt->accept_mutator(this);

      AddMemLoadsFromList(v1, stmt);
      PopList();

      if (stmt_new == stmt) {
        continue;
      }
      v1->replace_stmt(stmt, stmt_new);
    }
    return v1;
  }

  const Expr* mutate(const IfThenElse* v) override {
    nested_if_then_else_++;
    const Expr* new_v = IRMutator::mutate(v);
    nested_if_then_else_--;
    return new_v;
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

  void AddMemLoadsFromList(Block* block, Stmt* last) {
    MemLoadList& load_list = load_stack_.back();
    if (load_list.empty()) {
      return;
    }

    for (const auto& pair : load_list) {
      Stmt* news = new Let(pair.first, pair.second);
      block->insert_stmt_before(news, last);
    }
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
  int nested_if_then_else_{0};
  const Store* nested_store_{nullptr};
  bool nested_let_{false};
  std::unordered_set<const Var*> thread_local_bufs_;
};

std::string CudaCodeGen::GetUniqueFuncName(const std::string& func_prefix) {
  int64_t counter = 0;
  std::string name = func_prefix;
  while (taken_func_names.count(name)) {
    name = func_prefix + "_" + std::to_string(counter++);
  }

  taken_func_names.insert(name);
  return name;
}

bool GPUMetaVarRewriter::isFullExtent() {
  {
    auto& extents = cuda_analysis_->gpu_block_extents();
    for (int i = 0; i < 3; ++i) {
      if (!exprEquals(current_block_reach_[i], extents[i])) {
        return false;
      }
    }
  }

  {
    auto& extents = cuda_analysis_->gpu_thread_extents();
    for (int i = 0; i < 3; ++i) {
      if (!exprEquals(current_thread_reach_[i], extents[i])) {
        return false;
      }
    }
  }

  return true;
}

Stmt* GPUMetaVarRewriter::mutate(const For* v) {
  Stmt* body = v->body();
  const Expr* old_reach = nullptr;
  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    int gpu_block_index = loop_options.gpu_block_index();
    if (gpu_block_index >= 3) {
      throw std::runtime_error("support only 3D gpu_block_index");
    }
    old_reach = current_block_reach_[gpu_block_index];

    // Extents must be positive, assume >= 1.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (old_reach->isConstant() && immediateEquals(old_reach, 1)) {
      current_block_reach_[gpu_block_index] = v->stop();
    } else {
      current_block_reach_[gpu_block_index] =
          IRSimplifier::simplify(new Max(old_reach, v->stop(), true));
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const Var* metaVar = gpu_block_vars_[gpu_block_index];
    body = Substitute(Stmt::clone(body), {{v->var(), metaVar}});
  } else if (loop_options.is_gpu_thread_index()) {
    int gpu_thread_index = loop_options.gpu_thread_index();
    if (gpu_thread_index >= 3) {
      throw std::runtime_error("support only 3D gpu_thread_index");
    }
    old_reach = current_thread_reach_[gpu_thread_index];

    // Extents must be positive, assume >= 1.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (old_reach->isConstant() && immediateEquals(old_reach, 1)) {
      current_thread_reach_[gpu_thread_index] = v->stop();
    } else {
      current_thread_reach_[gpu_thread_index] =
          IRSimplifier::simplify(new Max(old_reach, v->stop(), true));
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const Var* metaVar = gpu_thread_vars_[gpu_thread_index];
    body = Substitute(Stmt::clone(body), {{v->var(), metaVar}});
  }

  // Recurse into body block.
  body = Stmt::clone(body->accept_mutator(this));

  // pop the internal reach off the stack.
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (loop_options.is_gpu_block_index()) {
    current_block_reach_[loop_options.gpu_block_index()] = old_reach;
    return body;
  } else if (loop_options.is_gpu_thread_index()) {
    current_thread_reach_[loop_options.gpu_thread_index()] = old_reach;
    return body;
  }

  return v->cloneWithNewBody(body);
}

Stmt* GPUMetaVarRewriter::mutate(const Block* v) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<Segment> innerSegments;
  Segment current;

  auto pushAndReset = [&](bool mask) {
    if (!current.empty()) {
      innerSegments.push_back(current);
    }
    current.reset(mask);
  };

  // Here's we're slicing the Block's contents into segments that should have
  // the same launch reach. Segments are comprised of all statements that aren't
  // loops - which are their own segments. Some operations, such as threading
  // and memory ops should never be masked and so also get their own segment.
  for (Stmt* stmt : *v) {
    Stmt* stmt_new = stmt->accept_mutator(this);
    if (stmt == stmt_new) {
      stmt_new = Stmt::clone(stmt_new);
    }

    // Likewise, Allocate and Free should never be masked.
    if (dynamic_cast<Allocate*>(stmt) || dynamic_cast<Free*>(stmt)) {
      pushAndReset(false);
    }

    // If the current stmt *was* a loop, it's a segment boundary.
    if (For* f = dynamic_cast<For*>(stmt)) {
      pushAndReset(false);
    }

    current.stmts().push_back(stmt_new);
    // if the current segment should not be masked, it's a segment boundary on
    // the far side as well.
    if (!current.mask()) {
      pushAndReset(true);
    }
  }

  if (!current.empty()) {
    innerSegments.push_back(current);
  }

  // We are max extent in all dimensions, so need no masks at this level.
  if (isFullExtent()) {
    // flatten inner segments.
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Stmt*> stmts;
    for (auto& v : innerSegments) {
      for (auto* s : v.stmts()) {
        stmts.push_back(s);
      }
    }

    return new Block(stmts);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<Stmt*> stmts;
  for (auto& segment : innerSegments) {
    bool need_sync = false;
    // We never mask loops, they'll mask their contents.
    if (!segment.mask()) {
      TORCH_INTERNAL_ASSERT(segment.stmts().size() == 1);
      stmts.push_back(segment.stmts()[0]);
      continue;
    }

    // If we get here, we must mask since we're not full reach and our direct
    // child isn't a For.
    Stmt* inner = new Block(segment.stmts());
    // threads inside blocks.
    auto& thread_extents = cuda_analysis_->gpu_thread_extents();
    for (size_t i = 0; i < gpu_thread_vars_.size(); ++i) {
      if (!exprEquals(current_thread_reach_[i], thread_extents[i])) {
        need_sync = true;
        // Mask it against the current dimensions.
        inner = new Cond(
            new CompareSelect(
                gpu_thread_vars_[i],
                current_thread_reach_[i],
                CompareSelectOperation::kLT),
            inner,
            nullptr);
      }
    }
    auto& block_extents = cuda_analysis_->gpu_block_extents();
    for (size_t i = 0; i < gpu_block_vars_.size(); ++i) {
      if (!exprEquals(current_block_reach_[i], block_extents[i])) {
        // Mask it against the current dimensions.
        inner = new Cond(
            new CompareSelect(
                gpu_block_vars_[i],
                current_block_reach_[i],
                CompareSelectOperation::kLT),
            inner,
            nullptr);
      }
    }

    if (need_sync) {
      stmts.push_back(new SyncThreads());
    }
    stmts.push_back(inner);
    if (need_sync) {
      stmts.push_back(new SyncThreads());
    }
  }

  return new Block(stmts);
}

static std::ostream& operator<<(
    std::ostream& out,
    const std::vector<const Expr*>& exprs) {
  size_t i = 0;
  for (auto expr : exprs) {
    if (i++ > 0) {
      out << ", ";
    }
    out << *expr;
  }
  return out;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static const char* device_resource_string = R"(
#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

)";

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static const char* shared_resource_string = R"(
template<typename T>
__device__ T maximum(T a, T b) {
  return isnan(a) ? a : (a > b ? a : b);
}

template<typename T>
__device__ T minimum(T a, T b) {
  return isnan(a) ? a : (a < b ? a : b);
}

)";

void CudaCodeGen::Initialize() {
  // TODO: handle multiple kernels.
  // TODO: handle dynamic dimension.
  // TODO: call nvrtc.
  // TODO: merge HasRand with CudaAnalysis.
  GenericIntrinsicsExpander intrinsics_expander;
  apply_mutator(&intrinsics_expander);

  HasRand has_rand_func(stmt());
  has_random_ = has_rand_func.has_rand();
  cuda_analysis_ = std::make_unique<CudaAnalysis>();
  printer_ =
      std::make_unique<CudaPrinter>(&oss_, cuda_analysis_.get(), has_random_);
  metavar_rewriter_ =
      std::make_unique<GPUMetaVarRewriter>(cuda_analysis_.get());

  // Check whether the statement uses the Half type, if so add the
  // half_support_literal.
  Stmt* stmt_v = stmt();
  HalfChecker halfChecker(buffer_args());
  stmt_v->accept(&halfChecker);

#if __HIP_PLATFORM_HCC__
#if ROCM_VERSION < 40200
  os() << "#include <hip/hip_runtime.h>" << std::endl;
  if (halfChecker.hasHalf()) {
    os() << "#include <hip/hip_fp16.h>" << std::endl;
  }
#endif
#endif
  os() << device_resource_string << shared_resource_string;

  if (has_random_) {
    os() << philox_random_string << std::endl;
  }

  if (halfChecker.hasHalf()) {
    os() << fuser::cuda::half_support_literal << std::endl;
  }

  std::string func_name = GetUniqueFuncName(kernel_func_name());
  os() << "extern \"C\" __global__" << std::endl;
#ifdef USE_ROCM
  // CUDA has a default limit of threads per block (=flat work group size)
  // of 1024, but ROCm uses 256 by default. At the time of writing
  // (#45506), I am unaware of a stricter limit that TensorExpr imposes
  // (maybe for perf),so I use 1024 as maximum flat work group size.
  // We put a minimum value of 1, this is also used by hip (ROCm 3.8) in
  // the __launch_bound__ implementation. The arguments for the attribute
  // are (min, max), for details see the documentation at
  // https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
  os() << "__attribute__((amdgpu_flat_work_group_size(1, 1024)))" << std::endl;
#endif
  os() << "void " << func_name << "(";
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (size_t i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    const BufferArg& buffer_arg = buffer_args[i];
    const Var* var = buffer_arg.var();
    Dtype dtype = buffer_arg.dtype();

    os() << printer_->dtypeToCppString(dtype)
         << (buffer_arg.isVar() ? " " : "* ")
         << name_manager()->get_unique_name(var);
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const Var* rand_seed;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const Var* rand_func = printer_->rand_func();
    os() << "Philox " << *rand_func << "(" << *rand_seed << ", " << *idx << ", "
         << *rand_offset << ");" << std::endl;
    os() << std::endl;
  }

  stmt_v->accept(cuda_analysis_.get());

  stmt_v = stmt_v->accept_mutator(metavar_rewriter_.get());

  AtomicAddFuser atomic_add_fuser(
      cuda_analysis_->thread_local_bufs(), *metavar_rewriter_.get());
  stmt_v = stmt_v->accept_mutator(&atomic_add_fuser);

  stmt_v = registerize(stmt_v);

  PrioritizeLoad prioritize_load;
  stmt_v = stmt_v->accept_mutator(&prioritize_load);

  // The registerizer might insert half-type scalars, we don't want this.
  HalfRewriter hsFix;
  stmt_v = stmt_v->accept_mutator(&hsFix);

  stmt_v = IRSimplifier::simplify(stmt_v);
  set_stmt(stmt_v);

  stmt_v->accept(printer_.get());
  os() << std::endl;
  os() << "}";

  // Check that all block extents had been set.
  const std::vector<const Expr*>& gpu_block_extents =
      metavar_rewriter_->gpu_block_extents();
  for (size_t i = 0; i < gpu_block_extents.size(); i++) {
    if (!gpu_block_extents[i]) {
      throw std::runtime_error("Missing gpu_block_index: " + std::to_string(i));
    }
  }

  GRAPH_DEBUG(
      "Fused TE CUDA kernel:\n",
      oss_.str(),
      "\n",
      "gpu_block_extents: (",
      metavar_rewriter_->gpu_block_extents(),
      ")\n",
      "gpu_thread_extents: (",
      metavar_rewriter_->gpu_thread_extents(),
      ")");

  CompileToNVRTC(oss_.str(), func_name);
}

void CudaCodeGen::call_raw(const std::vector<void*>& raw_args) {
  auto const& buffer_args = this->buffer_args();

  // TODO: move as much of this into the constructors.
  const std::vector<const Expr*>& gpu_block_extents =
      metavar_rewriter_->gpu_block_extents();
  const std::vector<const Expr*>& gpu_thread_extents =
      metavar_rewriter_->gpu_thread_extents();
  if (gpu_block_extents.size() > 3 || gpu_thread_extents.size() > 3) {
    throw malformed_input(
        "cuda_codegen: block or thread extent greater than 3D");
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int> gpu_block_extents_v(3, 1);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int> gpu_thread_extents_v(3, 1);

  // evaluate all the block/thread extents into values
  // TODO: eventually, codegen these calculations and make them part of the
  // module.
  for (size_t i = 0; i < gpu_block_extents.size(); i++) {
    if (gpu_block_extents[i]->isConstant()) {
      gpu_block_extents_v[i] = immediateAs<int>(gpu_block_extents[i]);
      continue;
    }
    ExprEval<SimpleIREvaluator> eval(
        ExprHandle(gpu_block_extents[i]), buffer_args);
    gpu_block_extents_v[i] = eval.value<int>(raw_args);
  }
  for (size_t i = 0; i < gpu_thread_extents.size(); i++) {
    if (gpu_thread_extents[i]->isConstant()) {
      gpu_thread_extents_v[i] = immediateAs<int>(gpu_thread_extents[i]);
      continue;
    }
    ExprEval<SimpleIREvaluator> eval(
        ExprHandle(gpu_thread_extents[i]), buffer_args);
    gpu_thread_extents_v[i] = eval.value<int>(raw_args);
  }

  // Skip launching the kernel if there are no elements to process.
  for (int extent : gpu_block_extents_v) {
    if (extent == 0) {
      return;
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int ptr_count = buffer_args.size();
  // If the kernel has a rand call in it, add two extra arguments for random
  // seed and offset.
  if (has_random_) {
    ptr_count += 2;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<void*> ptr_to_args(ptr_count);

  // In CUDA we need to pass pointers to pointers for buffers, thus we need to
  // go over raw_args and add an extra indirection for such non-scalar
  // arguments.
  // Why? See some details here:
  // https://stackoverflow.com/questions/34388712/cannot-understand-how-jcuda-culaunchkernel-work
  for (size_t i = 0; i < buffer_args.size(); i++) {
    ptr_to_args[i] =
        buffer_args[i].isVar() ? raw_args[i] : const_cast<void**>(&raw_args[i]);
  }

  if (has_random_) {
    uint64_t rand_seed = uint64_t(-1);
    uint64_t rand_offset = uint64_t(-1);
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

  const auto prior_device = at::cuda::current_device();
  if (prior_device != this->device().index()) {
    at::cuda::set_device(this->device().index());
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

  if (prior_device != this->device().index()) {
    at::cuda::set_device(prior_device);
  }
}

void CudaCodeGen::call(const std::vector<CallArg>& args) {
  if (args.size() != buffer_args().size()) {
    throw malformed_input("cuda_codegen: wrong number of args in call");
  }

  auto const& buffer_args = this->buffer_args();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<void*> raw_args(buffer_args.size());
  for (size_t i = 0; i < buffer_args.size(); i++) {
    auto const& bufferArg = buffer_args[i];
    auto const& callArg = args[i];
    raw_args[i] = argToPtr(bufferArg, callArg);
  }
  call_raw(raw_args);
}

at::Tensor CudaCodeGen::empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    c10::optional<c10::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  c10::DeviceGuard device_guard(device_opt.value());
  return at::native::empty_strided_cuda(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

void CudaCodeGen::CompileToNVRTC(
    const std::string& code,
    const std::string& func_name) {
  // NOLINTNEXTLINE(modernize-use-nullptr)
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUcontext pctx = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  // Note: hacked at::DeviceGuard since at::DeviceGuard was failing to work
  // properly in some scenarios
  const auto prior_device = at::cuda::current_device();
  if (prior_device != this->device().index()) {
    at::cuda::set_device(this->device().index());
  }
  // cudaSetDevice does not have to really change the underlying device if it
  // doesn't have to, so calling cudaFree to force that change
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
    AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  }
  // Acquires device and NVRTC properties (for compile arch and occupancy
  // calculations)
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int major, minor;
  bool compile_to_sass = false;
  codegenOutputQuery(prop, major, minor, compile_to_sass);

  // Creates the NVRTC program
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

#ifdef __HIP_PLATFORM_HCC__
  std::vector<const char*> args = {"--std=c++14"};
#if ROCM_VERSION >= 40200
  args.push_back("-hip-pch");
#endif
#else
  const std::string compute = std::string("--gpu-architecture=") +
#if CUDA_VERSION >= 11010
      // CUDA 11.1 allows going directly to SASS (sm_) instead of PTX (compute_)
      // which gives better backwards compatibility to work on older driver,
      // (since older driver doesn't necessrily recognize PTX emitted by new
      // toolkit);
      // Meanwhile, for forward compatibility (future device with
      // `compile_to_sass==false`), since SASS are not necessarily compatible,
      // we fallback to PTX instead.
      (compile_to_sass ? "sm_" : "compute_") +
#else
      "compute_" +
#endif
      std::to_string(major) + std::to_string(minor);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};
#endif

  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result != NVRTC_SUCCESS) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetProgramLogSize(program, &logsize));
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t ptx_size;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<char> ptx;
#if CUDA_VERSION >= 11010
  // compile_to_sass determines whether we are generating SASS or PTX, hence
  // the different API.
  const auto getSize = compile_to_sass
      ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
      : at::globalContext().getNVRTC().nvrtcGetPTXSize;
  const auto getFunc = compile_to_sass
      ? at::globalContext().getNVRTC().nvrtcGetCUBIN
      : at::globalContext().getNVRTC().nvrtcGetPTX;
#else
  const auto getSize = at::globalContext().getNVRTC().nvrtcGetPTXSize;
  const auto getFunc = at::globalContext().getNVRTC().nvrtcGetPTX;
#endif
  AT_CUDA_NVRTC_CHECK(getSize(program, &ptx_size));
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(getFunc(program, ptx.data()));

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUmodule module;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&module, ptx.data()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&function_, module, func_name.c_str()));

  if (prior_device != this->device().index()) {
    at::cuda::set_device(prior_device);
  }
}

CudaCodeGen::~CudaCodeGen() = default;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterCodeGen<CudaCodeGen> cuda_codegen_reg("cuda_codegen");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
