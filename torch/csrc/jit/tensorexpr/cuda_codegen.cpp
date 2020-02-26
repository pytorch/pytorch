#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"

#include "ATen/CUDAGenerator.h"
#include "c10/cuda/CUDAFunctions.h"
#include "torch/csrc/jit/tensorexpr/cuda_random.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/execution_counter.h"

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
  ScopedVarName(
      VarNameMap* mapping,
      const Var* var,
      const std::string& name)
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

  ~ScopedVarName() noexcept(false) {
    auto iter = mapping_->find(var_);
    TORCH_CHECK(iter != mapping_->end(), "Invalid var entry");
    mapping_->erase(var_);
  }

 private:
  ScopedVarName(const ScopedVarName&) = delete;
  ScopedVarName& operator=(const ScopedVarName&) = delete;

  VarNameMap* mapping_ = nullptr;
  const Var* var_ = nullptr;
};

static int as_int(const Expr* expr) {
  auto v = dynamic_cast<const IntImm*>(expr);
  TORCH_CHECK(v, "Expression is not an integer constant");
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

void CudaPrinter::visit(const For* v) {
  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    ScopedVarName var_name(
        name_manager(), v->var(), loop_options.gpu_block_index_str());
    v->body()->accept(this);
    int gpu_block_index = loop_options.gpu_block_index();
    if (gpu_block_extents_.size() <= gpu_block_index) {
      gpu_block_extents_.resize(gpu_block_index + 1);
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(ExprHandle(v->start())));
    }
    gpu_block_extents_[gpu_block_index] = v->stop();
  } else if (loop_options.is_gpu_thread_index()) {
    ScopedVarName var_name(
        name_manager(), v->var(), loop_options.gpu_thread_index_str());
    v->body()->accept(this);
    int gpu_thread_index = loop_options.gpu_thread_index();
    if (gpu_thread_extents_.size() <= gpu_thread_index) {
      gpu_thread_extents_.resize(gpu_thread_index + 1);
    }
    if (!is_zero(v->start())) {
      throw std::runtime_error(
          "start must be zero for gpu_block_index: " +
          std::to_string(ExprHandle(v->start())));
    }
    gpu_thread_extents_[gpu_thread_index] = v->stop();
  } else {
    IRPrinter::visit(v);
  }
}

void CudaPrinter::visit(const Intrinsics* v) {
  std::string func_name;
  // TODO: handle other data types.
  switch (v->op_type()) {
    case IntrinsicsOp::kSin:
      func_name = "sinf";
      break;
    case IntrinsicsOp::kCos:
      func_name = "cosf";
      break;
    case IntrinsicsOp::kExp:
      func_name = "expf";
      break;
    case IntrinsicsOp::kRand:
      os() << "Uint32ToFloat(" << *rand_func_ << "())";
      return;
    default:
      IRPrinter::visit(v);
      return;
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
  os() << "__ldg(" << *v->base_handle() << " + " << *v->index() << ")";
}

void CudaPrinter::visit(const Max* v) {
  auto dtype = v->dtype();
  if (dtype == kFloat32) {
    os() << "fmaxf";
  }
  os() << "(";
  v->lhs()->accept(this);
  os() << ",";
  v->rhs()->accept(this);
  os() << ")";
}

void CudaPrinter::visit(const Min* v) {
  auto dtype = v->dtype();
  if (dtype == kFloat32) {
    os() << "fminf";
  }
  os() << "(";
  v->lhs()->accept(this);
  os() << ",";
  v->rhs()->accept(this);
  os() << ")";
}

void CudaPrinter::visit(const IfThenElse* v) {
  os() << "(";
  v->condition()->accept(this);
  os() << ") ? ";
  v->true_value()->accept(this);
  os() << " : ";
  v->false_value()->accept(this);
}

class PrioritizeLoad : public IRMutator {
 public:
  virtual const Expr* mutate(const Load* v) {
    MemLoadList& load_list = load_stack_.back();
    const Var* load_new_var = new Var("v", v->dtype());
    const Expr* new_value = IRMutator::mutate(v);
    load_list.push_back(std::make_pair(load_new_var, new_value));
    return load_new_var;
  }

  // TODO: merge this with the IRMutator::mutate version.
  virtual Stmt* mutate(const For* v) {
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
    if (var == var_new && start == start_new &&
        stop == stop_new && body == body_with_loads) {
      return (Stmt*)v;
    }
    return new For(
        var_new, start_new, stop_new, body_with_loads, loop_options);
  }

  virtual Stmt* mutate(const LetStmt* v) {
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
    if (var == var_new && value == value_new &&
        body == body_with_loads) {
      return (Stmt*)v;
    }
    return new LetStmt(var_new, value_new, body_with_loads);
  }

  virtual Stmt* mutate(const Cond* v) {
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
    for (int i = load_list.size() - 1; i >= 0; i--) {
      const MemLoadEntry& entry = load_list[i];
      Var* var_ptr = const_cast<Var*>(entry.first);
      stmt_v = new LetStmt(var_ptr, entry.second, stmt_v);
    }
    return stmt_v;
  }

  MemoryLoadStack load_stack_;
};

class HasRand : public IRVisitor {
 public:
  HasRand(Stmt* stmt) : stmt_(stmt) {
    stmt_->accept(this);
  }

  bool has_rand() const {
    return has_rand_;
  }

 private:
  virtual void visit(const Intrinsics* v) {
    if (v->op_type() == IntrinsicsOp::kRand) {
      has_rand_ = true;
    } else {
      IRVisitor::visit(v);
    }
  }
  Stmt* stmt_;
  bool has_rand_ = false;
};

void CudaCodeGen::Initialize() {
  // TODO: handle multiple kernels.
  // TODO: handle dynamic dimension.
  // TODO: call nvrtc.
  HasRand has_rand_func(stmt());
  has_random_ = has_rand_func.has_rand();
  printer_.reset(new CudaPrinter(&oss_, has_random_));
  if (has_random_) {
    os() << philox_random_string << std::endl;
  }
  os() << "extern \"C\" __global__" << std::endl << "void f(";
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (int i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    const BufferArg& buffer_arg = buffer_args[i];
    const Var* var = buffer_arg.var();
    Dtype dtype = buffer_arg.dtype();
    os() << dtype.ToCppString() << (buffer_arg.isVar() ? " " : "* ")
         << name_manager()->get_unique_name(var);
  }
  const Var* rand_seed;
  const Var* rand_offset;
  if (has_random_) {
    // TODO: switch to kUint64 when it is available.
    rand_seed = new Var("rand_seed", kInt32);
    rand_offset = new Var("rand_offset", kInt32);
    std::string uint64_str = "unsigned long long";
    os() << ", " << uint64_str << " " << *rand_seed << ", " << uint64_str << " "
         << *rand_offset;
  }
  os() << ") {";
  os() << std::endl;

  if (has_random_) {
    const Var* idx = new Var("idx", kInt32);
    os() << "int " << *idx << " = blockIdx.x*blockDim.x + threadIdx.x;"
         << std::endl;
    const Var* rand_func = printer_->rand_func();
    os() << "Philox " << *rand_func << "(" << *rand_seed << ", " << *idx << ", "
         << *rand_offset << ");" << std::endl;
    os() << std::endl;
  }

  Stmt* stmt_v = stmt();
  PrioritizeLoad prioritize_load;
  stmt_v = prioritize_load.Process(stmt_v);
  stmt_v->accept(printer_.get());
  os() << std::endl;
  os() << "}";

  // Check that all block extents had been set.
  const std::vector<const Expr*>& gpu_block_extents = printer_->gpu_block_extents();
  const std::vector<const Expr*>& gpu_thread_extents = printer_->gpu_thread_extents();
  for (int i = 0; i < gpu_block_extents.size(); i++) {
    if (!gpu_block_extents[i]) {
      throw std::runtime_error("Missing gpu_block_index: " + std::to_string(i));
    }
  }

#if DEBUG_PRINT
  std::cout << "stmt: " << std::endl;
  std::cout << oss_.str() << std::endl;
  std::cout << "block(";
  for (int i = 0; i < gpu_block_extents.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << gpu_block_extents[i];
  }
  std::cout << "), thread(";
  for (int i = 0; i < gpu_thread_extents.size(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << gpu_thread_extents[i];
  }
  std::cout << ")" << std::endl;
  ;
#endif

  CompileToNVRTC(oss_.str());
  USE_TRIGGER(cuda_codegen_created);
}

void CudaCodeGen::call(const std::vector<CallArg>& args) {
  CHECK_EQ(args.size(), buffer_args().size());

  // TODO: move as much of this into the constructors.
  const std::vector<const Expr*>& gpu_block_extents = printer_->gpu_block_extents();
  const std::vector<const Expr*>& gpu_thread_extents = printer_->gpu_thread_extents();
  CHECK(gpu_block_extents.size() <= 3);
  CHECK(gpu_thread_extents.size() <= 3);
  std::vector<int> gpu_block_extents_v(3, 1);
  std::vector<int> gpu_thread_extents_v(3, 1);
  // evaluate all the block/thread extents into values
  // TODO: eventually, codegen these calculations and make them part of the
  // module.
  for (int i = 0; i < gpu_block_extents.size(); i++) {
    ExprEval<SimpleIREvaluator> eval(
        ExprHandle(gpu_block_extents[i]), buffer_args());
    gpu_block_extents_v[i] = eval.value<int>(args);
  }
  for (int i = 0; i < gpu_thread_extents.size(); i++) {
    ExprEval<SimpleIREvaluator> eval(
        ExprHandle(gpu_thread_extents[i]), buffer_args());
    gpu_thread_extents_v[i] = eval.value<int>(args);
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
  for (int i = 0; i < buffer_args.size(); i++) {
    auto const& bufferArg = buffer_args[i];
    if (bufferArg.isVar()) {
      auto const& dtype = bufferArg.dtype();
      if (dtype == kInt32) {
        ptr_to_args[i] = args[i].intPtr();
      } else if (dtype == kFloat32) {
        ptr_to_args[i] = args[i].floatPtr();
      } else {
        LOG(FATAL) << "Unhandled dtype in argument";
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
      std::lock_guard<std::mutex> lock(gen->mutex_);
      auto philox_engine_inputs =
          gen->philox_engine_inputs(total_elements_per_thread);
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

void CudaCodeGen::CompileToNVRTC(const std::string& code) {
  // Initializes driver's API context (if necessary)
  CUdevice device = 0;
  CUcontext pctx = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(0);
  }

  // Note: hacked at::DeviceGuard since at::DeviceGuard was failing to work
  // properly in some scenarios
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(device);

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
  std::string name = "f";
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&module, ptx.data()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&function_, module, name.c_str()));
}

RegisterCodeGen<CudaCodeGen> reg("cuda_codegen");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
