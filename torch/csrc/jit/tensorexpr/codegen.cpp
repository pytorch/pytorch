#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>

#include <sstream>

namespace torch::jit::tensorexpr {

CodeGen::CodeGen(
    StmtPtr stmt,
    std::vector<BufferArg> buffer_args,
    at::Device device,
    std::string kernel_func_name)
    : stmt_(std::move(stmt)),
      buffer_args_(std::move(buffer_args)),
      device_(device),
      kernel_func_name_(std::move(kernel_func_name)) {
  ExtCallMemoryReuse extCallMemoryReuse(buffer_args_);
  apply_mutator(&extCallMemoryReuse);
  allocIntermediateBufs();
}

RegisterCodeGenList::StmtFactoryMethod RegisterCodeGenList::
    FindStmtFactoryMethod(const std::string& name) {
  auto iter = stmt_factory_methods_.find(name);
  if (iter == stmt_factory_methods_.end()) {
    std::ostringstream oss;
    oss << "Invalid stmt codegen name: " << name << ". ";
    oss << "Existing codegen names: [";
    int index = 0;
    for (auto& entry : stmt_factory_methods_) {
      if (index != 0) {
        oss << ", ";
      }
      oss << entry.first;
      index++;
    }
    oss << "]";
    throw std::runtime_error(oss.str());
  }
  return iter->second;
}

void RegisterCodeGenList::AddStmtFactoryMethod(
    const std::string& name,
    const StmtFactoryMethod& stmt_factory_method) {
  stmt_factory_methods_[name] = stmt_factory_method;
}

std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& params,
    at::Device device,
    const std::string& kernel_func_name) {
  RegisterCodeGenList::StmtFactoryMethod method =
      RegisterCodeGenList::GetInstance().FindStmtFactoryMethod(name);
  return method(stmt, params, device, kernel_func_name);
}

ExprPtr GenericIntrinsicsExpander::mutate(IntrinsicsPtr v) {
  if (v->op_type() == kSigmoid) {
    auto x = v->param(0)->accept_mutator(this);
    auto one = expr_to_vec(
        ExprHandle(getImmediateByType(v->dtype(), 1.0)), v->dtype().lanes());
    auto zero = expr_to_vec(
        ExprHandle(getImmediateByType(v->dtype(), 0.0)), v->dtype().lanes());
    ExprHandle y = one / (one + exp(zero - ExprHandle(x)));
    return y.node();
  }
  return IRMutator::mutate(v);
}

void* CodeGen::argToPtr(const BufferArg& bufferArg, const CallArg& callArg) {
  if (!bufferArg.isVar()) {
    return callArg.data();
  }

  switch (bufferArg.dtype().scalar_type()) {
#define TYPE_CASE(_1, Name) \
  case ScalarType::Name:    \
    return callArg.Name##Ptr();

    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE

    default:
      throw unsupported_dtype();
  }
  return nullptr;
}

void CodeGen::call_with_numel(void** args, int64_t numel) {
  TORCH_INTERNAL_ASSERT(
      false, "This codegen backend does not implement call_with_numel");
}

static c10::optional<size_t> bufSize(BufPtr buf) {
  size_t size = elementSize(buf->dtype().scalar_type()) * buf->dtype().lanes();
  for (auto& d : buf->dims()) {
    if (!d->isConstant()) {
      return c10::nullopt;
    }
    size = size * (*intValue(d));
  }
  return size;
}

// This algorithm takes the list of intermediate buffers and their liveness
// ranges, and returns the allocations of these buffers. A buffer 'A' can be
// allocated in the memory (appears as a pair of 'A's in the allocation results)
// or reuse another buffer such as 'B' (appears as ('A', 'B')). Specifically, we
// linearly scan the intermediate buffers by the time they appear, and try to
// assign it an existing non-occupied memory allocation. If there are no such
// allocations available, we'll create memory for it. Once we are beyond the
// liveness range of this buffer, we'll mark its corresponding memory allocation
// as "up for grabs" for future reuse.
static std::vector<std::pair<BufPtr, BufPtr>> AllocBufsWithMemReuse(
    const std::unordered_set<BufPtr>& bufs,
    const std::unordered_map<BufPtr, std::tuple<int32_t, int32_t>>& buf_ranges,
    const std::unordered_set<BufPtr>& bufs_external_allocs) {
  // Sort buffers by the time they appear.
  std::vector<BufPtr> bufs_sorted(bufs.begin(), bufs.end());
  auto sorting_function_by_start_time = [&buf_ranges](
                                            BufPtr b1, BufPtr b2) -> bool {
    return std::get<0>(buf_ranges.at(b1)) < std::get<0>(buf_ranges.at(b2));
  };
  std::sort(
      bufs_sorted.begin(), bufs_sorted.end(), sorting_function_by_start_time);

  // Map intermediate buffers to the most recently used memory if any.
  std::list<BufPtr> mem_up_for_grabs;
  std::unordered_map<BufPtr, BufPtr> buf_mem_map;
  std::vector<std::pair<BufPtr, BufPtr>> buf_allocs;

  auto sorting_function_by_end_time = [&buf_ranges](
                                          BufPtr b1, BufPtr b2) -> bool {
    return std::get<1>(buf_ranges.at(b1)) < std::get<1>(buf_ranges.at(b2));
  };
  for (const auto& buf : bufs_sorted) {
    // If the buf has dynamic shapes, we'll skip it (i.e., allocate memory for
    // it, and there are no future reuses on its memory).
    // TODO: reuse memory for bufs with dynamic shapes
    if (!bufSize(buf)) {
      buf_allocs.emplace_back(buf, buf);
      continue;
    }

    auto start = std::get<0>(buf_ranges.at(buf));

    // Release memory for buffers whose liveness range ends before the creation
    // time of this buf.
    // TODO: optimize in-place operations and copy operations
    std::vector<BufPtr> buf_to_release;
    for (auto& mapped : buf_mem_map) {
      auto buf_mapped = mapped.first;
      auto end_buf_mapped = std::get<1>(buf_ranges.at(buf_mapped));
      if (end_buf_mapped < start) {
        buf_to_release.push_back(buf_mapped);
      }
    }

    // Sort the buffers in the order of used time so the head of the release
    // list contains the most recently used buf.
    std::sort(
        buf_to_release.begin(),
        buf_to_release.end(),
        sorting_function_by_end_time);
    for (auto& buf_rl : buf_to_release) {
      mem_up_for_grabs.push_front(buf_mem_map.at(buf_rl));
      buf_mem_map.erase(buf_rl);
    }

    bool allocated = false;
    if (bufs_external_allocs.find(buf) == bufs_external_allocs.end()) {
      // Check whether there are free memories that this buf can reuse.
      for (auto it = mem_up_for_grabs.begin(); it != mem_up_for_grabs.end();
           it++) {
        auto m = *it;
        if (bufSize(m) >= bufSize(buf)) {
          buf_mem_map[buf] = m;
          buf_allocs.emplace_back(buf, m);
          allocated = true;
          mem_up_for_grabs.erase(it);
          break;
        }
      }
    }

    // If there are no memories to reuse, we'll have to allocate new memory for
    // it.
    if (!allocated) {
      buf_mem_map[buf] = buf;
      buf_allocs.emplace_back(buf, buf);
    }
  }

  return buf_allocs;
}

static StmtPtr insertAllocFree(
    std::vector<std::pair<BufPtr, BufPtr>>& buf_allocs,
    const std::unordered_set<BufPtr>& bufs_external_allocs,
    StmtPtr stmt) {
  BlockPtr b = to<Block>(stmt);
  if (!b) {
    b = alloc<Block>(std::vector<StmtPtr>({stmt}));
  }

  std::vector<BufPtr> bufs_ext_to_free;
  // Insert allocations and frees for temporary buffers at global scope.
  for (auto rit = buf_allocs.rbegin(); rit != buf_allocs.rend(); ++rit) {
    if (rit->first == rit->second) {
      BufPtr buf = rit->first;
      if (bufs_external_allocs.find(buf) == bufs_external_allocs.end()) {
        b->prepend_stmt(alloc<Allocate>(buf));
        b->append_stmt(alloc<Free>(buf));
      } else {
        bufs_ext_to_free.push_back(buf);
      }
    } else {
      b->prepend_stmt(alloc<PlacementAllocate>(rit->first, rit->second));
    }
  }

  b->append_stmt(alloc<FreeExt>(bufs_ext_to_free));
  return b;
}

std::unordered_map<std::string, std::string> ExtCallMemoryReuse::
    makeExtCallFuncNameMap() {
  return {
      {"nnc_aten_quantize_per_tensor", "nnc_aten_quantize_per_tensor_out"},
      {"nnc_aten_dequantize", "nnc_aten_dequantize_out"},
      {"nnc_aten_quantized_mul", "nnc_aten_quantized_mul_out"},
      {"nnc_aten_quantized_conv2d", "nnc_aten_quantized_conv2d_out"},
      {"nnc_aten_quantized_conv2d_relu", "nnc_aten_quantized_conv2d_relu_out"},
      {"nnc_aten_quantized_mul", "nnc_aten_quantized_mul_out"},
      {"nnc_aten_quantized_sigmoid", "nnc_aten_quantized_sigmoid_out"},
      {"nnc_aten_upsample_nearest2d", "nnc_aten_upsample_nearest2d_out"},
      {"nnc_aten_quantized_linear", "nnc_aten_quantized_linear_out"},
      {"nnc_aten_quantized_conv1d", "nnc_aten_quantized_conv1d_out"},
      {"nnc_aten_quantized_mul_scalar", "nnc_aten_quantized_mul_scalar_out"},
      {"nnc_aten_max_red", "nnc_aten_max_red_out"},
      {"nnc_aten_conv1d", "nnc_aten_conv1d_out"},
  };
}

const std::unordered_map<std::string, std::string>
    ExtCallMemoryReuse::extCallFuncNameMap_ = makeExtCallFuncNameMap();

ExtCallMemoryReuse::ExtCallMemoryReuse(
    const std::vector<CodeGen::BufferArg>& bufferArgs) {
  for (const auto& ba : bufferArgs) {
    if (ba.buf()) {
      bufferArgs_.insert(ba.buf());
    }
  }
}

StmtPtr ExtCallMemoryReuse::mutate(ExternalCallPtr v) {
  if (extCallFuncNameMap_.count(v->func_name()) &&
      bufferArgs_.count(v->buf()) == 0) {
    std::vector<BufPtr> buf_out_args = {v->buf()};
    return alloc<ExternalCallWithAlloc>(
        extCallFuncNameMap_.at(v->func_name()),
        buf_out_args,
        v->buf_args(),
        v->args());
  }
  return v;
}

// We allocate intermediate buffers by inserting Allocate/Free or
// PlacementAllocate stmts. Allocate/Free stmts will allocate memory at runtime,
// and PlacementAllocate stmt reuses the memory of one buffer for another
// buffer. In current implementation, we use linear scan for memory reuses.
// TODO: try more memory reuse algorithms and compare their memory efficiency.
void CodeGen::allocIntermediateBufs() {
  // Identify intermediate buffers that are not allocated yet.
  auto bufs = NodeFinder<Buf>::find(stmt_);
  std::unordered_set<BufPtr> bufs_allocated;
  for (const auto& b : buffer_args_) {
    bufs_allocated.insert(b.buf());
  }
  auto allocs = NodeFinder<Allocate>::find(stmt_);
  for (const auto& a : allocs) {
    bufs_allocated.insert(a->buf());
  }

  std::unordered_set<BufPtr> interm_bufs;
  std::unordered_map<BufPtr, std::tuple<int32_t, int32_t>> interm_buf_ranges;
  for (const auto& buf : bufs) {
    if (!bufs_allocated.count(buf) && !interm_bufs.count(buf)) {
      interm_bufs.insert(buf);

      // Identify the access stmts to each unallocated intermediate buffer.
      auto range = BufLiveRange::liveRange(stmt_, buf);
      interm_buf_ranges.emplace(buf, range);
    }
  }

  const auto bufs_external_allocs = ExternalAllocBufFinder::find(stmt_);

  // For each intermediate buffer, we reuse the memory of an old buffer whose
  // liveness range does not overlap with the current buffer, or allocate memory
  // if reusing buffer is impossible.
  auto buf_allocs = AllocBufsWithMemReuse(
      interm_bufs, interm_buf_ranges, bufs_external_allocs);

  // Insert memory allocation/mapping nodes.
  if (!buf_allocs.empty()) {
    auto stmt_new = insertAllocFree(buf_allocs, bufs_external_allocs, stmt_);
    set_stmt(stmt_new);
  }

  GRAPH_DEBUG("\nMemory Allocation:\n\n", *stmt(), "\n");
}

} // namespace torch::jit::tensorexpr
