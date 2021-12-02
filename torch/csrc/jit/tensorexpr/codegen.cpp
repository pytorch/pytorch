#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>

#include <sstream>

namespace torch {
namespace jit {
namespace tensorexpr {

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

c10::optional<size_t> bufSize(BufPtr buf) {
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
// ragnes, and returns a map of these buffers to memory blocks. A memory block
// is indicated by the buffer for which it is allocated. Specifically, we
// linearly scan the intermediate buffers by the time they appear, and try to
// assign it an existing non-occupied memory block. If there are no such memory
// blocks available, we'll create memory for it. Once we are beyond the liveness
// range of this buffer, we'll mark its corresponding memory block as "up for
// grabs" for future reuse.
std::vector<std::pair<BufPtr, BufPtr>> linearScan(
    const std::unordered_set<BufPtr>& bufs,
    const std::unordered_map<BufPtr, std::tuple<int32_t, int32_t>>&
        buf_ranges) {
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
  std::unordered_map<BufPtr, BufPtr> curr_buf_mem_map;
  std::vector<std::pair<BufPtr, BufPtr>> global_buf_mem_map;

  auto sorting_function_by_end_time = [&buf_ranges](
                                          BufPtr b1, BufPtr b2) -> bool {
    return std::get<1>(buf_ranges.at(b1)) < std::get<1>(buf_ranges.at(b2));
  };
  std::vector<BufPtr> buf_to_release;
  for (auto buf : bufs_sorted) {
    auto start = std::get<0>(buf_ranges.at(buf));
    auto end = std::get<1>(buf_ranges.at(buf));

    // Release memory for buffers whose liveness range ends before the creation
    // time of this buf.
    // TODO: optimize in-place opererations and copy operations
    buf_to_release.clear();
    for (auto& mapped : curr_buf_mem_map) {
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
      mem_up_for_grabs.push_front(curr_buf_mem_map[buf_rl]);
      curr_buf_mem_map.erase(buf_rl);
    }

    // If the buf has dynamic shapes, we'll skip it (i.e., allocate memory for
    // it, and there are no future reuses on its memory).
    // TODO: reuse memory for bufs with dynamic shapes
    if (!bufSize(buf)) {
      global_buf_mem_map.emplace_back(std::make_pair(buf, buf));
      continue;
    }

    bool allocated = false;
    // Check whether there are free memories that this buf can reuse.
    for (auto it = mem_up_for_grabs.begin(); it != mem_up_for_grabs.end();
         it++) {
      auto m = *it;
      if (bufSize(m) >= bufSize(buf)) {
        curr_buf_mem_map[buf] = m;
        global_buf_mem_map.emplace_back(std::make_pair(buf, m));
        allocated = true;
        mem_up_for_grabs.erase(it);
        break;
      }
    }

    // If there are no memories to reuse, we'll have to allocate new memory for
    // it.
    if (!allocated) {
      curr_buf_mem_map[buf] = buf;
      global_buf_mem_map.emplace_back(std::make_pair(buf, buf));
    }
  }

  return global_buf_mem_map;
}

StmtPtr insertAllocFree(
    std::vector<std::pair<BufPtr, BufPtr>>& b2m,
    StmtPtr stmt) {
  BlockPtr b = to<Block>(stmt);
  if (!b) {
    b = alloc<Block>(std::vector<StmtPtr>({stmt}));
  }

  // Insert allocations and frees for temporary buffers at global scope.
  for (auto rit = b2m.rbegin(); rit != b2m.rend(); ++rit) {
    if (rit->first == rit->second) {
      BufPtr buf = rit->first;
      b->prepend_stmt(alloc<Allocate>(buf));
      b->append_stmt(alloc<Free>(buf));
    } else {
      b->prepend_stmt(alloc<PlacementAllocate>(rit->first, rit->second));
    }
  }

  return b;
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
  for (auto b : buffer_args_) {
    bufs_allocated.insert(b.buf());
  }
  auto allocs = NodeFinder<Allocate>::find(stmt_);
  for (auto a : allocs) {
    bufs_allocated.insert(a->buf());
  }

  std::unordered_set<BufPtr> interm_bufs;
  std::unordered_map<BufPtr, std::tuple<int32_t, int32_t>> interm_buf_ranges;
  for (auto buf : bufs) {
    if (!bufs_allocated.count(buf) && !interm_bufs.count(buf)) {
      interm_bufs.insert(buf);

      // Identify the access stmts to each unallocated intermeiate buffer.
      auto range = BufLiveRange::liveRange(stmt_, buf);
      // TORCH_INTERNAL_ASSERT(accesses.size() >= 1);
      // auto range =
      //    std::make_pair(accesses.at(0), accesses.at(accesses.size() - 1));
      interm_buf_ranges.emplace(buf, range);
    }
  }

  // For each intermediate buffer, we reuse the memory of an old buffer which
  // dies, or allocate memory if reusing buffer is impossible.
  auto b2m = linearScan(interm_bufs, interm_buf_ranges);

  // Insert memory allocation/mapping nodes.
  if (b2m.size() > 0) {
    auto stmt_new = insertAllocFree(b2m, stmt_);
    set_stmt(stmt_new);
  }

  GRAPH_DEBUG("\nMemory Allocation:\n\n", *stmt(), "\n");
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
