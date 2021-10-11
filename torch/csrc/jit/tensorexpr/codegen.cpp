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

size_t bufSize(BufPtr buf) {
  size_t size = elementSize(buf->dtype().scalar_type()) * buf->dtype().lanes();
  for (auto& d : buf->dims()) {
    if (!d->isConstant()) {
      return 0;
    }
    size = size * (*intValue(d));
  }
  return size;
}

using BufRangeInfo =
    std::unordered_map<BufPtr, std::pair<BufAccessInfo, BufAccessInfo>>;

std::vector<std::pair<BufPtr, BufPtr>> linerScan(
    std::unordered_set<BufPtr>& bufs,
    BufRangeInfo& buf_ranges) {
  // Sort buffers by the time they appear.
  std::vector<BufPtr> bufs_sorted(bufs.begin(), bufs.end());
  auto sorting_function_by_start_time = [&buf_ranges](
                                            BufPtr b1, BufPtr b2) -> bool {
    auto start1 = buf_ranges.at(b1).first;
    auto start2 = buf_ranges.at(b2).first;
    return std::get<2>(start1) < std::get<2>(start2);
  };
  std::sort(
      bufs_sorted.begin(), bufs_sorted.end(), sorting_function_by_start_time);
  for (auto buf : bufs_sorted) {
    auto start = buf_ranges.at(buf).first;
    auto end = buf_ranges.at(buf).second;
  }

  auto sorting_function_by_end_time = [&buf_ranges](
                                          BufPtr b1, BufPtr b2) -> bool {
    auto end1 = buf_ranges.at(b1).second;
    auto end2 = buf_ranges.at(b2).second;
    return std::get<2>(end1) < std::get<2>(end2);
  };

  // Map intermediate buffers to the most recent used memory if any.
  std::unordered_set<BufPtr> mm;
  std::list<BufPtr> mm_free;
  std::unordered_map<BufPtr, BufPtr> b2m;
  std::vector<std::pair<BufPtr, BufPtr>> b2m_ret;

  std::vector<BufPtr> buf_to_release;
  for (auto buf : bufs_sorted) {
    auto start = buf_ranges.at(buf).first;
    auto end = buf_ranges.at(buf).second;

    // Release memory for buffers whose live range ends before the creation time
    // of this buf.
    // TODO: optimize in-place opererations and copy operations
    buf_to_release.clear();
    for (auto& mapped : b2m) {
      auto buf_mapped = mapped.first;
      auto end_buf_mapped = buf_ranges.at(buf_mapped).second;
      if (std::get<2>(end_buf_mapped) < std::get<2>(start)) {
        buf_to_release.push_back(buf_mapped);
      }
    }
    std::sort(
        buf_to_release.begin(),
        buf_to_release.end(),
        sorting_function_by_end_time);
    for (auto& buf_rl : buf_to_release) {
      mm_free.push_front(b2m[buf_rl]);
      b2m.erase(buf_rl);
    }

    // If the buf has dynamic shapes, we'll skip it (i.e., allocate memory for
    // it, and there are no future reuses on its memory).
    // TODO: reuse memory for bufs with dynamic shapes
    if (bufSize(buf) == 0) {
      b2m_ret.emplace_back(std::make_pair(buf, buf));
      continue;
    }

    bool allocated = false;
    // Check whether there are free memories that this buf can reuse.
    for (auto it = mm_free.begin(); it != mm_free.end(); it++) {
      auto m = *it;
      TORCH_INTERNAL_ASSERT(bufSize(buf) != 0);
      TORCH_INTERNAL_ASSERT(bufSize(m) != 0);
      if (bufSize(m) >= bufSize(buf)) {
        b2m[buf] = m;
        b2m_ret.emplace_back(std::make_pair(buf, m));
        allocated = true;
        mm_free.erase(it);
        break;
      }
    }

    // If there are no memories to reuse, we'll have to allocate new memory for
    // it.
    if (!allocated) {
      mm.insert(buf);
      b2m[buf] = buf;
      b2m_ret.emplace_back(std::make_pair(buf, buf));
    }
  }

  return b2m_ret;
}

StmtPtr insertMemNodes(
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
      b->prepend_stmt(alloc<BufMap>(rit->first, rit->second));
    }
  }

  return b;
}

void CodeGen::allocBuf() {
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
  BufRangeInfo interm_buf_ranges;
  for (auto buf : bufs) {
    if (!bufs_allocated.count(buf) && !interm_bufs.count(buf)) {
      interm_bufs.insert(buf);

      // Identify the access stmts to each unallocated intermeiate buffer.
      auto accesses = BufAccesses::find(stmt_, buf);
      TORCH_INTERNAL_ASSERT(accesses.size() >= 1);
      auto range =
          std::make_pair(accesses.at(0), accesses.at(accesses.size() - 1));
      interm_buf_ranges.emplace(buf, range);
    }
  }

  // For each intermediate buffer, we reuse the memory of an old buffer which
  // dies, or allocate memory if reusing buffer is impossible.
  auto b2m = linerScan(interm_bufs, interm_buf_ranges);

  // Insert memory allocation/mapping nodes.
  if (b2m.size() > 0) {
    auto stmt_new = insertMemNodes(b2m, stmt_);
    set_stmt(stmt_new);
  }

  GRAPH_DEBUG("\nMemomry Allocation:\n\n", *stmt(), "\n");
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
