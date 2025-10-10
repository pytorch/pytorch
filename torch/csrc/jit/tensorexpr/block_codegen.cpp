#include <torch/csrc/jit/tensorexpr/block_codegen.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch::jit::tensorexpr {

static std::string blockDtypeCppString(const Dtype& dtype) {
  switch (dtype.scalar_type()) {
    case ScalarType::Bool:
      return "1";
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Half:
      return "2";
    case ScalarType::BFloat16:
      return "2";
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ScalarType::Char:
      return "1";
    case ScalarType::Byte:
      return "1";
    case ScalarType::Short:
      return "4";
    case ScalarType::Long:
      return "8";
    case ScalarType::Float:
      return "2"; // Return Half for now
    default:
      return dtype.ToCppString();
  }
}

bool BlockAnalysis::areBufsInMap(const std::unordered_set<BufPtr>& bufs) const {
  for (auto const& arg : bufs) {
    auto got = map_input_to_tensor_bufs_.find(arg->name_hint());
    if (got == map_input_to_tensor_bufs_.end()) {
      return false;
    }
  }
  return true;
}

BufPtr BlockAnalysis::getMultiDimBuf(const BufPtr& buf) const {
  auto input_ = map_input_to_tensor_bufs_.find(buf->name_hint());
  if (input_ != map_input_to_tensor_bufs_.end()) {
    return input_->second;
  } else {
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  }
}

std::string BlockAnalysis::getInputName(const BufPtr& buf) const {
  auto input_ = map_input_to_tensor_bufs_.find(buf->name_hint());
  if (input_ != map_input_to_tensor_bufs_.end()) {
    return input_->second->name_hint();
  } else {
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  }
}

void BlockAnalysis::visit(const StorePtr& v) {
  store_targets_.insert(v->buf());
  v->value()->accept(this);
}

void BlockAnalysis::visit(const LoadPtr& v) {
  loads_.insert(v->buf());
}

void BlockAnalysis::visit(const ForPtr& v) {
  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    map_input_to_tensor_bufs_ = loop_options.get_buffer_mapping();
    v->body()->accept(this);
  } else if (loop_options.is_gpu_thread_index()) {
    auto block_size = v->stop();
    block_size_ = *intValue(block_size);
    v->body()->accept(this);
  } else {
    IRVisitor::visit(v);
  }
}

// For both Add, Mul we only print out the opening
// parenthesis. This behavior is to handle blocks add Op
// where c=a+b becomes add(a, b, c). The closing parenthesis is
// added in the store statement.
// TODO: When handling fused ops d = a + b + c, the correct
// way would be to mutate the expression to Block version and print.

void BlockPrinter::visit(const AddPtr& v) {
  emitIndent();
  os() << "add(";
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

void BlockPrinter::visit(const MulPtr& v) {
  emitIndent();
  os() << "mul(";
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

void BlockPrinter::visit(const ForPtr& v) {
  const LoopOptions& loop_options = v->loop_options();

  auto buf_reads = block_analysis_->loads();
  auto buf_writes = block_analysis_->stores();
  std::unordered_set<BufPtr> bufs(buf_reads.begin(), buf_reads.end());
  bufs.insert(buf_writes.begin(), buf_writes.end());

  if (loop_options.is_gpu_block_index()) {
    emitIndent();
    PrintTensorInfo(bufs);
    PrintDistribution(bufs);
    PrintBufferInfo(buf_reads);
    PrintArguments(bufs);

    emitIndent();
    os() << "compute {" << '\n';

    PrintReshapeInfo(bufs);

    emitIndent();
    PrintLoop(bufs, true);
    v->body()->accept(this);

    os() << '\n';
    emitIndent();
    PrintReshapeInfo(buf_writes, true); // print reverse reshape
    os() << "}";
    os() << '\n';
  } else if (loop_options.is_gpu_thread_index()) {
    PrintDMAs(buf_reads);
    PrintLoop(buf_reads, false);
    v->body()->accept(this);
    os() << '\n';
    PrintAdjustBuffers(buf_reads);

  } else {
    IRPrinter::visit(v);
  }
}

void BlockPrinter::PrintTensorInfo(const std::unordered_set<BufPtr>& bufs) {
  os() << "tensors {";
  for (auto& buf : bufs) {
    os() << '\n';
    emitIndent();
    emitIndent();
    auto num_dims = block_analysis_->getMultiDimBuf(buf)->dims().size();
    os() << block_analysis_->getInputName(buf) << " = ";
    os() << "{";
    for (unsigned long d = 0; d < num_dims; d++) {
      os() << "{" << dim_names[d] << "};";
    }
    os() << " elem : " << blockDtypeCppString(buf->dtype());
    os() << "}";
  }

  for (auto& buf : bufs) {
    os() << '\n';
    emitIndent();
    emitIndent();
    auto num_dims = block_analysis_->getMultiDimBuf(buf)->dims().size();
    os() << block_analysis_->getFlatInputName(buf) << " = ";
    os() << "{";
    os() << "{" << flat_dim_names[num_dims - 1] << "};";
    os() << " elem : " << blockDtypeCppString(buf->dtype());
    os() << "}"
         << " // flattened tensor";
  }
  os() << '\n';
  emitIndent();
  os() << "}" << '\n' << '\n';
}

void BlockPrinter::PrintArguments(const std::unordered_set<BufPtr>& bufs) {
  for (auto& buf : bufs) {
    auto multidimbuf = block_analysis_->getMultiDimBuf(buf);
    auto num_dims = multidimbuf->dims().size();

    // The dims for the multi-dim tensors
    for (unsigned long d = 0; d < num_dims; d++) {
      auto dim_val = *intValue(multidimbuf->dim(d));
      this->dim_values_map.emplace(this->dim_names[d], dim_val);
    }

    // The dimensions for the flattened tensors
    auto val = *intValue(buf->dim(0));
    if (block_analysis_->is_buf_store_target(buf)) {
      this->dim_values_map.emplace(this->flat_dim_names[num_dims - 1], val);
    }
  }

  emitIndent();
  os() << "arguments {" << '\n';

  for (auto const& arg : this->dim_values_map) {
    emitIndent();
    os() << "var " << arg.first << " = " << arg.second << '\n';
  }

  emitIndent();
  emitIndent();
  auto blck_sz = block_analysis_->block_size();
  os() << "var bs_N = " << blck_sz << '\n';
  emitIndent();
  emitIndent();
  os() << "var bs_DPE = " << blck_sz << '\n';
  emitIndent();
  os() << "}" << '\n' << '\n';
}

void BlockPrinter::PrintBufferInfo(const std::unordered_set<BufPtr>& bufs) {
  emitIndent();
  os() << "buffers {";
  for (auto& read : bufs) {
    os() << '\n';
    emitIndent();
    emitIndent();
    os() << block_analysis_->getFlatInputName(read) << " = ";
    os() << "{{"
         << "bs_DPE"
         << "}}";
  }
  os() << '\n';
  emitIndent();
  os() << "}" << '\n' << '\n';
}

void BlockPrinter::PrintDistribution(const std::unordered_set<BufPtr>& bufs) {
  emitIndent();
  os() << "distribution {" << '\n';
  for (auto& buf : bufs) {
    emitIndent();
    emitIndent();
    os() << block_analysis_->getFlatInputName(buf) << " = ";
    os() << "{(0, 1, )}" << '\n';
  }
  os() << "  }" << '\n' << '\n';
}

void BlockPrinter::PrintLoop(
    const std::unordered_set<BufPtr>& bufs,
    bool block_idx) {
  emitIndent();
  os() << "loop (";
  auto trip = 0;
  for (auto& buf : bufs) {
    if (trip > 0) {
      os() << ",";
    }
    os() << "{dim : ";
    os() << block_analysis_->getFlatInputName(buf) << ".dim.0, ";
    os() << (block_idx ? "block: bs_N}" : "block: bs_DPE}");
    ++trip;
  }
  os() << ")";
}

void BlockPrinter::PrintReshapeInfo(
    const std::unordered_set<BufPtr>& bufs,
    bool reverse) {
  for (auto& buf : bufs) {
    emitIndent();
    os() << "reshape("
         << (reverse ? block_analysis_->getFlatInputName(buf)
                     : block_analysis_->getInputName(buf))
         << ", "
         << (reverse ? block_analysis_->getInputName(buf)
                     : block_analysis_->getFlatInputName(buf))
         << ")" << '\n';
  }
}

void BlockPrinter::PrintDMAs(const std::unordered_set<BufPtr>& bufs) {
  for (auto& read : bufs) {
    emitIndent();
    os() << "dma_in(";
    os() << block_analysis_->getFlatInputName(read);
    os() << ")" << '\n';
  }
}
void BlockPrinter::PrintAdjustBuffers(const std::unordered_set<BufPtr>& bufs) {
  for (auto& read : bufs) {
    emitIndent();
    os() << "adjust_buffer(";
    os() << block_analysis_->getFlatInputName(read);
    os() << ")" << '\n';
  }
}

void BlockPrinter::visit(const LoadPtr& v) {
  os() << block_analysis_->getFlatInputName(v->buf()) << ".buffer, ";
}
void BlockPrinter::visit(const StorePtr& v) {
  emitIndent();
  os() << *v->value() << block_analysis_->getFlatInputName(v->buf())
       << ".tensor)" << '\n';
}

void BlockPrinter::visit(const BlockPtr& v) {
  os() << "{" << '\n';
  indent_++;
  for (const StmtPtr& s : v->stmts()) {
    s->accept(this);
  }
  indent_--;
  emitIndent();
  os() << "}";
}

std::string BlockCodeGen::GetUniqueFuncName(const std::string& func_prefix) {
  // We are using a global counter here to make sure difference instances
  // within BlockCodeGen have different names.
  static int64_t counter = 0;
  ++counter;
  int64_t value = counter;
  return func_prefix + "_" + std::to_string(value);
}

void BlockCodeGen::Initialize() {
  block_analysis_ = std::make_unique<BlockAnalysis>();
  printer_ = std::make_unique<BlockPrinter>(&oss_, block_analysis_.get());

  StmtPtr stmt_v = stmt();
  stmt_v->accept(block_analysis_.get());

  auto buf_reads = block_analysis_->loads();
  auto buf_writes = block_analysis_->stores();
  // Ensure all Bufs in reads/writes are in the map
  std::unordered_set<BufPtr> bufs(buf_reads.begin(), buf_reads.end());
  bufs.insert(buf_writes.begin(), buf_writes.end());
  if (!block_analysis_->areBufsInMap(bufs)) {
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  };

  std::string func_name = GetUniqueFuncName("func");
  os() << "kernel " << func_name << "(";
  for (auto const& arg : buf_writes) {
    os() << block_analysis_->getInputName(arg);
  }
  for (auto const& arg : buf_reads) {
    os() << ";" << block_analysis_->getInputName(arg);
  }
  os() << ")";

  stmt_v->accept(printer_.get());

  GRAPH_DEBUG("Generated Block code: ", oss_.str(), "\n");
}

void BlockCodeGen::call(const std::vector<CallArg>& args) {
  throw std::runtime_error("BlockCodeGen: Cannot call Block code ");
}
void BlockCodeGen::call_raw(const std::vector<void*>& args) {
  throw std::runtime_error("BlockCodeGen: Cannot call Block code ");
}

BlockCodeGen::~BlockCodeGen() = default;
static RegisterCodeGen<BlockCodeGen> block_codegen_reg("block_codegen");

} // namespace torch::jit::tensorexpr
