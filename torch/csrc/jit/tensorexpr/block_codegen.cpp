#include <torch/csrc/jit/tensorexpr/block_codegen.h>

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/execution_counter.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_TRIGGER(block_codegen_created);
std::string blockDtypeCppString(const Dtype& dtype) {
  switch (dtype.scalar_type()) {
    case ScalarType::Bool:
      return "1";
    case ScalarType::Half:
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

bool BlockAnalysis::areBufsInMap(
    const std::unordered_set<const Buf*>& bufs) const {
  for (auto const& arg : bufs) {
    auto got = map_input_to_tensor_bufs_.find(arg->name_hint());
    if (got == map_input_to_tensor_bufs_.end()) {
      return false;
    }
  }
  return true;
}

const Buf* BlockAnalysis::getMultiDimBuf(const Buf* buf) const {
  auto input_ = map_input_to_tensor_bufs_.find(buf->name_hint());
  if (input_ != map_input_to_tensor_bufs_.end()) {
    return input_->second;
  } else {
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  }
}

std::string BlockAnalysis::getInputName(const Buf* buf) const {
  auto input_ = map_input_to_tensor_bufs_.find(buf->name_hint());
  if (input_ != map_input_to_tensor_bufs_.end()) {
    return input_->second->name_hint();
  } else {
    throw std::runtime_error("BlockCodeGen: Entry not in input/Buffer map");
  }
}

void BlockAnalysis::visit(const Store* v) {
  store_targets_.insert(v->buf());
  v->value()->accept(this);
}

void BlockAnalysis::visit(const Load* v) {
  loads_.insert(v->buf());
}

void BlockAnalysis::visit(const For* v) {
  const LoopOptions& loop_options = v->loop_options();
  if (loop_options.is_gpu_block_index()) {
    map_input_to_tensor_bufs_ = loop_options.get_buffer_mapping();
    v->body()->accept(this);
  } else if (loop_options.is_gpu_thread_index()) {
    auto block_size = v->stop();
    block_size_ = dynamic_cast<const IntImm*>(block_size)->value();
    v->body()->accept(this);
  } else {
    IRVisitor::visit(v);
  }
}

// For both Add, Mul we only print out the opening
// paranthesis. This behavior is to handle blocks add Op
// where c=a+b becomes add(a, b, c). The closing paran is
// added in the store statement.
// TODO: When handling fused ops d = a + b + c, the correct
// way would be to mutate the expression to Block version and print.

void BlockPrinter::visit(const Add* v) {
  emitIndent();
  os() << "add(";
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

void BlockPrinter::visit(const Mul* v) {
  emitIndent();
  os() << "mul(";
  v->lhs()->accept(this);
  v->rhs()->accept(this);
}

void BlockPrinter::visit(const For* v) {
  const LoopOptions& loop_options = v->loop_options();

  auto buf_reads = block_analysis_->loads();
  auto buf_writes = block_analysis_->stores();
  std::unordered_set<const Buf*> bufs(buf_reads.begin(), buf_reads.end());
  bufs.insert(buf_writes.begin(), buf_writes.end());

  if (loop_options.is_gpu_block_index()) {
    emitIndent();
    PrintTensorInfo(bufs);
    PrintDistribution(bufs);
    PrintBufferInfo(buf_reads);
    PrintArguments(bufs);

    emitIndent();
    os() << "compute {" << std::endl;

    PrintReshapeInfo(bufs);

    emitIndent();
    PrintLoop(bufs, true);
    v->body()->accept(this);

    os() << std::endl;
    emitIndent();
    PrintReshapeInfo(buf_writes, true); // print reverse reshape
    os() << "}";
    os() << std::endl;
  } else if (loop_options.is_gpu_thread_index()) {
    PrintDMAs(buf_reads);
    PrintLoop(buf_reads, false);
    v->body()->accept(this);
    os() << std::endl;
    PrintAdjustBuffers(buf_reads);

  } else {
    IRPrinter::visit(v);
  }
}

void BlockPrinter::PrintTensorInfo(const std::unordered_set<const Buf*>& bufs) {
  os() << "tensors {";
  for (const auto& buf : bufs) {
    os() << std::endl;
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

  for (const auto& buf : bufs) {
    os() << std::endl;
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
  os() << std::endl;
  emitIndent();
  os() << "}" << std::endl << std::endl;
}

void BlockPrinter::PrintArguments(const std::unordered_set<const Buf*>& bufs) {
  for (const auto& buf : bufs) {
    auto multidimbuf = block_analysis_->getMultiDimBuf(buf);
    auto num_dims = multidimbuf->dims().size();

    // The dims for the multi-dim tensors
    for (unsigned long d = 0; d < num_dims; d++) {
      auto dim_val = dynamic_cast<const IntImm*>(multidimbuf->dim(d));
      this->dim_values_map.emplace(this->dim_names[d], dim_val->value());
    }

    // The dimensions for the flattened tensors
    auto val = dynamic_cast<const IntImm*>(buf->dim(0));
    if (block_analysis_->is_buf_store_target(buf)) {
      this->dim_values_map.emplace(
          this->flat_dim_names[num_dims - 1], val->value());
    }
  }

  emitIndent();
  os() << "arguments {" << std::endl;

  for (auto const& arg : this->dim_values_map) {
    emitIndent();
    os() << "var " << arg.first << " = " << arg.second << std::endl;
  }

  emitIndent();
  emitIndent();
  auto blck_sz = block_analysis_->block_size();
  os() << "var bs_N = " << blck_sz << std::endl;
  emitIndent();
  emitIndent();
  os() << "var bs_DPE = " << blck_sz << std::endl;
  emitIndent();
  os() << "}" << std::endl << std::endl;
}

void BlockPrinter::PrintBufferInfo(const std::unordered_set<const Buf*>& bufs) {
  emitIndent();
  os() << "buffers {";
  for (const auto& read : bufs) {
    os() << std::endl;
    emitIndent();
    emitIndent();
    os() << block_analysis_->getFlatInputName(read) << " = ";
    os() << "{{"
         << "bs_DPE"
         << "}}";
  }
  os() << std::endl;
  emitIndent();
  os() << "}" << std::endl << std::endl;
}

void BlockPrinter::PrintDistribution(
    const std::unordered_set<const Buf*>& bufs) {
  emitIndent();
  os() << "distribution {" << std::endl;
  for (const auto& buf : bufs) {
    emitIndent();
    emitIndent();
    auto buf_name = buf->name_hint();
    os() << block_analysis_->getFlatInputName(buf) << " = ";
    os() << "{(0, 1, )}" << std::endl;
  }
  os() << "  }" << std::endl << std::endl;
}

void BlockPrinter::PrintLoop(
    const std::unordered_set<const Buf*>& bufs,
    bool block_idx) {
  emitIndent();
  os() << "loop (";
  auto trip = 0;
  for (const auto& buf : bufs) {
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
    const std::unordered_set<const Buf*>& bufs,
    bool reverse) {
  for (const auto& buf : bufs) {
    emitIndent();
    os() << "reshape("
         << (reverse ? block_analysis_->getFlatInputName(buf)
                     : block_analysis_->getInputName(buf))
         << ", "
         << (reverse ? block_analysis_->getInputName(buf)
                     : block_analysis_->getFlatInputName(buf))
         << ")" << std::endl;
  }
}

void BlockPrinter::PrintDMAs(const std::unordered_set<const Buf*>& bufs) {
  for (const auto& read : bufs) {
    emitIndent();
    os() << "dma_in(";
    os() << block_analysis_->getFlatInputName(read);
    os() << ")" << std::endl;
  }
}
void BlockPrinter::PrintAdjustBuffers(
    const std::unordered_set<const Buf*>& bufs) {
  for (const auto& read : bufs) {
    emitIndent();
    os() << "adjust_buffer(";
    os() << block_analysis_->getFlatInputName(read);
    os() << ")" << std::endl;
  }
}

void BlockPrinter::visit(const Load* v) {
  os() << block_analysis_->getFlatInputName(v->buf()) << ".buffer, ";
}
void BlockPrinter::visit(const Store* v) {
  emitIndent();
  os() << *v->value() << block_analysis_->getFlatInputName(v->buf())
       << ".tensor)" << std::endl;
}

void BlockPrinter::visit(const Block* v) {
  os() << "{" << std::endl;
  indent_++;
  for (Stmt* s : v->stmts()) {
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
  return func_prefix + "_" + c10::to_string(value);
}

void BlockCodeGen::Initialize() {
  block_analysis_ = std::make_unique<BlockAnalysis>();
  printer_ = std::make_unique<BlockPrinter>(&oss_, block_analysis_.get());

  Stmt* stmt_v = stmt();
  stmt_v->accept(block_analysis_.get());

  auto buf_reads = block_analysis_->loads();
  auto buf_writes = block_analysis_->stores();
  // Ensure all Bufs in reads/writes are in the map
  std::unordered_set<const Buf*> bufs(buf_reads.begin(), buf_reads.end());
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

  USE_TRIGGER(block_codegen_created);
}

void BlockCodeGen::call(const std::vector<CallArg>& args) {
  throw std::runtime_error("BlockCodeGen: Cannot call Block code ");
}
void BlockCodeGen::call_raw(const std::vector<void*>& args) {
  throw std::runtime_error("BlockCodeGen: Cannot call Block code ");
}

BlockCodeGen::~BlockCodeGen() = default;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterCodeGen<BlockCodeGen> block_codegen_reg("block_codegen");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
