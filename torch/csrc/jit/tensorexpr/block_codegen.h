#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <ATen/ATen.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// A class that analyzes the given program relevant for Block backend.
class BlockAnalysis : public IRVisitor {
 public:
  bool is_buf_store_target(BufPtr buf) const {
    return store_targets_.count(buf) > 0;
  }

  const std::unordered_set<BufPtr>& loads() const {
    return loads_;
  }

  const std::unordered_set<BufPtr>& stores() const {
    return store_targets_;
  }

  int block_size() const {
    return block_size_;
  }

  bool areBufsInMap(const std::unordered_set<BufPtr>& bufs) const;

  BufPtr getMultiDimBuf(BufPtr buf) const;

  std::string getInputName(BufPtr buf) const;

  std::string getFlatInputName(BufPtr buf) const {
    return getInputName(std::move(buf)) + "_flat";
  }

  std::unordered_map<std::string, BufPtr> getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  void visit(StorePtr v) override;
  void visit(LoadPtr v) override;
  void visit(ForPtr v) override;

  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;
  std::unordered_set<BufPtr> store_targets_;
  std::unordered_set<BufPtr> loads_;
  int block_size_ = 32;
};

// A class that overrides the underlying IRPrinter to produce Block.
class BlockPrinter : public IRPrinter {
 public:
  BlockPrinter(std::ostream* os, BlockAnalysis* block_analysis)
      : IRPrinter(*os), block_analysis_(block_analysis) {}

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  BlockAnalysis* block_analysis_;
  std::unordered_map<std::string, int> dim_values_map;
  std::vector<std::string> dim_names = {"N", "H", "W", "C"};
  std::vector<std::string> flat_dim_names = {"N", "NH", "NHW", "NHWC"};
  void PrintTensorInfo(const std::unordered_set<BufPtr>& bufs);
  void PrintArguments(const std::unordered_set<BufPtr>& bufs);
  void PrintBufferInfo(const std::unordered_set<BufPtr>& bufs);
  void PrintDistribution(const std::unordered_set<BufPtr>& bufs);
  void PrintLoop(const std::unordered_set<BufPtr>& bufs, bool block_idx = true);
  void PrintReshapeInfo(
      const std::unordered_set<BufPtr>& bufs,
      bool reverse = false);
  void PrintDMAs(const std::unordered_set<BufPtr>& bufs);
  void PrintAdjustBuffers(const std::unordered_set<BufPtr>& bufs);

  void visit(ForPtr v) override;
  void visit(LoadPtr v) override;
  void visit(StorePtr v) override;
  void visit(BlockPtr v) override;
  void visit(AddPtr v) override;
  void visit(MulPtr v) override;
};

class TORCH_API BlockCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  /* implicit */
  BlockCodeGen(StmtPtr stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCPU)) {
    Initialize();
  }

  BlockCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::Device(at::kCPU),
      const std::string& kernel_func_name = "func")
      : CodeGen(stmt, buffer_args, device, kernel_func_name) {
    Initialize();
  }

  ~BlockCodeGen() override;

  void call(const std::vector<CallArg>& args) override;
  void call_raw(const std::vector<void*>& args) override;

  void Initialize();

  std::string getCodeText(const std::string& attr = "") override {
    return oss_.str();
  }

 private:
  UniqueNameManager* name_manager() {
    if (!printer_) {
      throw std::runtime_error("Null IRPrinter is not expected");
    }
    return printer_->name_manager();
  }

  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_;
  std::unique_ptr<BlockPrinter> printer_;
  std::unique_ptr<BlockAnalysis> block_analysis_;

  std::string GetUniqueFuncName(const std::string& func_prefix);
};
} // namespace tensorexpr
} // namespace jit
} // namespace torch
