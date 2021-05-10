#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

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
  bool is_buf_store_target(const Buf* buf) const {
    return store_targets_.count(buf) > 0;
  }

  const std::unordered_set<const Buf*>& loads() const {
    return loads_;
  }

  const std::unordered_set<const Buf*>& stores() const {
    return store_targets_;
  }

  int block_size() const {
    return block_size_;
  }

  bool areBufsInMap(const std::unordered_set<const Buf*>& bufs) const;

  const Buf* getMultiDimBuf(const Buf* buf) const;

  std::string getInputName(const Buf* buf) const;

  std::string getFlatInputName(const Buf* buf) const {
    return getInputName(buf) + "_flat";
  }

  std::unordered_map<std::string, const Buf*> getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  void visit(const Store* v) override;
  void visit(const Load* v) override;
  void visit(const For* v) override;

  std::unordered_map<std::string, const Buf*> map_input_to_tensor_bufs_;
  std::unordered_set<const Buf*> store_targets_;
  std::unordered_set<const Buf*> loads_;
  int block_size_ = 32;
};

// A class that overrides the underlying IRPrinter to produce Block.
class BlockPrinter : public IRPrinter {
 public:
  BlockPrinter(std::ostream* os, const BlockAnalysis* block_analysis)
      : IRPrinter(*os), block_analysis_(block_analysis) {}

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  const BlockAnalysis* block_analysis_;
  std::unordered_map<std::string, int> dim_values_map;
  std::vector<std::string> dim_names = {"N", "H", "W", "C"};
  std::vector<std::string> flat_dim_names = {"N", "NH", "NHW", "NHWC"};
  void PrintTensorInfo(const std::unordered_set<const Buf*>& bufs);
  void PrintArguments(const std::unordered_set<const Buf*>& bufs);
  void PrintBufferInfo(const std::unordered_set<const Buf*>& bufs);
  void PrintDistribution(const std::unordered_set<const Buf*>& bufs);
  void PrintLoop(
      const std::unordered_set<const Buf*>& bufs,
      bool block_idx = true);
  void PrintReshapeInfo(
      const std::unordered_set<const Buf*>& bufs,
      bool reverse = false);
  void PrintDMAs(const std::unordered_set<const Buf*>& bufs);
  void PrintAdjustBuffers(const std::unordered_set<const Buf*>& bufs);

  void visit(const For* v) override;
  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const Block* v) override;
  void visit(const Add* v) override;
  void visit(const Mul* v) override;
};

class TORCH_API BlockCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  /* implicit */
  BlockCodeGen(Stmt* stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCPU)) {
    Initialize();
  }

  BlockCodeGen(
      Stmt* stmt,
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
