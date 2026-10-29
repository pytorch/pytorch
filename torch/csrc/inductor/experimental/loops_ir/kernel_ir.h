#pragma once
// KernelIR: kernel functions + program-level dependency DAG with
// generated-kernel launches and extern calls. Strict numerics 'strict_eager'.
// Vertical pointwise fusion only; no horizontal/reduction fusion, no fusion
// across extern barriers. Conservative: failures leave the program unfused.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "ir.h"

namespace torch::inductor::loops_ir {

// A kernel function owns logical loop regions. Initially exactly 1 region
// (future parallel-region support not implemented).
class KernelFunction {
 public:
  KernelFunction() = default;
  KernelFunction(std::string n, std::vector<Compute> regs, std::string numerics = "strict_eager")
      : name_(std::move(n)), regions_(std::move(regs)), numerics_(std::move(numerics)) {
    if (!isValidNumerics(numerics_)) {
      throw std::invalid_argument("KernelFunction: unsupported numerics '" + numerics_ +
                                  "' (allowed: 'strict_eager', 'inductor')");
    }
  }

  const std::string& name() const {
    return name_;
  }
  const std::vector<Compute>& regions() const {
    return regions_;
  }
  // Numerics policy ('strict_eager' default, 'inductor' alternative).
  const std::string& numerics() const {
    return numerics_;
  }

 private:
  std::string name_;
  std::vector<Compute> regions_;
  std::string numerics_ = "strict_eager";
};

class KernelProgram {
 public:
  using Node = std::variant<KernelFunction, ExternCall>;

  KernelProgram() = default;
  explicit KernelProgram(std::string n, std::string numerics = "strict_eager")
      : name_(std::move(n)), numerics_(std::move(numerics)) {
    if (!isValidNumerics(numerics_)) {
      throw std::invalid_argument("KernelProgram: unsupported numerics '" + numerics_ +
                                  "' (allowed: 'strict_eager', 'inductor')");
    }
  }

  // C++ camelCase getters.
  const std::string& name() const {
    return name_;
  }
  const std::vector<Buffer>& inputs() const {
    return inputs_;
  }
  const std::vector<Descriptor>& inputDescriptors() const {
    return input_descriptors_;
  }
  const std::vector<View>& views() const {
    return views_;
  }
  const std::vector<Descriptor>& outputs() const {
    return outputs_;
  }
  const std::vector<Node>& nodes() const {
    return nodes_;
  }
  const std::vector<std::vector<size_t>>& dependencies() const {
    return dependencies_;
  }
  // Numerics policy ('strict_eager' default, 'inductor' alternative).
  const std::string& numerics() const {
    return numerics_;
  }
  // Every buffer definition across inputs + node results.
  std::vector<Buffer> buffers() const;

  void addInput(const Buffer& b) {
    inputs_.push_back(b);
  }
  void addInputDescriptor(const Descriptor& d) {
    input_descriptors_.push_back(d);
  }
  void addView(const View& v) {
    views_.push_back(v);
  }
  void addOutput(const Descriptor& d) {
    outputs_.push_back(d);
  }
  void addFunction(const KernelFunction& f) {
    if (f.numerics() != numerics_) {
      throw std::invalid_argument("KernelProgram.addFunction: function numerics '" + f.numerics() +
                                  "' does not match program numerics '" + numerics_ + "'");
    }
    nodes_.emplace_back(f);
  }
  void addExternCall(const ExternCall& e) {
    nodes_.emplace_back(e);
  }

  // Full verification; throws std::invalid_argument with node context.
  void verify() const;
  // Verifies first, then returns human-readable text (never parsed).
  std::string dump() const;

  // Dependency DAG: deps[i] lists node indices that must complete before
  // node i. Rebuilt by buildDependencies().
  void buildDependencies();

 private:
  std::string name_;
  std::vector<Buffer> inputs_;
  std::vector<Descriptor> input_descriptors_;
  std::vector<View> views_;
  std::vector<Descriptor> outputs_;
  std::vector<Node> nodes_;
  std::vector<std::vector<size_t>> dependencies_;
  std::string numerics_ = "strict_eager";
};

// Lower a verified logical Program into a KernelProgram: pointwise-only
// Compute nodes become single-region KernelFunctions; reduction Compute
// nodes must already be eager ExternCalls (generated reductions rejected).
KernelProgram lower_to_kernels(const Program& program);

// Vertical pointwise fusion over the KernelProgram DAG. Fuses adjacent
// single-use pointwise producer/consumer chains only when iteration/access
// maps and output coverage are proven compatible. Preserves original output
// descriptor layouts and live/exported intermediate stores. Rejects fusion
// when the proof is missing. No horizontal or reduction fusion, no
// fusion across extern barriers.
KernelProgram fuse_kernels(const KernelProgram& program);

} // namespace torch::inductor::loops_ir
