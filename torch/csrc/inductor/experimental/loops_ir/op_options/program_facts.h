#pragma once
// Read-only facts over a KernelProgram used by the op-option passes. Uses
// only inline KernelProgram/Compute/ExternCall getters (no kernel_ir.cpp
// link dependency). Precondition: the program passed core verification.

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "../kernel_ir.h"

namespace torch::inductor::loops_ir::op_options {

// IR name (buffer or view) resolved to its storage root. `offset` composes
// every view offset along the alias chain (same `rel` the backend wrapper
// applies as root.storage_offset() + rel).
struct RootRef {
  std::string root;
  int64_t offset = 0;
};

struct ProgramFacts {
  std::map<std::string, RootRef> roots; // every buffer and view name
  std::map<std::string, int64_t> root_elements;
  std::map<std::string, std::string> root_dtype; // core dtype spelling
  std::map<std::string, size_t> producer; // root -> node index (absent: program input)
  // root -> node indices reading it directly or through any view, ascending.
  std::map<std::string, std::vector<size_t>> users;
  std::set<std::string> escaping_roots; // referenced by a program output

  const RootRef& resolve(const std::string& name) const;
  bool isKernelFunction(const KernelProgram& p, size_t node) const;
};

ProgramFacts analyzeProgram(const KernelProgram& program);

} // namespace torch::inductor::loops_ir::op_options
