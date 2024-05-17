#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>

namespace torch {
namespace jit {

struct TORCH_API CanonicalizedSymbolicShape {
  // TODO: Consider in the future if it is reasonable to
  // merge code with SymbolicShape or VaryingShape while keeping
  // the two not implicitly convertable (and cause bugs).
  CanonicalizedSymbolicShape(
      const c10::SymbolicShape& orig_shape,
      std::unordered_map<int64_t, int64_t>& ss_map) {
    init(orig_shape, ss_map);
  }

  CanonicalizedSymbolicShape(c10::SymbolicShape& orig_shape) {
    std::unordered_map<int64_t, int64_t> new_ssmap;
    init(orig_shape, new_ssmap);
  }

  size_t hash() const;

  c10::SymbolicShape toSymbolicShape(
      std::unordered_map<int64_t, int64_t>& inverse_ss_map) const;

  TORCH_API friend bool operator==(
      const CanonicalizedSymbolicShape& a,
      const CanonicalizedSymbolicShape& b);

 private:
  std::optional<std::vector<int64_t>> values_;

  void init(
      const c10::SymbolicShape& orig_shape,
      std::unordered_map<int64_t, int64_t>& ss_map);
};

// SHAPE CACHE API
TORCH_API std::optional<std::vector<at::SymbolicShape>>
get_cached_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec);

TORCH_API void cache_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    const std::vector<at::SymbolicShape>& ret_vec);

// For use in test code
TORCH_API void clear_shape_cache();
TORCH_API size_t get_shape_cache_size();

} // namespace jit
} // namespace torch
