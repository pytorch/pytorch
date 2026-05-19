#pragma once

#include <string>
#include <vector>

// Shared struct for Triton kernel compilation results.
// Used by both cpp-wrapper JIT (filled at runtime via Python) and AOTInductor
// (filled at compile time from a generated config header).
struct LazyKernelCompileResult {
  std::string cubin_path;
  std::string mangled_name;
  int num_warps;
  int shared_mem;
  std::vector<int> xblocks;
  std::vector<int> yblocks;
  std::vector<int> zblocks;
  std::vector<int> r0blocks;
  std::vector<int> r1blocks;
  std::vector<int> r2blocks;
  int rsplit;
  int rsplit_size;
  int config_index;
  int global_scratch;
  int profile_scratch;
};
