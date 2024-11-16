#pragma once
#include <c10/macros/Export.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace torch::unwind {
// gather current stack, relatively fast.
// gets faster once the cache of program counter locations is warm.
TORCH_API std::vector<void*> unwind();

struct Frame {
  std::string filename;
  std::string funcname;
  uint64_t lineno;
};

enum class Mode { addr2line, fast, dladdr };

// note: symbolize is really slow
// it will launch an addr2line process that has to parse dwarf
// information from the libraries that frames point into.
// Callers should first batch up all the unique void* pointers
// across a number of unwind states and make a single call to
// symbolize.
TORCH_API std::vector<Frame> symbolize(
    const std::vector<void*>& frames,
    Mode mode);

// returns path to the library, and the offset of the addr inside the library
TORCH_API std::optional<std::pair<std::string, uint64_t>> libraryFor(
    void* addr);

struct Stats {
  size_t hits = 0;
  size_t misses = 0;
  size_t unsupported = 0;
  size_t resets = 0;
};
Stats stats();

} // namespace torch::unwind
