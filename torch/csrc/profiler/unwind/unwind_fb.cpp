#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__)) && \
    defined(__has_include) &&                                              \
    __has_include("ext/stdio_filebuf.h") && defined(FBCODE_CAFFE2)
#include <c10/util/flat_hash_map.h>
#include <llvm/DebugInfo/Symbolize/Symbolize.h>
#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch::unwind {

std::vector<Frame> symbolize(const std::vector<void*>& frames, Mode mode) {
  static std::mutex symbolize_mutex;
  static llvm::symbolize::LLVMSymbolizer symbolizer;
  static ska::flat_hash_map<void*, Frame> frame_map_;

  std::lock_guard<std::mutex> guard(symbolize_mutex);
  std::vector<Frame> results;
  results.reserve(frames.size());
  for (auto addr : frames) {
    if (!frame_map_.count(addr)) {
      auto frame = Frame{"??", "<unwind unsupported>", 0};
      auto maybe_library = libraryFor(addr);
      if (maybe_library) {
        auto libaddress = maybe_library->second - 1;
        auto r = symbolizer.symbolizeCode(
            maybe_library->first,
            {libaddress, llvm::object::SectionedAddress::UndefSection});
        if (r) {
          frame.filename = r->FileName;
          frame.funcname = r->FunctionName;
          frame.lineno = r->Line;
        }
      }
      frame_map_[addr] = std::move(frame);
    }
    results.emplace_back(frame_map_[addr]);
  }
  return results;
}

} // namespace torch::unwind

#endif
