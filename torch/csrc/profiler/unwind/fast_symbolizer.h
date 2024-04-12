#pragma once
#include <torch/csrc/profiler/unwind/debug_info.h>
#include <torch/csrc/profiler/unwind/line_number_program.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>

namespace torch::unwind {

#define UNWIND_WARN(w, ...)                   \
  do {                                        \
    w.emplace_back(fmt::format(__VA_ARGS__)); \
    LOG_INFO("WARNING: {}\n", w.back());      \
  } while (0);

struct CompilationUnit {
  CompilationUnit(
      Sections& s,
      uint64_t debug_info_offset,
      std::vector<std::string>& warnings)
      : s_(s),
        debug_info_offset_(debug_info_offset),
        debug_info_(s_),
        line_number_program_(s_) {
    try {
      debug_info_.parse(debug_info_offset_);
    } catch (UnwindError& err) {
      UNWIND_WARN(
          warnings,
          "failed to read debug_info [{:x}] {}",
          debug_info_offset_,
          err.what());
    }
    try {
      auto lnp = debug_info_.lineNumberProgramOffset();
      if (lnp) {
        line_number_program_.parse(*lnp);
      }
    } catch (UnwindError& err) {
      UNWIND_WARN(
          warnings,
          "failed to read line number program [{:x}] {}",
          debug_info_offset_,
          err.what());
    }
  }

  unwind::optional<std::string> findSubprogramName(uint64_t address) {
    return debug_info_.getSubprogramName(address);
  }
  unwind::optional<std::pair<std::string, int64_t>> findLine(uint64_t address) {
    if (auto e = line_number_program_.find(address)) {
      return std::make_pair(line_number_program_.filename(e->file), e->line);
    }
    return std::nullopt;
  }

 private:
  Sections& s_;
  uint64_t debug_info_offset_;
  DebugInfo debug_info_;
  LineNumberProgram line_number_program_;
};

struct FastSymbolizer {
  FastSymbolizer() {}
  Frame symbolize(const std::string& library, uint64_t offset) {
    LOG_INFO("symbolizing {} + 0x{:x}\n", library, offset);
    Frame frame;
    frame.filename = library;
    frame.lineno = offset;
    frame.funcname = "??";
    auto cu = getOrCreateCompilationUnit(library, offset);
    if (!cu) {
      auto s = getOrCreateSections(library);
      UNWIND_WARN(
          warnings_,
          "failed to find compilation unit for {} 0x{:x} ({} compilation units)",
          library,
          offset,
          s->compilationUnitCount());
      return frame;
    }
    if (auto e = cu->findSubprogramName(offset)) {
      frame.funcname = *e;
    } else {
      UNWIND_WARN(
          warnings_,
          "failed to find subprogram name for {} 0x{:x}",
          library,
          offset);
    }
    if (auto e = cu->findLine(offset)) {
      frame.filename = e->first;
      frame.lineno = e->second;
    } else {
      UNWIND_WARN(
          warnings_, "failed to find file/line for {} 0x{:x}", library, offset);
    }
    return frame;
  }
  const std::vector<std::string>& warnings() {
    return warnings_;
  }

 private:
  Sections* getOrCreateSections(const std::string& library) {
    auto it = libraries_.find(library);
    if (it == libraries_.end()) {
      it = libraries_.insert({library, std::make_unique<Sections>()}).first;
      try {
        it->second->parse(library.c_str());
        Sections* s = it->second.get();
        if (false) {
          // preload every compilation unit to test parsing logic
          for (auto& p : s->debugInfoOffsets()) {
            CompilationUnit cu(*s, p, warnings_);
          }
        }
      } catch (UnwindError& err) {
        UNWIND_WARN(warnings_, "failed to parse {}: {}", library, err.what());
      }
    }
    return it->second.get();
  }
  CompilationUnit* getOrCreateCompilationUnit(
      const std::string& library,
      uint64_t offset) {
    Sections* s = getOrCreateSections(library);
    auto dbg_offset = s->findDebugInfoOffset(offset);
    if (!dbg_offset) {
      LOG_INFO("no debug info offset for {} + 0x{:x}\n", library, offset);
      return nullptr;
    }
    uint64_t dbg_info_addr = ptrdiff_t(s->debug_info.data + *dbg_offset);
    auto it = compilation_units.find(dbg_info_addr);
    if (it == compilation_units.end()) {
      it = compilation_units
               .insert(
                   {dbg_info_addr,
                    std::make_unique<CompilationUnit>(
                        *s, *dbg_offset, warnings_)})
               .first;
    }
    return it->second.get();
  }
  std::unordered_map<std::string, std::unique_ptr<Sections>> libraries_;
  std::unordered_map<ptrdiff_t, std::unique_ptr<CompilationUnit>>
      compilation_units;
  std::vector<std::string> warnings_;
};

} // namespace torch::unwind
