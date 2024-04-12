#pragma once
#include <debug_info.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <line_number_program.h>
#include <torch/csrc/profiler/unwind/sections.h>

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
      debug_info_.parse(debug_info_offset);
    } catch (UnwindError& err) {
      warnings.emplace_back(fmt::format(
          "failed to read debug_info [{:x}] {}",
          debug_info_offset,
          err.what()));
    }
    try {
      auto lnp = debug_info_.lineNumberProgramOffset();
      if (lnp) {
        line_number_program_.parse(*lnp);
      }
    } catch (UnwindError& err) {
      warnings.emplace_back(fmt::format(
          "failed to read line number program [{:x}] {}",
          debug_info_offset,
          err.what()));
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

struct Frame {
  std::string filename = "??";
  uint64_t line = 0;
  std::string function = "??";
};

struct Symbolizer {
  Symbolizer() {}
  Frame symbolize(const std::string& library, uint64_t offset) {
    LOG_INFO("symbolizing {} + 0x{:x}\n", library, offset);
    Frame frame;
    frame.filename = library;
    frame.line = offset;
    auto cu = getOrCreateCompilationUnit(library, offset);
    if (!cu) {
      return frame;
    }
    if (auto e = cu->findSubprogramName(offset)) {
      frame.function = *e;
    }
    if (auto e = cu->findLine(offset)) {
      frame.filename = e->first;
      frame.line = e->second;
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
      it = libraries_
               .insert({library, std::make_unique<Sections>(library.c_str())})
               .first;
      try {
        it->second->parse();
      } catch (UnwindError& err) {
        warnings_.emplace_back(
            fmt::format("failed to parse {}: {}", library, err.what()));
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
