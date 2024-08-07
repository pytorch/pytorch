#pragma once

#include <fmt/format.h>
#include <sys/types.h>
#include <torch/csrc/profiler/unwind/debug_info.h>
#include <torch/csrc/profiler/unwind/line_number_program.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstddef>
#include <memory>

namespace torch::unwind {

#define UNWIND_WARN(w, ...)                   \
  do {                                        \
    w.emplace_back(fmt::format(__VA_ARGS__)); \
    LOG_INFO("WARNING: {}\n", w.back());      \
  } while (0);

struct FastSymbolizer {
  FastSymbolizer() = default;
  Frame symbolize(const std::string& library, uint64_t offset) {
    LOG_INFO("symbolizing {} + 0x{:x}\n", library, offset);
    Frame frame;
    frame.funcname = "??";
    frame.filename = library;
    frame.lineno = offset;
    auto s = getOrCreateSections(library);
    if (auto e = s->findSubprogramName(offset)) {
      frame.funcname = *e;
    } else {
      UNWIND_WARN(
          warnings_,
          "failed to find subprogram name for {} 0x{:x}",
          library,
          offset);
    }
    if (auto e = findLine(s, offset)) {
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
  void parseDebugInfo(Sections* s) {
    uint64_t offset = 0;
    while (offset < s->debug_info.size) {
      DebugInfo info(*s);
      info.parse(offset);
      if (auto lnp_offset = info.lineNumberProgramOffset()) {
        for (auto r : info.ranges()) {
          s->addDebugInfoRange(r.first, r.second, line_number_programs_.size());
        }
        line_number_programs_.emplace_back(
            std::make_unique<LineNumberProgram>(*s, *lnp_offset));
      }
      offset = info.nextOffset();
    }
  }
  Sections* getOrCreateSections(const std::string& library) {
    auto it = libraries_.find(library);
    if (it == libraries_.end()) {
      it = libraries_.insert({library, std::make_unique<Sections>()}).first;
      try {
        Sections* s = it->second.get();
        s->parse(library.c_str());
        parseDebugInfo(s);
      } catch (UnwindError& err) {
        UNWIND_WARN(
            warnings_, "failed to parse library {}: {}", library, err.what());
      }
    }
    return it->second.get();
  }
  std::optional<std::pair<std::string, int64_t>> findLine(
      Sections* s,
      uint64_t offset) {
    if (auto idx = s->findDebugInfoOffset(offset)) {
      auto r = line_number_programs_.at(*idx).get();
      try {
        r->parse();
      } catch (UnwindError& err) {
        UNWIND_WARN(
            warnings_,
            "failed to read line number program [{:x}] {}",
            r->offset(),
            err.what());
      }
      if (auto e = r->find(offset)) {
        return std::make_pair(r->filename(e->file), e->line);
      }
    }
    return std::nullopt;
  }
  std::unordered_map<std::string, std::unique_ptr<Sections>> libraries_;
  std::vector<std::unique_ptr<LineNumberProgram>> line_number_programs_;
  std::vector<std::string> warnings_;
};

} // namespace torch::unwind
