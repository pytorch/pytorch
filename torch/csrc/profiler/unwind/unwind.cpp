#include <c10/util/Exception.h>
#include <torch/csrc/profiler/unwind/unwind.h>

#if !defined(__linux__) || !defined(__x86_64__) || !defined(__has_include) || \
    !__has_include("ext/stdio_filebuf.h")
namespace torch {
namespace unwind {
std::vector<void*> unwind() {
  TORCH_CHECK(
      false,
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

std::vector<Frame> symbolize(const std::vector<void*>& frames) {
  TORCH_CHECK(
      false,
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

Stats stats() {
  TORCH_CHECK(
      false,
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
}

} // namespace unwind
} // namespace torch

#else
#include <c10/util/flat_hash_map.h>
#include <elf.h>
#include <limits.h>
#include <link.h>
#include <linux/limits.h>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <c10/util/irange.h>
#include <torch/csrc/profiler/unwind/communicate.h>
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/eh_frame_hdr.h>
#include <torch/csrc/profiler/unwind/fde.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/unwinder.h>
#include <shared_mutex>

struct UpgradeExclusive {
  UpgradeExclusive(std::shared_lock<std::shared_timed_mutex>& rdlock)
      : rdlock_(rdlock) {
    rdlock_.unlock();
    rdlock_.mutex()->lock();
  }
  ~UpgradeExclusive() {
    rdlock_.mutex()->unlock();
    rdlock_.lock();
  }

 private:
  std::shared_lock<std::shared_timed_mutex>& rdlock_;
};

struct LibraryInfo {
  LibraryInfo(
      std::string name,
      uint64_t load_bias,
      uint64_t last_addr,
      void* eh_frame_hdr_ptr_)
      : name_(std::move(name)),
        load_bias_(load_bias),
        last_addr_(last_addr),
        eh_frame_hdr_(eh_frame_hdr_ptr_) {}

  uint64_t load_bias() const {
    return load_bias_;
  }
  uint64_t last_addr() const {
    return last_addr_;
  }
  Unwinder unwinderFor(uint64_t addr) const {
    void* fde_data = eh_frame_hdr_.entryForAddr(addr);
    FDE fde(fde_data, name().c_str(), load_bias());
    TableState state = fde.readUpTo(addr);
    return Unwinder(state.cfa, state.registers[D_RIP], state.registers[D_RBP]);
  }
  const std::string& name() const {
    return name_;
  }

 private:
  std::string name_;
  uint64_t load_bias_; // addr >= load_bias_
  uint64_t last_addr_; // addr < last_addr_
  EHFrameHdr eh_frame_hdr_;
};

const char* process_name() {
  static char name[PATH_MAX + 1] = "";
  if (*name == '\0') {
    ssize_t len = readlink("/proc/self/exe", name, PATH_MAX);
    TORCH_INTERNAL_ASSERT(len != -1, "can't get path to exe")
    name[len] = '\0';
  }
  return name;
}

struct Version {
  uint64_t adds_ = LONG_LONG_MAX;
  uint64_t subs_ = LONG_LONG_MAX;
};

struct UnwindCache {
  Version currentVersion() {
    Version r;
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t size, void* data) {
          Version* v = (Version*)data;
          v->adds_ = info->dlpi_adds;
          v->subs_ = info->dlpi_subs;
          return 1;
        },
        &r);
    return r;
  }
  void refreshLibraries() {
    ++stats_.resets;
    all_libraries_.clear();
    ip_cache_.clear();
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t size, void* data) {
          auto self = (UnwindCache*)data;
          uint64_t last_addr = 0;
          auto segments = (Elf64_Phdr*)info->dlpi_phdr;
          for (auto i : c10::irange(info->dlpi_phnum)) {
            if (segments[i].p_type == PT_LOAD) {
              auto begin = ((uint64_t)info->dlpi_addr + segments[i].p_vaddr);
              auto end = (begin + segments[i].p_memsz);
              last_addr = std::max(end, last_addr);
            }
            if (segments[i].p_type == PT_GNU_EH_FRAME) {
              std::string library_name = info->dlpi_name;
              if (library_name == "") {
                library_name = process_name();
              }
              auto eh_frame_hdr =
                  (void*)(segments[i].p_vaddr + info->dlpi_addr);
              self->all_libraries_.emplace_back(
                  std::move(library_name),
                  info->dlpi_addr,
                  last_addr,
                  eh_frame_hdr);
              return 0;
            }
          }
          TORCH_WARN_ONCE(
              "Did not find a PT_GNU_EH_FRAME segment for ", info->dlpi_name);
          return 0;
        },
        this);
    std::sort(
        all_libraries_.begin(),
        all_libraries_.end(),
        [](const LibraryInfo& lhs, const LibraryInfo& rhs) {
          return lhs.load_bias() < rhs.load_bias();
        });
  }
  void checkRefresh(std::shared_lock<std::shared_timed_mutex>& rdlock) {
    Version current_version = currentVersion();
    if (current_version.subs_ != last_version_.subs_) {
      UpgradeExclusive lock(rdlock);
      refreshLibraries();
    }
  }

  const Unwinder& unwinderFor(
      uint64_t addr,
      std::shared_lock<std::shared_timed_mutex>& rdlock) {
    auto it = ip_cache_.find(addr);
    if (it != ip_cache_.end()) {
      ++stats_.hits;
      return it->second;
    }

    // we are about to modify the cache
    UpgradeExclusive lock(rdlock);
    ++stats_.misses;

    Unwinder unwinder = Unwinder::unknown();
    try {
      unwinder = libraryFor(addr).unwinderFor(addr);
    } catch (UnwindError& err) {
      // because unwinders are cached this will only print
      // once per frame that cannot be unwound.
      TORCH_WARN("Unsupported unwinding pattern: ", err.what());
    }
    auto r = ip_cache_.insert_or_assign(addr, std::move(unwinder));
    return r.first->second;
  }

  const LibraryInfo& libraryFor(uint64_t addr) {
    Version current_version = currentVersion();
    if (current_version.subs_ != last_version_.subs_) {
      refreshLibraries();
      last_version_ = current_version;
    }
    auto& r = searchFor(addr);
    if (addr >= r.last_addr()) {
      if (current_version.adds_ != last_version_.adds_) {
        refreshLibraries();
        last_version_ = current_version;
      }
      auto& r = searchFor(addr);
      if (addr >= r.last_addr()) {
        throw UnwindError("addr not in range of known libraries");
      }
      return r;
    }
    return r;
  }
  torch::unwind::Stats stats() {
    return stats_;
  }

 private:
  const LibraryInfo& searchFor(uint64_t addr) {
    uint64_t low = 0;
    uint64_t high = all_libraries_.size();
    while (low + 1 != high) {
      auto mid = (low + high) / 2;
      if (addr < all_libraries_.at(mid).load_bias()) {
        high = mid;
      } else {
        low = mid;
      }
    }
    return all_libraries_.at(low);
  }

  // sorted by load_bias
  std::vector<LibraryInfo> all_libraries_;
  ska::flat_hash_map<uint64_t, Unwinder> ip_cache_;

  torch::unwind::Stats stats_;

  // to keep track of whether we need to refresh this info
  Version last_version_;
};

static UnwindCache unwind_cache;
static std::shared_timed_mutex cache_mutex_;

extern "C" void unwind_c(std::vector<void*>* result, int64_t rsp, int64_t rbp) {
  std::shared_lock lock(cache_mutex_);
  UnwindState state;
  state.rip = *(int64_t*)(rsp);
  // +8 because we saved rsp after the return address was already pushed
  // to the stack
  state.rsp = rsp + 8;
  state.rbp = rbp;
  unwind_cache.checkRefresh(lock);
  while (true) { // unwind for _start sets rip as being undefined
    result->push_back((void*)state.rip);
    const Unwinder& uw = unwind_cache.unwinderFor(state.rip, lock);
    if (uw.terminator()) {
      if (uw.isUnknown()) {
        result->push_back(nullptr);
      }
      break;
    }
    state = uw.run(state);
  }
}

extern "C" void unwind_entry(std::vector<void*>* result);

// calling convention puts the first three pointer/int64_t arguments in
// rdi rsi rdx (all caller-saved)
// rdi already holds the pointer to the result vector
// we add arguments for current rsp and rbp and then tail call
// into unwind_c
__asm__(
    ".global unwind_entry\n"
    "unwind_entry:\n"
    "mov %rsp, %rsi;\n"
    "mov %rbp, %rdx;\n"
    "jmp unwind_c;\n");

namespace torch {
namespace unwind {
std::vector<void*> unwind() {
  std::vector<void*> frames;
  unwind_entry(&frames);
  return frames;
}

#ifdef FBCODE_CAFFE2
// in CUDA binaries, we have to use the internal symbolizer because
// addr2line seems to hang.
__attribute__((weak))
#endif
std::vector<Frame>
symbolize(const std::vector<void*>& frames) {
  // we need to make sure we don't write more than 64k bytes to
  // a pipe before reading the results. Otherwise the buffer may
  // get filled and block before we read the results.
  // Each line is < 32 characters,
  // so this limits us to < 32k bytes before we read rules.
  constexpr int BLOCK = 1024;
  struct Entry {
    const LibraryInfo* lib;
    std::unique_ptr<Communicate> comm;
    std::vector<void*> queried;
    size_t completed = 0;
  };
  std::vector<Entry> entries;
  auto get_or_create = [&](const LibraryInfo& info) -> Entry& {
    for (auto& e : entries) {
      if (e.lib->load_bias() == info.load_bias()) {
        return e;
      }
    }
    const char* args[] = {
        "addr2line", "-C", "-f", "-e", info.name().c_str(), nullptr};
    entries.emplace_back(
        Entry{&info, std::make_unique<Communicate>("addr2line", args), {}});
    return entries.back();
  };

  std::unordered_map<void*, Frame> results_map;

  auto read_pending_results = [&](Entry& e) {
    size_t N = e.queried.size();
    for (; e.completed < N; ++e.completed) {
      Frame frame;
      std::getline(e.comm->in(), frame.funcname);
      std::string filename_lineno;
      std::getline(e.comm->in(), filename_lineno);
      auto colon = filename_lineno.find_last_of(':');
      frame.filename = filename_lineno.substr(0, colon);
      std::string lineno_str = filename_lineno.substr(colon + 1);
      frame.lineno = lineno_str == "?" ? 0 : std::stoi(lineno_str);
      results_map[e.queried[e.completed]] = std::move(frame);
    }
  };
  for (auto f : frames) {
    if (f == nullptr) {
      continue;
    }
    auto& entry = get_or_create(unwind_cache.libraryFor((uint64_t)f));
    entry.queried.push_back(f);
    auto libaddress = ((uint64_t)f - entry.lib->load_bias() - 1);
    entry.comm->out() << (void*)libaddress << "\n";
    if (entry.queried.size() - entry.completed > BLOCK) {
      entry.comm->out().flush();
      read_pending_results(entry);
    }
  }

  for (auto& e : entries) {
    e.comm->out().flush();
  }

  for (auto& e : entries) {
    read_pending_results(e);
  }

  std::vector<Frame> results;
  for (auto f : frames) {
    if (f == nullptr) {
      results.emplace_back(Frame{"??", "<unwind unsupported>", 0});
      continue;
    }
    results.emplace_back(results_map.at(f));
  }
  return results;
}

Stats stats() {
  return unwind_cache.stats();
}

} // namespace unwind
} // namespace torch
#endif
