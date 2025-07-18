#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <torch/csrc/profiler/unwind/unwind.h>
#include <torch/csrc/utils/cpp_stacktraces.h>

#if !defined(__linux__) || !defined(__x86_64__) || !defined(__has_include) || \
    !__has_include("ext/stdio_filebuf.h")
namespace torch::unwind {
std::vector<void*> unwind() {
  TORCH_WARN_ONCE(
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
  return {};
}

std::optional<std::pair<std::string, uint64_t>> libraryFor(void* addr) {
  TORCH_WARN_ONCE(
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
  return {};
}

#ifndef FBCODE_CAFFE2
std::vector<Frame> symbolize(const std::vector<void*>& frames, Mode mode) {
  TORCH_WARN_ONCE(
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
  return {};
}
#endif

Stats stats() {
  TORCH_WARN_ONCE(
      "record_context_cpp is not support on non-linux non-x86_64 platforms");
  return {};
}

} // namespace torch::unwind

#else

#include <c10/util/flat_hash_map.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <linux/limits.h>
#include <algorithm>
#include <climits>
#include <vector>

#include <c10/util/irange.h>
#include <cxxabi.h>
#include <torch/csrc/profiler/unwind/communicate.h>
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/eh_frame_hdr.h>
#include <torch/csrc/profiler/unwind/fast_symbolizer.h>
#include <torch/csrc/profiler/unwind/fde.h>
#include <torch/csrc/profiler/unwind/unwinder.h>
#include <shared_mutex>

extern "C" void unwind_c(std::vector<void*>* result, int64_t rsp, int64_t rbp);
extern "C" void unwind_entry(std::vector<void*>* result);

namespace torch::unwind {
struct UpgradeExclusive {
  UpgradeExclusive(std::shared_lock<std::shared_timed_mutex>& rdlock)
      : rdlock_(rdlock) {
    rdlock_.unlock();
    rdlock_.mutex()->lock();
  }
  UpgradeExclusive(const UpgradeExclusive&) = delete;
  UpgradeExclusive(UpgradeExclusive&&) = delete;
  UpgradeExclusive& operator=(const UpgradeExclusive&) = delete;
  UpgradeExclusive& operator=(UpgradeExclusive&&) = delete;
  ~UpgradeExclusive() {
    rdlock_.mutex()->unlock();
    rdlock_.lock();
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
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

static const char* process_name() {
  // NOLINTNEXTLINE(*-c-arrays*)
  static char name[PATH_MAX + 1] = "";
  if (*name == '\0') {
    ssize_t len = readlink("/proc/self/exe", name, PATH_MAX);
    TORCH_INTERNAL_ASSERT(len != -1, "can't get path to exe")
    name[len] = '\0';
  }
  return name;
}

struct Version {
  uint64_t adds_ = LLONG_MAX;
  uint64_t subs_ = LLONG_MAX;
};

struct UnwindCache {
  Version currentVersion() {
    Version r;
    dl_iterate_phdr(
        [](struct dl_phdr_info* info,
           size_t size [[maybe_unused]],
           void* data) {
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
        [](struct dl_phdr_info* info,
           size_t size [[maybe_unused]],
           void* data) {
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
              if (library_name.empty()) {
                library_name = process_name();
              }
              auto eh_frame_hdr =
                  // NOLINTNEXTLINE(performance-no-int-to-ptr)
                  (void*)(segments[i].p_vaddr + info->dlpi_addr);
              self->all_libraries_.emplace_back(
                  std::move(library_name),
                  info->dlpi_addr,
                  last_addr,
                  eh_frame_hdr);
              return 0;
            }
          }
          self->libraries_with_no_unwind_.emplace_back(info->dlpi_name);
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
    } catch (unwind::UnwindError& err) {
      // because unwinders are cached this will only print
      // once per frame that cannot be unwound.
      TORCH_WARN("Unsupported unwinding pattern: ", err.what());
    }
    auto r = ip_cache_.insert_or_assign(addr, unwinder);
    return r.first->second;
  }

  const LibraryInfo* findLibraryFor(uint64_t addr) {
    Version current_version = currentVersion();
    if (current_version.subs_ != last_version_.subs_) {
      refreshLibraries();
      last_version_ = current_version;
    }
    auto* r = searchFor(addr);
    if (!r) {
      if (current_version.adds_ != last_version_.adds_) {
        refreshLibraries();
        last_version_ = current_version;
      }
      r = searchFor(addr);
    }
    return r;
  }

  const LibraryInfo& libraryFor(uint64_t addr) {
    auto* r = findLibraryFor(addr);
    if (!r) {
      for ([[maybe_unused]] const auto& l : libraries_with_no_unwind_) {
        TORCH_WARN("Did not find a PT_GNU_EH_FRAME segment for ", l);
      }
      libraries_with_no_unwind_.clear();
      throw UnwindError("addr not in range of known libraries");
    }
    return *r;
  }

  torch::unwind::Stats stats() {
    return stats_;
  }

 private:
  const LibraryInfo* searchFor(uint64_t addr) {
    if (all_libraries_.empty()) {
      return nullptr;
    }
    uint64_t low = 0;
    uint64_t high = all_libraries_.size();
    while (low + 1 < high) {
      auto mid = (low + high) / 2;
      if (addr < all_libraries_.at(mid).load_bias()) {
        high = mid;
      } else {
        low = mid;
      }
    }
    LibraryInfo* r = &all_libraries_.at(low);
    if (addr < r->load_bias() || addr >= r->last_addr()) {
      return nullptr;
    }
    return r;
  }

  // sorted by load_bias
  std::vector<LibraryInfo> all_libraries_;
  ska::flat_hash_map<uint64_t, Unwinder> ip_cache_;

  torch::unwind::Stats stats_;

  // to keep track of whether we need to refresh this info
  Version last_version_;

  std::vector<std::string> libraries_with_no_unwind_;
};

static UnwindCache unwind_cache;
static std::shared_timed_mutex cache_mutex_;

std::vector<void*> unwind() {
  std::vector<void*> frames;
  unwind_entry(&frames);
  return frames;
}

std::optional<std::pair<std::string, uint64_t>> libraryFor(void* addr) {
  if (!addr) {
    return std::nullopt;
  }
  std::shared_lock lock(cache_mutex_);
  const LibraryInfo* library_info = unwind_cache.findLibraryFor((uint64_t)addr);
  if (!library_info) {
    return std::nullopt;
  }
  return std::make_pair(
      library_info->name(), (uint64_t)addr - library_info->load_bias());
}

static std::string dladdr_lookup(void* addr) {
  Dl_info dlinfo;
  std::string funcname = "??";
  if (dladdr(addr, &dlinfo) && dlinfo.dli_sname) {
    funcname = demangle(dlinfo.dli_sname);
  }
  return funcname;
}

struct Symbolizer {
  Symbolizer() {
    auto envar = c10::utils::get_env("TORCH_ADDR2LINE_BINARY");
    if (envar.has_value()) {
      // currently we take user's input as is without checking
      addr2line_binary_ = std::move(envar.value());
      TORCH_WARN("Use custom addr2line binary: ", addr2line_binary_);
    } else {
      addr2line_binary_ = "addr2line"; // default
    }
  }
  static std::lock_guard<std::mutex> guard() {
    static std::mutex mutex;
    return std::lock_guard<std::mutex>(mutex);
  }
  static Symbolizer& get() {
    static Symbolizer singleton;
    return singleton;
  }

  void request(void* addr) {
    if (frame_map_.count(addr)) {
      return;
    }
    auto maybe_library = libraryFor(addr);
    if (!maybe_library) {
      frame_map_[addr] = Frame{"??", "<unwind unsupported>", 0};
      return;
    }
    has_pending_results_ = true;
    auto& entry = getOrCreate(maybe_library->first);
    entry.queried.push_back(addr);
    auto libaddress = maybe_library->second - 1;
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    entry.comm->out() << (void*)libaddress << "\n";
    // we need to make sure we don't write more than 64k bytes to
    // a pipe before reading the results. Otherwise the buffer may
    // get filled and block before we read the results.
    // Each line is < 32 characters,
    // so this limits us to < 32k bytes before we read rules.
    if (entry.queried.size() - entry.completed > BLOCK) {
      entry.comm->out().flush();
      readPendingResults(entry);
    }
  }
  const Frame& lookup(void* addr) {
    if (has_pending_results_) {
      for (auto& kv : entries_) {
        kv.second.comm->out().flush();
      }
      for (auto& kv : entries_) {
        readPendingResults(kv.second);
      }
      has_pending_results_ = false;
    }
    return frame_map_.at(addr);
  }

 private:
  static constexpr int BLOCK = 1024;
  std::string addr2line_binary_;
  struct Entry {
    std::unique_ptr<Communicate> comm;
    std::vector<void*> queried;
    size_t completed = 0;
  };
  ska::flat_hash_map<std::string, Entry> entries_;
  ska::flat_hash_map<void*, Frame> frame_map_;
  bool has_pending_results_ = true;

  Entry& getOrCreate(const std::string& name) {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
      // NOLINTNEXTLINE(*-c-arrays*)
      const char* args[] = {
          addr2line_binary_.c_str(), "-C", "-f", "-e", name.c_str(), nullptr};
      it = entries_
               .insert_or_assign(
                   name,
                   Entry{
                       std::make_unique<Communicate>(
                           addr2line_binary_.c_str(), args),
                       {}})
               .first;
    }
    return it->second;
  }
  void readPendingResults(Entry& e) {
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
      frame_map_[e.queried[e.completed]] = std::move(frame);
    }
  }
};

static std::vector<Frame> symbolize_fast(
    const std::vector<void*>& frames,
    Mode mode) {
  static std::mutex cache_mutex;
  static std::array<ska::flat_hash_map<void*, Frame>, 2> frame_maps;
  auto& frame_map = frame_maps[mode == Mode::fast ? 0 : 1];

  std::vector<uint32_t> indices_to_lookup;
  std::vector<Frame> results;
  results.reserve(frames.size());
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto i : c10::irange(frames.size())) {
      void* f = frames.at(i);
      auto it = frame_map.find(f);
      if (it == frame_map.end()) {
        indices_to_lookup.push_back(i);
        results.emplace_back(Frame{"??", "??", 0});
      } else {
        results.emplace_back(it->second);
      }
    }
  }
  if (!indices_to_lookup.empty()) {
    // do symbolizer work
    FastSymbolizer symbolizer;
    for (auto i : indices_to_lookup) {
      void* addr = frames.at(i);
      Frame& f = results.at(i);
      auto library = libraryFor(frames.at(i));
      if (library) {
        if (mode == Mode::fast) {
          f = symbolizer.symbolize(library->first, library->second - 1);
        } else {
          f = Frame{library->first, "??", library->second - 1};
        }
      }
      if (f.funcname == "??") {
        f.funcname = dladdr_lookup(addr);
      }
    }
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto i : indices_to_lookup) {
      frame_map.emplace(frames.at(i), results.at(i));
    }
  }
  return results;
}

static std::vector<Frame> symbolize_addr2line(
    const std::vector<void*>& frames) {
  auto guard = Symbolizer::guard();
  Symbolizer& s = Symbolizer::get();
  for (auto f : frames) {
    s.request(f);
  }
  std::vector<Frame> results;
  results.reserve(frames.size());
  for (auto f : frames) {
    results.emplace_back(s.lookup(f));
  }
  return results;
}

// fbcode will use llvm symbolize since there is an llvm dependency already
#ifndef FBCODE_CAFFE2
std::vector<Frame> symbolize(const std::vector<void*>& frames, Mode mode) {
  if (mode == Mode::addr2line) {
    return symbolize_addr2line(frames);
  } else {
    return symbolize_fast(frames, mode);
  }
}
#endif

Stats stats() {
  return unwind_cache.stats();
}

} // namespace torch::unwind

extern "C" C10_USED void unwind_c(
    std::vector<void*>* result,
    int64_t rsp,
    int64_t rbp) {
  std::shared_lock lock(torch::unwind::cache_mutex_);
  torch::unwind::UnwindState state{};
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  state.rip = *(int64_t*)(rsp);
  // +8 because we saved rsp after the return address was already pushed
  // to the stack
  state.rsp = rsp + 8;
  state.rbp = rbp;
  torch::unwind::unwind_cache.checkRefresh(lock);
  while (true) { // unwind for _start sets rip as being undefined
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    result->push_back((void*)state.rip);
    const torch::unwind::Unwinder& uw =
        torch::unwind::unwind_cache.unwinderFor(state.rip, lock);
    if (uw.terminator()) {
      if (uw.isUnknown()) {
        result->push_back(nullptr);
      }
      break;
    }
    state = uw.run(state);
  }
}

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

#endif
