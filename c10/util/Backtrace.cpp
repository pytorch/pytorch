#include <c10/util/Backtrace.h>
#include <c10/util/Optional.h>
#include <c10/util/Type.h>
#include <c10/util/irange.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <c10/util/win32-headers.h>
#include <iomanip>
#pragma comment(lib, "Dbghelp.lib")
#endif

#if SUPPORTS_BACKTRACE
#include <cxxabi.h>
#ifdef C10_ANDROID
#include <dlfcn.h>
#include <unwind.h>
#else
#include <execinfo.h>
#endif
#endif

#ifdef FBCODE_CAFFE2
#include <common/process/StackTrace.h>
#endif

namespace c10 {

namespace {

#ifdef FBCODE_CAFFE2

// For some reason, the stacktrace implementation in fbcode is better than ours,
// see https://github.com/pytorch/pytorch/issues/56399 When it's available, just
// use that.
class GetBacktraceImpl {
 public:
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t frames_to_skip,
      size_t /* maximum_number_of_frames */,
      bool /* skip_python_frames */)
      : st_(/*skipFrames=*/frames_to_skip) {}

  std::string symbolize() const {
    return st_.toString();
  }

 private:
  facebook::process::StackTrace st_;
};

#elif SUPPORTS_BACKTRACE && defined(C10_ANDROID)

struct AndroidBacktraceState {
  std::vector<void*> buffer;
};

_Unwind_Reason_Code android_unwind_callback(
    struct _Unwind_Context* context,
    void* arg) {
  AndroidBacktraceState* state = (AndroidBacktraceState*)arg;
  uintptr_t pc = _Unwind_GetIP(context);
  if (pc) {
    state->buffer.emplace_back(reinterpret_cast<void*>(pc));
  }
  return _URC_NO_REASON;
}

class GetBacktraceImpl {
 public:
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t /* frames_to_skip */,
      size_t /* maximum_number_of_frames */,
      bool /* skip_python_frames */) {
    _Unwind_Backtrace(android_unwind_callback, &state_);
  }

  std::string symbolize() const {
    std::ostringstream os;
    int idx = 0;
    char* demangled = nullptr;
    size_t length = 0;

    for (const void* addr : state_.buffer) {
      const char* symbol = "";

      Dl_info info;
      if (dladdr(addr, &info) && info.dli_sname) {
        symbol = info.dli_sname;
      }

      int status = 0;
      demangled = __cxxabiv1::__cxa_demangle(
          /*mangled_name*/ symbol,
          /*output_buffer*/ demangled,
          /*length*/ &length,
          /*status*/ &status);

      os << " frame #" << idx++ << "\t"
         << ((demangled != NULL && status == 0) ? demangled : symbol) << "["
         << addr << "]\t" << std::endl;
    }
    free(demangled);
    return os.str();
  }

 private:
  AndroidBacktraceState state_;
};

#elif SUPPORTS_BACKTRACE // !defined(C10_ANDROID)

struct FrameInformation {
  /// If available, the demangled name of the function at this frame, else
  /// whatever (possibly mangled) name we got from `backtrace()`.
  std::string function_name;
  /// This is a number in hexadecimal form (e.g. "0xdead") representing the
  /// offset into the function's machine code at which the function's body
  /// starts, i.e. skipping the "prologue" that handles stack manipulation and
  /// other calling convention things.
  std::string offset_into_function;
  /// NOTE: In debugger parlance, the "object file" refers to the ELF file that
  /// the symbol originates from, i.e. either an executable or a library.
  std::string object_file;
};

bool is_python_frame(const FrameInformation& frame) {
  return frame.object_file == "python" || frame.object_file == "python3" ||
      (frame.object_file.find("libpython") != std::string::npos);
}

std::optional<FrameInformation> parse_frame_information(
    const std::string& frame_string) {
  FrameInformation frame;

  // This is the function name in the CXX ABI mangled format, e.g. something
  // like _Z1gv. Reference:
  // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
  std::string mangled_function_name;

#if defined(__GLIBCXX__)
  // In GLIBCXX, `frame_string` follows the pattern
  // `<object-file>(<mangled-function-name>+<offset-into-function>)
  // [<return-address>]`

  auto function_name_start = frame_string.find('(');
  if (function_name_start == std::string::npos) {
    return c10::nullopt;
  }
  function_name_start += 1;

  auto offset_start = frame_string.find('+', function_name_start);
  if (offset_start == std::string::npos) {
    return c10::nullopt;
  }
  offset_start += 1;

  const auto offset_end = frame_string.find(')', offset_start);
  if (offset_end == std::string::npos) {
    return c10::nullopt;
  }

  frame.object_file = frame_string.substr(0, function_name_start - 1);
  frame.offset_into_function =
      frame_string.substr(offset_start, offset_end - offset_start);

  // NOTE: We don't need to parse the return address because
  // we already have it from the call to `backtrace()`.

  mangled_function_name = frame_string.substr(
      function_name_start, (offset_start - 1) - function_name_start);
#elif defined(_LIBCPP_VERSION)
  // In LIBCXX, The pattern is
  // `<frame number> <object-file> <return-address> <mangled-function-name> +
  // <offset-into-function>`
  std::string skip;
  std::istringstream input_stream(frame_string);
  // operator>>() does not fail -- if the input stream is corrupted, the
  // strings will simply be empty.
  input_stream >> skip >> frame.object_file >> skip >> mangled_function_name >>
      skip >> frame.offset_into_function;
#else
#warning Unknown standard library, backtraces may have incomplete debug information
  return c10::nullopt;
#endif // defined(__GLIBCXX__)

  // Some system-level functions don't have sufficient debug information, so
  // we'll display them as "<unknown function>". They'll still have a return
  // address and other pieces of information.
  if (mangled_function_name.empty()) {
    frame.function_name = "<unknown function>";
    return frame;
  }

  frame.function_name = demangle(mangled_function_name.c_str());
  return frame;
}

class GetBacktraceImpl {
 public:
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t frames_to_skip,
      size_t maximum_number_of_frames,
      bool skip_python_frames)
      : skip_python_frames_(skip_python_frames),
        callstack_(frames_to_skip + maximum_number_of_frames, nullptr) {
    // We always skip this frame (backtrace).
    frames_to_skip += 1;

    // backtrace() gives us a list of return addresses in the current call
    // stack. NOTE: As per man (3) backtrace it can never fail
    // (http://man7.org/linux/man-pages/man3/backtrace.3.html).
    auto number_of_frames = static_cast<size_t>(
        ::backtrace(callstack_.data(), static_cast<int>(callstack_.size())));

    // Skip as many frames as requested.
    frames_to_skip = std::min(frames_to_skip, number_of_frames);
    number_of_frames -= frames_to_skip;
    callstack_.erase(
        callstack_.begin(),
        callstack_.begin() + static_cast<ssize_t>(frames_to_skip));
    callstack_.resize(number_of_frames);
  }

  std::string symbolize() const {
    // `backtrace_symbols` takes the return addresses obtained from
    // `backtrace()` and fetches string representations of each stack.
    // Unfortunately it doesn't return a struct of individual pieces of
    // information but a concatenated string, so we'll have to parse the string
    // after. NOTE: The array returned by `backtrace_symbols` is malloc'd and
    // must be manually freed, but not the strings inside the array.
    std::unique_ptr<char*, std::function<void(char**)>> raw_symbols(
        ::backtrace_symbols(
            callstack_.data(), static_cast<int>(callstack_.size())),
        /*deleter=*/free);
    const std::vector<std::string> symbols(
        raw_symbols.get(), raw_symbols.get() + callstack_.size());

    // The backtrace string goes into here.
    std::ostringstream stream;

    // Toggles to true after the first skipped python frame.
    bool has_skipped_python_frames = false;

    for (const auto frame_number : c10::irange(callstack_.size())) {
      const auto frame = parse_frame_information(symbols[frame_number]);

      if (skip_python_frames_ && frame && is_python_frame(*frame)) {
        if (!has_skipped_python_frames) {
          stream << "<omitting python frames>\n";
          has_skipped_python_frames = true;
        }
        continue;
      }

      // frame #<number>:
      stream << "frame #" << frame_number << ": ";

      if (frame) {
        // <function_name> + <offset> (<return-address> in <object-file>)
        stream << frame->function_name << " + " << frame->offset_into_function
               << " (" << callstack_[frame_number] << " in "
               << frame->object_file << ")\n";
      } else {
        // In the edge-case where we couldn't parse the frame string, we can
        // just use it directly (it may have a different format).
        stream << symbols[frame_number] << "\n";
      }
    }

    return stream.str();
  }

 private:
  const bool skip_python_frames_;
  std::vector<void*> callstack_;
};

#elif defined(_MSC_VER) // !SUPPORTS_BACKTRACE

const int max_name_len = 256;
std::string get_module_base_name(void* addr) {
  HMODULE h_module;
  char module[max_name_len];
  strcpy(module, "");
  GetModuleHandleEx(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      (LPCTSTR)addr,
      &h_module);
  if (h_module != NULL) {
    GetModuleFileNameA(h_module, module, max_name_len);
  }
  char* last_slash_pos = strrchr(module, '\\');
  if (last_slash_pos) {
    std::string module_base_name(last_slash_pos + 1);
    return module_base_name;
  } else {
    std::string module_base_name(module);
    return module_base_name;
  }
}
class SymbolHelper {
 public:
  static SymbolHelper& getInstance() {
    static SymbolHelper instance;
    return instance;
  }
  bool inited = false;
  HANDLE process;

 private:
  SymbolHelper() {
    process = GetCurrentProcess();
    DWORD flags = SymGetOptions();
    SymSetOptions(flags | SYMOPT_DEFERRED_LOADS);
    inited = SymInitialize(process, NULL, TRUE);
  }
  ~SymbolHelper() {
    if (inited) {
      SymCleanup(process);
    }
  }

 public:
  SymbolHelper(SymbolHelper const&) = delete;
  void operator=(SymbolHelper const&) = delete;
};

// This backtrace retrieval is implemented on Windows via the Windows API using
// `CaptureStackBackTrace`, `SymFromAddr` and `SymGetLineFromAddr64`.
// https://stackoverflow.com/questions/5693192/win32-backtrace-from-c-code
// https://stackoverflow.com/questions/26398064/counterpart-to-glibcs-backtrace-and-backtrace-symbols-on-windows
// https://docs.microsoft.com/en-us/windows/win32/debug/capturestackbacktrace
// https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symfromaddr
// https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symgetlinefromaddr64
// TODO: Support skipping python frames
class GetBacktraceImpl {
 public:
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t frames_to_skip,
      size_t maximum_number_of_frames,
      bool /* skip_python_frames */)
      : back_trace_(new void*[maximum_number_of_frames]) {
    // We always skip this frame (backtrace).
    frames_to_skip += 1;

    // Get the frames
    n_frame_ = CaptureStackBackTrace(
        static_cast<DWORD>(frames_to_skip),
        static_cast<DWORD>(maximum_number_of_frames),
        back_trace_.get(),
        NULL);
  }

  std::string symbolize() const {
    DWORD64 displacement;
    DWORD disp;
    std::unique_ptr<IMAGEHLP_LINE64> line;

    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO p_symbol = (PSYMBOL_INFO)buffer;

    bool with_symbol = false;
    bool with_line = false;

    // The backtrace string goes into here.
    std::ostringstream stream;

    // Initialize symbols if necessary
    SymbolHelper& sh = SymbolHelper::getInstance();

    for (USHORT i_frame = 0; i_frame < n_frame_; ++i_frame) {
      // Get the address and the name of the symbol
      if (sh.inited) {
        p_symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        p_symbol->MaxNameLen = MAX_SYM_NAME;
        with_symbol = SymFromAddr(
            sh.process, (ULONG64)back_trace_[i_frame], &displacement, p_symbol);
      }

      // Get the line number and the module
      if (sh.inited) {
        line.reset(new IMAGEHLP_LINE64());
        line->SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        with_line = SymGetLineFromAddr64(
            sh.process, (ULONG64)back_trace_[i_frame], &disp, line.get());
      }

      // Get the module basename
      std::string module = get_module_base_name(back_trace_[i_frame]);

      // The pattern on Windows is
      // `<return-address> <symbol-address>
      // <module-name>!<demangled-function-name> [<file-name> @ <line-number>]
      stream << std::setfill('0') << std::setw(16) << std::uppercase << std::hex
             << back_trace_[i_frame] << std::dec;
      if (with_symbol) {
        stream << std::setfill('0') << std::setw(16) << std::uppercase
               << std::hex << p_symbol->Address << std::dec << " " << module
               << "!" << p_symbol->Name;
      } else {
        stream << " <unknown symbol address> " << module << "!<unknown symbol>";
      }
      stream << " [";
      if (with_line) {
        stream << line->FileName << " @ " << line->LineNumber;
      } else {
        stream << "<unknown file> @ <unknown line number>";
      }
      stream << "]" << std::endl;
    }

    return stream.str();
  }

 private:
  std::unique_ptr<void*[]> back_trace_;
  USHORT n_frame_;
};

#else

class GetBacktraceImpl {
 public:
  C10_ALWAYS_INLINE GetBacktraceImpl(
      size_t /* frames_to_skip */,
      size_t /* maximum_number_of_frames */,
      bool /* skip_python_frames */) {}

  std::string symbolize() const {
    return "(no backtrace available)";
  }
};

#endif

} // namespace

std::string get_backtrace(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames) {
  return GetBacktraceImpl{
      frames_to_skip, maximum_number_of_frames, skip_python_frames}
      .symbolize();
}

Backtrace get_lazy_backtrace(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames) {
  class LazyBacktrace : public OptimisticLazyValue<std::string> {
   public:
    LazyBacktrace(GetBacktraceImpl&& impl) : impl_(std::move(impl)) {}

   private:
    std::string compute() const override {
      return impl_.symbolize();
    }

    GetBacktraceImpl impl_;
  };

  return std::make_shared<LazyBacktrace>(GetBacktraceImpl{
      frames_to_skip, maximum_number_of_frames, skip_python_frames});
}

} // namespace c10
