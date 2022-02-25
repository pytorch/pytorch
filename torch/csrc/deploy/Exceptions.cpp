#include <torch/csrc/deploy/Exceptions.h>

#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <cstdlib>
#include <functional>

namespace multipyError {

Error::Error(std::string msg, std::string backtrace, const void* caller)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), caller_(caller) {
  refresh_what();
}
namespace {
std::string get_backtrace(
    size_t frames_to_skip,
    size_t maximum_number_of_frames,
    bool skip_python_frames) {
#ifdef FBCODE_CAFFE2
  // For some reason, the stacktrace implementation in fbcode is
  // better than ours, see  https://github.com/pytorch/pytorch/issues/56399
  // When it's available, just use that.
  facebook::process::StackTrace st;
  return st.toString();

#elif SUPPORTS_BACKTRACE && !defined(C10_ANDROID)

  // We always skip this frame (backtrace).
  frames_to_skip += 1;

  std::vector<void*> callstack(
      frames_to_skip + maximum_number_of_frames, nullptr);
  // backtrace() gives us a list of return addresses in the current call stack.
  // NOTE: As per man (3) backtrace it can never fail
  // (http://man7.org/linux/man-pages/man3/backtrace.3.html).
  auto number_of_frames =
      ::backtrace(callstack.data(), static_cast<int>(callstack.size()));

  // Skip as many frames as requested. This is not efficient, but the sizes here
  // are small and it makes the code nicer and safer.
  for (; frames_to_skip > 0 && number_of_frames > 0;
       --frames_to_skip, --number_of_frames) {
    callstack.erase(callstack.begin());
  }

  // `number_of_frames` is strictly less than the current capacity of
  // `callstack`, so this is just a pointer subtraction and makes the subsequent
  // code safer.
  callstack.resize(static_cast<size_t>(number_of_frames));

  // `backtrace_symbols` takes the return addresses obtained from `backtrace()`
  // and fetches string representations of each stack. Unfortunately it doesn't
  // return a struct of individual pieces of information but a concatenated
  // string, so we'll have to parse the string after. NOTE: The array returned
  // by `backtrace_symbols` is malloc'd and must be manually freed, but not the
  // strings inside the array.
  std::unique_ptr<char*, std::function<void(char**)>> raw_symbols(
      ::backtrace_symbols(callstack.data(), static_cast<int>(callstack.size())),
      /*deleter=*/free);
  const std::vector<std::string> symbols(
      raw_symbols.get(), raw_symbols.get() + callstack.size());

  // The backtrace string goes into here.
  std::ostringstream stream;

  // Toggles to true after the first skipped python frame.
  bool has_skipped_python_frames = false;

  for (const auto frame_number : c10::irange(callstack.size())) {
    const auto frame = parse_frame_information(symbols[frame_number]);

    if (skip_python_frames && frame && is_python_frame(*frame)) {
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
             << " (" << callstack[frame_number] << " in " << frame->object_file
             << ")\n";
    } else {
      // In the edge-case where we couldn't parse the frame string, we can
      // just use it directly (it may have a different format).
      stream << symbols[frame_number] << "\n";
    }
  }

  return stream.str();

#elif SUPPORTS_BACKTRACE && defined(C10_ANDROID)

  std::ostringstream oss;
  dump_stack(oss, frames_to_skip, maximum_number_of_frames);
  return oss.str().c_str();

#elif defined(_MSC_VER) // !SUPPORTS_BACKTRACE
  // This backtrace retrieval is implemented on Windows via the Windows
  // API using `CaptureStackBackTrace`, `SymFromAddr` and
  // `SymGetLineFromAddr64`.
  // https://stackoverflow.com/questions/5693192/win32-backtrace-from-c-code
  // https://stackoverflow.com/questions/26398064/counterpart-to-glibcs-backtrace-and-backtrace-symbols-on-windows
  // https://docs.microsoft.com/en-us/windows/win32/debug/capturestackbacktrace
  // https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symfromaddr
  // https://docs.microsoft.com/en-us/windows/win32/api/dbghelp/nf-dbghelp-symgetlinefromaddr64
  // TODO: Support skipping python frames

  // We always skip this frame (backtrace).
  frames_to_skip += 1;

  DWORD64 displacement;
  DWORD disp;
  std::unique_ptr<IMAGEHLP_LINE64> line;

  char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
  PSYMBOL_INFO p_symbol = (PSYMBOL_INFO)buffer;

  std::unique_ptr<void*[]> back_trace(new void*[maximum_number_of_frames]);
  bool with_symbol = false;
  bool with_line = false;

  // The backtrace string goes into here.
  std::ostringstream stream;

  // Get the frames
  const USHORT n_frame = CaptureStackBackTrace(
      static_cast<DWORD>(frames_to_skip),
      static_cast<DWORD>(maximum_number_of_frames),
      back_trace.get(),
      NULL);

  // Initialize symbols if necessary
  SymbolHelper& sh = SymbolHelper::getInstance();

  for (USHORT i_frame = 0; i_frame < n_frame; ++i_frame) {
    // Get the address and the name of the symbol
    if (sh.inited) {
      p_symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
      p_symbol->MaxNameLen = MAX_SYM_NAME;
      with_symbol = SymFromAddr(
          sh.process, (ULONG64)back_trace[i_frame], &displacement, p_symbol);
    }

    // Get the line number and the module
    if (sh.inited) {
      line.reset(new IMAGEHLP_LINE64());
      line->SizeOfStruct = sizeof(IMAGEHLP_LINE64);
      with_line = SymGetLineFromAddr64(
          sh.process, (ULONG64)back_trace[i_frame], &disp, line.get());
    }

    // Get the module basename
    std::string module = get_module_base_name(back_trace[i_frame]);

    // The pattern on Windows is
    // `<return-address> <symbol-address>
    // <module-name>!<demangled-function-name> [<file-name> @ <line-number>]
    stream << std::setfill('0') << std::setw(16) << std::uppercase << std::hex
           << back_trace[i_frame] << std::dec;
    if (with_symbol) {
      stream << std::setfill('0') << std::setw(16) << std::uppercase << std::hex
             << p_symbol->Address << std::dec << " " << module << "!"
             << p_symbol->Name;
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
#else // !SUPPORTS_BACKTRACE && !_WIN32
  return "(no backtrace available)";
#endif // SUPPORTS_BACKTRACE
}

// NOLINTNEXTLINE(modernize-redundant-void-arg)
std::function<std::string(void)>* GetFetchStackTrace() {
  static std::function<std::string(void)> func = []() {
    return multipyError::get_backtrace(/*frames_to_skip=*/1, /*maximum_number_of_frames=*/64, /*skip_python_frames=*/true);
  };
  return &func;
};
} // namespace

// PyTorch-style error message
Error::Error(SourceLocation source_location, std::string msg)
    : Error(
          std::move(msg),
          "Exception raised from " +
              source_location.function +
              source_location.file +
              std::to_string(source_location.line) +
              " (most recent call first):\n" +
              (*GetFetchStackTrace())()) {}

// NB: This is defined in Logging.cpp for access to GetFetchStackTrace
// Caffe2-style error message
Error::Error(
    const std::string& file,
    const uint32_t line,
    const std::string& condition,
    const std::string& msg,
    const std::string& backtrace,
    const void* caller)
    : Error(
          "[enforce fail at " +
              detail::StripBasename(file) +
              ":" +
              std::to_string(line) +
              "] " +
              condition +
              ". " +
              msg,
          backtrace,
          caller) {}

std::string Error::compute_what(bool include_backtrace) const {
  std::ostringstream oss;

  oss << msg_;

  if (context_.size() == 1) {
    // Fold error and context in one line
    oss << " (" << context_[0] << ")";
  } else {
    for (const auto& c : context_) {
      oss << "\n  " << c;
    }
  }

  if (include_backtrace) {
    oss << "\n" << backtrace_;
  }

  return oss.str();
}

void Error::refresh_what() {
  what_ = compute_what(/*include_backtrace*/ true);
  what_without_backtrace_ = compute_what(/*include_backtrace*/ false);
}

namespace detail {

std::string StripBasename(const std::string& full_path) {
  const char kSeparator = '/';
  size_t pos = full_path.rfind(kSeparator);
  if (pos != std::string::npos) {
    return full_path.substr(pos + 1, std::string::npos);
  } else {
    return full_path;
  }
}

void multipyCheckFail(
    const std::string& func,
    const std::string& file,
    uint32_t line,
    const std::string& msg) {
  throw ::multipyError::Error({func, file, line}, msg);
}

void multipyInternalAssertFail(
    const std::string& func,
    const std::string& file,
    uint32_t line,
    const std::string& condMsg,
    const std::string& userMsg) {
  multipyCheckFail(func, file, line, condMsg + userMsg);
}

} // namespace detail
} // namespace multipyError
