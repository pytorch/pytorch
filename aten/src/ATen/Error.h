#pragma once

#include <ATen/ATenGeneral.h> // for AT_API
#include <ATen/optional.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <stdarg.h>

#if !defined(_WIN32)
#include <cxxabi.h>
#include <execinfo.h>
#endif // !defined(_WIN32)

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

namespace at {
namespace detail {

// TODO: This backtrace retrieval can be implemented on Windows via the Windows
// API using `CaptureStackBackTrace` and `SymFromAddr`.
// https://stackoverflow.com/questions/5693192/win32-backtrace-from-c-code
// https://stackoverflow.com/questions/26398064/counterpart-to-glibcs-backtrace-and-backtrace-symbols-on-windows
// https://msdn.microsoft.com/en-us/library/windows/desktop/bb204633%28v=vs.85%29.aspx.
#if !defined(_WIN32)
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

inline at::optional<FrameInformation> parse_frame_information(
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

  auto function_name_start = frame_string.find("(");
  if (function_name_start == std::string::npos) {
    return at::nullopt;
  }
  function_name_start += 1;

  auto offset_start = frame_string.find('+', function_name_start);
  if (offset_start == std::string::npos) {
    return at::nullopt;
  }
  offset_start += 1;

  const auto offset_end = frame_string.find(')', offset_start);
  if (offset_end == std::string::npos) {
    return at::nullopt;
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
  return at::nullopt;
#endif // defined(__GLIBCXX__)

  // Some system-level functions don't have sufficient debug information, so
  // we'll display them as "<unknown function>". They'll still have a return
  // address and other pieces of information.
  if (mangled_function_name.empty()) {
    frame.function_name = "<unknown function>";
    return frame;
  }

  int status = -1;
  // This function will demangle the mangled function name into a more human
  // readable format, e.g. _Z1gv -> g().
  // More information:
  // https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/libsupc%2B%2B/cxxabi.h
  // NOTE: `__cxa_demangle` returns a malloc'd string that we have to free
  // ourselves.
  std::unique_ptr<char, std::function<void(char*)>> demangled_function_name(
      abi::__cxa_demangle(
          mangled_function_name.c_str(),
          /*__output_buffer=*/nullptr,
          /*__length=*/0,
          &status),
      /*deleter=*/free);

  // Demangling may fail, for example when the name does not follow the
  // standard C++ (Itanium ABI) mangling scheme. This is the case for `main`
  // or `clone` for example, so the mangled name is a fine default.
  if (status == 0) {
    frame.function_name = demangled_function_name.get();
  } else {
    frame.function_name = mangled_function_name;
  }

  return frame;
}

inline std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64) {
  // We always skip this frame (backtrace).
  frames_to_skip += 1;

  std::vector<void*> callstack(
      frames_to_skip + maximum_number_of_frames, nullptr);
  // backtrace() gives us a list of return addresses in the current call stack.
  // NOTE: As per man (3) backtrace it can never fail
  // (http://man7.org/linux/man-pages/man3/backtrace.3.html).
  auto number_of_frames = ::backtrace(callstack.data(), callstack.size());

  // Skip as many frames as requested. This is not efficient, but the sizes here
  // are small and it makes the code nicer and safer.
  for (; frames_to_skip > 0 && number_of_frames > 0;
       --frames_to_skip, --number_of_frames) {
    callstack.erase(callstack.begin());
  }

  // `number_of_frames` is strictly less than the current capacity of
  // `callstack`, so this is just a pointer subtraction and makes the subsequent
  // code safer.
  callstack.resize(number_of_frames);

  // `backtrace_symbols` takes the return addresses obtained from `backtrace()`
  // and fetches string representations of each stack. Unfortunately it doesn't
  // return a struct of individual pieces of information but a concatenated
  // string, so we'll have to parse the string after. NOTE: The array returned
  // by `backtrace_symbols` is malloc'd and must be manually freed, but not the
  // strings inside the array.
  std::unique_ptr<char*, std::function<void(char**)>> raw_symbols(
      ::backtrace_symbols(callstack.data(), callstack.size()),
      /*deleter=*/free);
  const std::vector<std::string> symbols(
      raw_symbols.get(), raw_symbols.get() + callstack.size());

  // The backtrace string goes into here.
  std::ostringstream stream;

  for (size_t frame_number = 0; frame_number < callstack.size();
       ++frame_number) {
    const auto frame = parse_frame_information(symbols[frame_number]);

    // frame #<number>:
    stream << "frame #" << frame_number << ": ";

    if (!frame.has_value()) {
      // In the edge-case where we couldn't parse the frame string, we can just
      // use it directly (it may have a different format).
      stream << symbols[frame_number] << "\n";
    } else {
      // <function_name> + <offset> (<return-address> in <object-file>)
      stream << frame->function_name << " + " << frame->offset_into_function
             << " (" << callstack[frame_number] << " in " << frame->object_file
             << ")\n";
    }
  }

  return stream.str();
}
#endif // !defined(_WIN32)

/// A tiny implementation of static `all_of`.
template <bool...>
struct pack;
template <bool... values>
struct all_of : std::is_same<pack<values..., true>, pack<true, values...>> {};

/// A printf wrapper that returns an std::string.
inline std::string format(const char* format_string, ...) {
  static constexpr size_t kMaximumStringLength = 4096;
  char buffer[kMaximumStringLength];

  va_list format_args;
  va_start(format_args, format_string);
  vsnprintf(buffer, sizeof(buffer), format_string, format_args);
  va_end(format_args);

  return buffer;
}

/// Represents a location in source code (for debugging).
struct SourceLocation {
  std::string toString() const {
    return format("%s at %s:%d", function, file, line);
  }

  const char* function;
  const char* file;
  uint32_t line;
};
} // namespace detail

/// The primary ATen error class.
/// Provides a complete error message with source location information via
/// `what()`, and a more concise message via `what_without_backtrace()`. Should
/// primarily be used with the `AT_ERROR` macro.
struct AT_API Error : public std::exception {
  template <typename... FormatArgs>
  Error(
      detail::SourceLocation source_location,
      const char* format_string,
      FormatArgs&&... format_args)
      : what_without_backtrace_(detail::format(
            format_string,
            std::forward<FormatArgs>(format_args)...)),
        what_(what_without_backtrace_) {
    // NOTE: A "literal type"
    // (http://en.cppreference.com/w/cpp/concept/LiteralType) could also be a
    // constexpr struct, so it's not 100% guaranteed that the `printf` call
    // inside `format` is safe, but it will catch 99.9% of all errors we'll run
    // into, such as passing `std::string`.
    static_assert(
        detail::all_of<std::is_literal_type<FormatArgs>::value...>::value,
        "format arguments must be literal types!");
    what_ += " (" + source_location.toString() + ")\n";
#if !defined(_WIN32)
    // Skip this constructor's frame.
    what_ += detail::get_backtrace(/*frames_to_skip=*/1);
#endif // !defined(_WIN32)
  }

  /// Returns the complete error message, including the source location.
  const char* what() const noexcept override {
    return what_.c_str();
  }

  /// Returns only the error message string, without source location.
  const char* what_without_backtrace() const noexcept {
    return what_without_backtrace_.c_str();
  }

 private:
  std::string what_without_backtrace_;
  std::string what_;
};
} // namespace at

#define AT_ERROR(...) \
  throw at::Error({__func__, __FILE__, __LINE__}, __VA_ARGS__)

#define AT_ASSERT(cond, ...) \
  if (!(cond)) {             \
    AT_ERROR(__VA_ARGS__);   \
  }
