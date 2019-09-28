#include <c10/cuda/CUDAGuard.h>

#include <stdio.h>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "CUDAAssert.h"

namespace c10 {
namespace cuda {

template <typename T>
static std::string format_token(const std::string& fmt, const T& data) {
  int ret = snprintf(nullptr, 0, fmt.c_str(), data);
  if (ret < 0) {
    throw std::runtime_error("Calculating token output size failed.");
  }

  if (ret == 0) {
    return std::string{};
  }

  size_t size = ret + 1; // ret does not include the terminating '\0'
  auto buff_ptr = std::unique_ptr<char[]>(new char[size]);
  ret = snprintf(buff_ptr.get(), size, fmt.c_str(), data);
  if (ret < 0 || ret >= size) {
    throw std::runtime_error("Formatting token failed.");
  }

  return std::string(buff_ptr.get());
}

static std::string format_assert_output(const char* format, char* data) {
  std::stringstream ss;

  const char* p = std::strchr(format, '%');
  while (p != '\0') {
    ss << std::string(format, p - format);

    // handle the format specifier
    const char* fmt = p++; // move past the '%'
    p += std::strcspn(p, "%cdiouxXeEfgGaAnps");
    if (*p == '\0') { // if no format specifier follows, print the whole thing
      format = fmt;
      break;
    }

    // get data pointer
    size_t arglen = *reinterpret_cast<size_t*>(data);
    if (arglen > c10::cuda::C10_ASSERT_BUFFER_SIZE)
      throw std::out_of_range(
          "Kernel assert state argument size is out of bounds.");

    data += c10::cuda::C10_ASSERT_ARG_ALIGN_SIZE;

    // cut out the format spec
    char specifier = *p++;
    std::string format_spec(fmt, p - fmt);

    switch (specifier) {
      // integer arguments
      case 'c':
      case 'd':
      case 'i':
      case 'o':
      case 'u':
      case 'x':
      case 'X':
      case 'p':
        ss << format_token(format_spec, *reinterpret_cast<int*>(data));
        break;

      // floating point arguments
      case 'e':
      case 'E':
      case 'f':
      case 'g':
      case 'G':
      case 'a':
      case 'A':
        if (arglen == sizeof(float)) // float or double
          ss << format_token(format_spec, *reinterpret_cast<float*>(data));
        else
          ss << format_token(format_spec, *reinterpret_cast<double*>(data));
        break;

      // Strings are handled in a special way
      case 's':
        ss << format_token(format_spec, (const char*)data);
        break;

      // % is special
      case '%':
        ss << format_token("%%", 0);
        break;

      // Everything else is just printed out as-is
      default:
        ss << format_token("%s", fmt);
        break;
    }

    // Move on to next argument
    data += arglen;
    size_t misalign = arglen % c10::cuda::C10_ASSERT_ARG_ALIGN_SIZE;
    if (misalign > 0) {
      data += c10::cuda::C10_ASSERT_ARG_ALIGN_SIZE - misalign;
    }

    format = p; // Adjust format string to be past the specifier
    p = std::strchr(format, '%'); // and get the next specifier
  }

  // Print out the remaining part of the string
  ss << format;
  return ss.str();
}

static char* next_field(char*& ptr) {
  constexpr size_t align_size = c10::cuda::C10_ASSERT_ARG_ALIGN_SIZE;

  size_t size = *reinterpret_cast<size_t*>(ptr);
  char* field_data =
      ptr + align_size; // alignment is also used  as size for the size field
  ptr = field_data + size;
  size_t misalign = size % align_size;
  if (misalign > 0) {
    ptr += align_size - misalign;
  }
  return field_data;
}

void check_assert_error(c10::cuda::CUDAAssert* assert_state) {
  if (assert_state->error != 0) {
    cudaError_t error = cudaDeviceSynchronize(); // wait for kernel to complete
    if (error != cudaErrorAssert) {
      C10_CUDA_CHECK(error);
    }

    c10::cuda::CUDAAssert isolation_copy(*assert_state);

    // reset assert state
    memset(assert_state, 0, sizeof(c10::cuda::CUDAAssert));

    // decode output
    char* ptr = isolation_copy.buffer;
    const char* file = next_field(ptr);
    const char* func = next_field(ptr);
    const char* expression = next_field(ptr);
    char* format = next_field(ptr);
    int32_t line = *reinterpret_cast<int32_t*>(next_field(ptr));
    char* args = ptr;

    auto message = format_assert_output(format, args);

    // generate full error message
    auto error_message = c10::str(
        message,
        '\n',
        file,
        ":",
        line,
        ": Assertion `",
        expression,
        "` failed.");

    throw ::c10::Error({func, file, line}, error_message);
  }
}

} // namespace cuda
} // namespace c10
