#include <c10/cuda/CUDAGuard.h>

#include <stdio.h>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "CUDAAssert.h"

namespace c10 {
namespace cuda {

struct FileErrorRange {
  std::string file;
  int32_t first;
  int32_t last;
};

#define FILE_ERROR_RANGE(file, tag)                 \
  {                                                 \
    (int32_t) tag::FIRST, {                         \
      file, (int32_t)tag::FIRST, (int32_t)tag::LAST \
    }                                               \
  }

static std::map<int32_t, FileErrorRange> error_range_file_table{
    FILE_ERROR_RANGE("MultinomialKernel.cu", AssertTag::MultinomialKernel),
    FILE_ERROR_RANGE("BinaryOpsKernell.cu", AssertTag::BinaryOpsKernel),
    FILE_ERROR_RANGE("IndexKernel.cu", AssertTag::IndexKernel),
    FILE_ERROR_RANGE("THCTensorIndex.cu", AssertTag::THCTensorIndex),
    FILE_ERROR_RANGE("ClassNLLCriterion.cu", AssertTag::ClassNLLCriterion),
    FILE_ERROR_RANGE("Distributions.cu", AssertTag::Distributions),
    FILE_ERROR_RANGE("EmbeddingBag.cu", AssertTag::EmbeddingBag),
    FILE_ERROR_RANGE("FractionalMaxPoo2d.cu", AssertTag::FractionalMaxPoo2d),
    FILE_ERROR_RANGE(
        "SpatialClassNLLCriterion.cu",
        AssertTag::SpatialClassNLLCriterion)};

// global assert error output format table
static std::unordered_map<int32_t, const char*> assert_format_table{
    // MultinomialKernel.cu:
    {(int32_t)AssertTag::MultinomialKernel::_004,
     "invalid multinomial distribution (encountering probability entry < 0)"},
    {(int32_t)AssertTag::MultinomialKernel::_005,
     "invalid multinomial distribution (encountering probability entry = infinity)"},
    {(int32_t)AssertTag::MultinomialKernel::_006,
     "invalid multinomial distribution (encountering probability entry = NaN)"}};

template <typename T>
static std::string formatToken(const std::string& fmt, const T& data) {
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

static std::string formatAssertOutput(const char* format, char* data) {
  std::stringstream ss;

  const char* p = std::strchr(format, '%');
  while (p != nullptr) {
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
        ss << formatToken(format_spec, *reinterpret_cast<int*>(data));
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
          ss << formatToken(format_spec, *reinterpret_cast<float*>(data));
        else
          ss << formatToken(format_spec, *reinterpret_cast<double*>(data));
        break;

      // Strings are handled in a special way
      case 's':
        ss << formatToken(format_spec, (const char*)data);
        break;

      // % is special
      case '%':
        ss << formatToken("%%", 0);
        break;

      // Everything else is just printed out as-is
      default:
        ss << formatToken("%s", fmt);
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

std::string fileFromErrorCode(int32_t error_code) {
  auto iter = error_range_file_table.lower_bound(error_code);
  if (iter != error_range_file_table.end()) {
    const auto& range = iter->second;
    if (error_code >= range.first && error_code <= range.last) {
      return range.file;
    }
  }

  return "unknown";
}

std::string generateErrorMessage(CUDAAssert* assert_state) {
  char* args = assert_state->buffer;
  const int32_t error_code = assert_state->error;
  auto iter = assert_format_table.find(error_code);
  if (iter != assert_format_table.end()) {
    // format error specific message
    const char* format = iter->second;
    return formatAssertOutput(format, args);
  } else {
    auto file = fileFromErrorCode((int32_t)assert_state->error);
    const auto line = assert_state->line;

    // if no fomat message was specified, fall back to default messages
    switch (assert_state->kind) {
      case AssertKind::ZERO_DIVISION:
        return "ZeroDivisionError: integer division or modulo by zero";
      case AssertKind::INDEX_ERROR: {
        return formatAssertOutput(
            "Index %d is out of bounds for dimension %d with size %d", args);
      }
      default:
        return c10::str(
            "Kernel Assertion failed [#",
            assert_state->error,
            "; ",
            file,
            ":",
            line,
            "]");
    }
  }
}

void checkAssertError(c10::cuda::CUDAAssert* assert_state) {
  if (assert_state->error != 0) {
    cudaError_t error = cudaDeviceSynchronize(); // wait for kernels to complete
    if (error != cudaErrorAssert) {
      C10_CUDA_CHECK(error);
    }

    std::lock_guard<std::mutex> lock(*assert_state->mutex);

    // generate full error message
    auto error_message = generateErrorMessage(assert_state);

    // instead of the real function name (which we do not know)
    // use the error code to identify location
    auto function = c10::str("kernel_assert #", assert_state->error);
    auto file = fileFromErrorCode(assert_state->error);
    auto kind = assert_state->kind;
    auto line = assert_state->line;

    if (!assert_state->persistent) {
      // reset assert state if error is not persistent
      memset(assert_state->buffer, 0, sizeof(assert_state->buffer));
      assert_state->length = 0;
      assert_state->persistent = false;
      assert_state->kind = AssertKind::DEFAULT;
      std::atomic_thread_fence(std::memory_order_release);
      assert_state->error = 0;
    }

    switch (kind) {
      case AssertKind::INDEX_ERROR:
        throw ::c10::IndexError(
            {function.c_str(), file.c_str(), line}, error_message);
        break;
      case AssertKind::ZERO_DIVISION:
      case AssertKind::DEFAULT:
      default:
        throw ::c10::Error(
            {function.c_str(), file.c_str(), line}, error_message);
    }
  }
}

} // namespace cuda
} // namespace c10
