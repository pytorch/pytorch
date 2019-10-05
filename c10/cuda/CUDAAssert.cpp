#include <c10/cuda/CUDAGuard.h>

#include <stdio.h>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "CUDAAssert.h"

namespace c10 {
namespace cuda {

#define SOURCE_ENTRY(sym, file_name) \
  { AssertSource::sym, file_name }

static std::unordered_map<AssertSource, std::string> assert_source_map{
    SOURCE_ENTRY(MultinomialKernel, "MultinomialKernel.cu"),
    SOURCE_ENTRY(BinaryOpsKernel, "BinaryOpsKernel.cu"),
    SOURCE_ENTRY(IndexKernel, "IndexKernel.cu"),
    SOURCE_ENTRY(THCTensorIndex, "THCTensorIndex.cu"),
    SOURCE_ENTRY(ClassNLLCriterion, "ClassNLLCriterion.cu"),
    SOURCE_ENTRY(Distributions, "Distributions.cu"),
    SOURCE_ENTRY(EmbeddingBag, "EmbeddingBag.cu"),
    SOURCE_ENTRY(FractionalMaxPoo2d, "FractionalMaxPoo2d.cu"),
    SOURCE_ENTRY(SpatialClassNLLCriterion, "SpatialClassNLLCriterion.cu")};

// global assert error output format table
static std::unordered_map<AssertFormat, const char*> assert_format_table{
    {AssertFormat::ZeroDivisionError,
     "ZeroDivisionError: integer division or modulo by zero"},
    {AssertFormat::IndexOutOfBounds,
     "Index %d is out of bounds for dimension %d with size %d"},
    // MultinomialKernel.cu:
    {AssertFormat::MultinomialProbLessZero,
     "invalid multinomial distribution (encountering probability entry < 0)"},
    {AssertFormat::MultinomialProbInf,
     "invalid multinomial distribution (encountering probability entry = infinity)"},
    {AssertFormat::MultinomialProbNaN,
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

std::string fileFromAssertSource(AssertSource source) {
  auto iter = assert_source_map.find(source);
  if (iter != assert_source_map.end()) {
    return iter->second;
  }

  return "unknown";
}

std::string generateErrorMessage(CUDAAssert* assert_state) {
  char* args = assert_state->buffer;
  const int32_t error_code = assert_state->error;
  auto iter = assert_format_table.find(assert_state->format);
  if (iter != assert_format_table.end()) {
    // format error specific message
    const char* format = iter->second;
    return formatAssertOutput(format, args);
  }

  // fall back to default assert error message
  auto file = fileFromAssertSource(assert_state->file);
  const auto line = assert_state->line;
  return c10::str("Kernel Assertion failed [", file, ":", line, "]");
}

void checkAssertError(CUDAAssert* assert_state) {
  if (assert_state->error != 0) {
    // wait for kernels to complete
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaErrorAssert) {
      C10_CUDA_CHECK(error);
    }

    std::lock_guard<std::mutex> lock(*assert_state->mutex);
    if (assert_state->error == 0) {
      return;
    }

    // generate full error message
    auto assert_error = static_cast<AssertError>(assert_state->error);
    auto error_message = generateErrorMessage(assert_state);
    auto file = fileFromAssertSource(assert_state->file);
    auto line = assert_state->line;

    if (!assert_state->persistent) {
      // reset assert state if error is not persistent
      memset(assert_state->buffer, 0, sizeof(assert_state->buffer));
      assert_state->length = 0;
      assert_state->persistent = false;
      assert_state->format = AssertFormat::Default;
      std::atomic_thread_fence(std::memory_order_release);
      assert_state->error = 0;
    }

    switch (assert_error) {
      case AssertError::IndexError:
        throw ::c10::IndexError(
            {"kernel_assert", file.c_str(), line}, error_message);
      case AssertError::Error:
      default:
        throw ::c10::Error(
            {"kernel_assert", file.c_str(), line}, error_message);
    }
  }
}

} // namespace cuda
} // namespace c10
