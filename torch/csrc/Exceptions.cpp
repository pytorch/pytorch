#ifndef NO_PYTHON
#include "torch/csrc/python_headers.h"

#include <utility>
#include <vector>
#include <cstdarg>

#include "THP.h"

PyObject *THPException_FatalError;

#define ASSERT_TRUE(cond) if (!(cond)) return false
bool THPException_init(PyObject *module)
{
  ASSERT_TRUE(THPException_FatalError = PyErr_NewException("torch.FatalError", NULL, NULL));
  ASSERT_TRUE(PyModule_AddObject(module, "FatalError", THPException_FatalError) == 0);
  return true;
}

namespace torch {

void replaceAll(std::string & str,
    const std::string & old_str,
    const std::string & new_str) {
  std::string::size_type pos = 0u;
  while ((pos = str.find(old_str, pos)) != std::string::npos) {
    str.replace(pos, old_str.length(), new_str);
  }
}

std::string processErrorMsg(std::string str) {

  // Translate Aten types to their respective pytorch ones
  std::vector<std::pair<std::string, std::string>> changes {
    {"Variable[SparseCUDAByteType]", "torch.cuda.sparse.ByteTensor"},
    {"Variable[SparseCUDACharType]", "torch.cuda.sparse.CharTensor"},
    {"Variable[SparseCUDADoubleType]", "torch.cuda.sparse.DoubleTensor"},
    {"Variable[SparseCUDAFloatType]", "torch.cuda.sparse.FloatTensor"},
    {"Variable[SparseCUDAIntType]", "torch.cuda.sparse.IntTensor"},
    {"Variable[SparseCUDALongType]", "torch.cuda.sparse.LongTensor"},
    {"Variable[SparseCUDAShortType]", "torch.cuda.sparse.ShortTensor"},
    {"Variable[SparseCUDAHalfType]", "torch.cuda.sparse.HalfTensor"},
    {"Variable[SparseCPUByteType]", "torch.sparse.ByteTensor"},
    {"Variable[SparseCPUCharType]", "torch.sparse.CharTensor"},
    {"Variable[SparseCPUDoubleType]", "torch.sparse.DoubleTensor"},
    {"Variable[SparseCPUFloatType]", "torch.sparse.FloatTensor"},
    {"Variable[SparseCPUIntType]", "torch.sparse.IntTensor"},
    {"Variable[SparseCPULongType]", "torch.sparse.LongTensor"},
    {"Variable[SparseCPUShortType]", "torch.sparse.ShortTensor"},
    {"Variable[SparseCPUHalfType]", "torch.sparse.HalfTensor"},
    {"Variable[CUDAByteType]", "torch.cuda.ByteTensor"},
    {"Variable[CUDACharType]", "torch.cuda.CharTensor"},
    {"Variable[CUDADoubleType]", "torch.cuda.DoubleTensor"},
    {"Variable[CUDAFloatType]", "torch.cuda.FloatTensor"},
    {"Variable[CUDAIntType]", "torch.cuda.IntTensor"},
    {"Variable[CUDALongType]", "torch.cuda.LongTensor"},
    {"Variable[CUDAShortType]", "torch.cuda.ShortTensor"},
    {"Variable[CUDAHalfType]", "torch.cuda.HalfTensor"},
    {"Variable[CPUByteType]", "torch.ByteTensor"},
    {"Variable[CPUCharType]", "torch.CharTensor"},
    {"Variable[CPUDoubleType]", "torch.DoubleTensor"},
    {"Variable[CPUFloatType]", "torch.FloatTensor"},
    {"Variable[CPUIntType]", "torch.IntTensor"},
    {"Variable[CPULongType]", "torch.LongTensor"},
    {"Variable[CPUShortType]", "torch.ShortTensor"},
    {"Variable[CPUHalfType]", "torch.HalfTensor"},
    {"SparseCUDAByteType", "torch.cuda.sparse.ByteTensor"},
    {"SparseCUDACharType", "torch.cuda.sparse.CharTensor"},
    {"SparseCUDADoubleType", "torch.cuda.sparse.DoubleTensor"},
    {"SparseCUDAFloatType", "torch.cuda.sparse.FloatTensor"},
    {"SparseCUDAIntType", "torch.cuda.sparse.IntTensor"},
    {"SparseCUDALongType", "torch.cuda.sparse.LongTensor"},
    {"SparseCUDAShortType", "torch.cuda.sparse.ShortTensor"},
    {"SparseCUDAHalfType", "torch.cuda.sparse.HalfTensor"},
    {"SparseCPUByteType", "torch.sparse.ByteTensor"},
    {"SparseCPUCharType", "torch.sparse.CharTensor"},
    {"SparseCPUDoubleType", "torch.sparse.DoubleTensor"},
    {"SparseCPUFloatType", "torch.sparse.FloatTensor"},
    {"SparseCPUIntType", "torch.sparse.IntTensor"},
    {"SparseCPULongType", "torch.sparse.LongTensor"},
    {"SparseCPUShortType", "torch.sparse.ShortTensor"},
    {"SparseCPUHalfType", "torch.sparse.HalfTensor"},
    {"CUDAByteType", "torch.cuda.ByteTensor"},
    {"CUDACharType", "torch.cuda.CharTensor"},
    {"CUDADoubleType", "torch.cuda.DoubleTensor"},
    {"CUDAFloatType", "torch.cuda.FloatTensor"},
    {"CUDAIntType", "torch.cuda.IntTensor"},
    {"CUDALongType", "torch.cuda.LongTensor"},
    {"CUDAShortType", "torch.cuda.ShortTensor"},
    {"CUDAHalfType", "torch.cuda.HalfTensor"},
    {"CPUByteType", "torch.ByteTensor"},
    {"CPUCharType", "torch.CharTensor"},
    {"CPUDoubleType", "torch.DoubleTensor"},
    {"CPUFloatType", "torch.FloatTensor"},
    {"CPUIntType", "torch.IntTensor"},
    {"CPULongType", "torch.LongTensor"},
    {"CPUShortType", "torch.ShortTensor"},
    {"CPUHalfType", "torch.HalfTensor"},
  };

  for (const auto & it : changes) {
    replaceAll(str, it.first, it.second);
  }

  return str;
}

static std::string formatMessage(const char *format, va_list fmt_args) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);
  return std::string(error_buf);
}

IndexError::IndexError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

TypeError::TypeError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

ValueError::ValueError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

} // namespace torch

#else

#include "Exceptions.h"

#include <cstdarg>

namespace torch {

static std::string formatMessage(const char *format, va_list fmt_args) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);
  return std::string(error_buf);
}

ValueError::ValueError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

} // namespace torch

#endif
