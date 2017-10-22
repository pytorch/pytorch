#include <Python.h>

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

} // namespace torch
