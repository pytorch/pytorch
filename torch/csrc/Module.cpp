#include <torch/csrc/python_headers.h>
#include <sys/types.h>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include <unordered_map>
#include <cstdlib>
#include <libshm.h>
#include <TH/TH.h>
#include <c10/util/Logging.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/VmapMode.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DataLoader.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
sdfiosdjfsiodjf

// can happen if the same csrc files are compiled into multiple shared
// libraries.
inline void pytorch_duplicate_guard() {
  static int initialized = 0;
  if (initialized) {
    fprintf(stderr, "pytorch: _C shared library re-initialized\n");
    abort();
  }
  initialized = 1;
;}

struct call_duplicate_guard {
  call_duplicate_guard() { pytorch_duplicate_guard(); }
};

static call_duplicate_guard _call_duplicate_guard;
