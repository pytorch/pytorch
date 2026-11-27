#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <oneapi/dnnl/dnnl_version.h>

#ifndef DNNL_PREREQ
// oneDNN version check macro
// Checks if the current oneDNN version is >= the specified version
#if defined(DNNL_VERSION_MAJOR) && defined(DNNL_VERSION_MINOR) && \
  defined(DNNL_VERSION_PATCH)
#define DNNL_PREREQ(major, minor, patch) \
  (((DNNL_VERSION_MAJOR << 16) + (DNNL_VERSION_MINOR << 8) + \
   (DNNL_VERSION_PATCH << 0)) >= \
   ((major << 16) + (minor << 8) + (patch << 0)))
#else
#define DNNL_PREREQ(major, minor, patch) 0
#endif
#endif

#endif // AT_MKLDNN_ENABLED()
