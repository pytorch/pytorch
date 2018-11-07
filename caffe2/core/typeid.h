#pragma once

// If I omit this header, the Windows build fails.  The error message
// was sufficiently bad that I couldn't figure out which downstream file
// was missing the include of common.h.  So keep it here for BC.
#include <caffe2/core/common.h>
#include <ATen/core/typeid.h>
