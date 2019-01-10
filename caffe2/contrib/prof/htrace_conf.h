#pragma once

#include "caffe2/core/flags.h"

CAFFE2_DECLARE_string(caffe2_htrace_span_log_path);

namespace caffe2 {

const string defaultHTraceConf(const string& net_name);

} // namespace caffe2
