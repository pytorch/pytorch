#pragma once

#include "caffe2/core/flags.h"

C10_DECLARE_string(caffe2_htrace_span_log_path);

namespace caffe2 {

const std::string defaultHTraceConf(const std::string& net_name);

} // namespace caffe2
