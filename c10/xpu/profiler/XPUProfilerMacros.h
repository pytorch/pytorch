#pragma once

#include <libkineto.h>
#include <output_base.h>

#include <pti/pti_view.h>

#include <c10/xpu/XPUMacros.h>

namespace c10::xpu {

class XPUActivityApi;

using act_t = libkineto::ActivityType;
using logger_t = libkineto::ActivityLogger;
using itrace_t = libkineto::ITraceActivity;

}
