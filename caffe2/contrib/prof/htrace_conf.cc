#include "htrace_conf.h"

#include <htrace.hpp>
#include <algorithm>
#include <ctime>

CAFFE2_DEFINE_string(
    caffe2_htrace_span_log_path,
    "",
    "Span log path for htrace");

namespace caffe2 {

const string defaultHTraceConf(const string& net_name) {
  // create a duplicate because we may need to modify the name
  string net_name_copy(net_name);

  // make sure the net name is a valid file name
  std::replace(net_name_copy.begin(), net_name_copy.end(), '/', '_');
  std::replace(net_name_copy.begin(), net_name_copy.end(), '\\', '_');

  // take current local time
  time_t rawtime;
  std::time(&rawtime);
  struct tm timeinfo;
  localtime_r(&rawtime, &timeinfo);

  // and append it to the log file name, in a human-readable format
  std::string buf;
  buf.resize(30); // 15 should be enough, but apparently is too short.
  strftime(&buf[0], buf.size(), "%Y%m%d_%H%M%S", &timeinfo);
  auto datetime = buf.data();

  std::stringstream stream;
  stream << HTRACE_SPAN_RECEIVER_KEY << "=local.file;";
  stream << HTRACE_SAMPLER_KEY << "=always;";

  if (FLAGS_caffe2_htrace_span_log_path.empty()) {
    stream << HTRACE_LOCAL_FILE_RCV_PATH_KEY << "=/tmp/htrace_" << net_name_copy
           << "_span_log_" << datetime << ";";
  } else {
    stream << HTRACE_LOCAL_FILE_RCV_PATH_KEY << "="
           << FLAGS_caffe2_htrace_span_log_path << ";";
  }

  return stream.str();
}

} // namespace caffe2
