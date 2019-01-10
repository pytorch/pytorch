/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_CORE_NET_ASYNC_TRACING_H_
#define CAFFE2_CORE_NET_ASYNC_TRACING_H_

#include "caffe2/core/common.h"
#include "caffe2/core/net_async_base.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

CAFFE2_DECLARE_string(caffe2_net_async_tracing_filepath);
CAFFE2_DECLARE_string(caffe2_net_async_names_to_trace);
CAFFE2_DECLARE_int(caffe2_net_async_tracing_nth);

namespace caffe2 {
namespace tracing {

struct TracerEvent {
  int op_id_ = -1;
  int task_id_ = -1;
  int stream_id_ = -1;
  const char* name_ = nullptr;
  const char* category_ = nullptr;
  long timestamp_ = -1.0;
  bool is_beginning_ = false;
  long thread_label_ = -1;
  std::thread::id tid_;
};

enum TracingField {
  TRACE_OP,
  TRACE_TASK,
  TRACE_STREAM,
  TRACE_THREAD,
  TRACE_NAME,
  TRACE_CATEGORY,
};

class Tracer {
 public:
  Tracer(const NetBase* net, const std::string& net_name);

  void recordEvent(const TracerEvent& event);
  std::string opTraceName(const OperatorBase* op);
  std::string opBlobsInfo(const OperatorBase& op);
  std::string serializeEvent(const TracerEvent& event);
  void linearizeEvents();
  void renameThreads();
  void setEnabled(bool enabled);
  bool isEnabled() const;
  int bumpIter();

  virtual ~Tracer();

 private:
  const NetBase* net_ = nullptr;
  std::string filename_;
  std::vector<TracerEvent> events_;
  std::mutex tracer_mutex_;
  bool enabled_ = false;
  Timer timer_;
  int iter_;

  friend class TracerGuard;
};

class TracerGuard {
 public:
  TracerGuard() {}

  void init(Tracer* tracer);

  void addArgument();
  void addArgument(TracingField field, const char* value);
  void addArgument(TracingField field, int value);

  template <typename T, typename... Args>
  void addArgument(TracingField field, const T& value, const Args&... args) {
    addArgument(field, value);
    addArgument(args...);
  }

  void recordEventStart();

  virtual ~TracerGuard();

 private:
  bool enabled_ = false;
  TracerEvent event_;
  Tracer* tracer_;
};

bool isTraceableNet(const std::string& net_name);

std::shared_ptr<Tracer> create(const NetBase* net, const std::string& net_name);
bool startIter(const std::shared_ptr<Tracer>& tracer);

} // namespace tracing

#define TRACE_NAME_CONCATENATE(s1, s2) s1##s2
#define TRACE_ANONYMOUS_NAME(str) TRACE_NAME_CONCATENATE(str, __LINE__)

#define TRACE_EVENT_INIT(...)                                 \
  TRACE_ANONYMOUS_NAME(trace_guard).init(tracer_.get());      \
  TRACE_ANONYMOUS_NAME(trace_guard).addArgument(__VA_ARGS__); \
  TRACE_ANONYMOUS_NAME(trace_guard).recordEventStart();

// Supposed to be used only once per scope in AsyncNetBase-derived nets
#define TRACE_EVENT(...)                                  \
  tracing::TracerGuard TRACE_ANONYMOUS_NAME(trace_guard); \
  if (tracer_ && tracer_->isEnabled()) {                  \
    TRACE_EVENT_INIT(__VA_ARGS__)                         \
  }

#define TRACE_EVENT_IF(cond, ...)                         \
  tracing::TracerGuard TRACE_ANONYMOUS_NAME(trace_guard); \
  if (tracer_ && tracer_->isEnabled() && (cond)) {        \
    TRACE_EVENT_INIT(__VA_ARGS__)                         \
  }

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_TRACING_H_
