#include <caffe2/core/event_cpu.h>
#include <caffe2/core/operator.h>
#include <caffe2/proto/caffe2_pb.h>
#include <ideep/tensor.hpp>
#include "ideep_context.h"

namespace at {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CONTEXT(DeviceType::IDEEP, caffe2::IDEEPContext);

namespace {
void CopyBytesWrapper(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
  if (nbytes == 0) {
    return;
  }
  CAFFE_ENFORCE(src);
  CAFFE_ENFORCE(dst);
  memcpy(dst, src, nbytes);
}
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::IDEEP,
    DeviceType::CPU,
    CopyBytesWrapper);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CPU,
    DeviceType::IDEEP,
    CopyBytesWrapper);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::IDEEP,
    DeviceType::IDEEP,
    CopyBytesWrapper);
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(ideep::tensor);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(
    IDEEPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::IDEEP, IDEEPOperatorRegistry);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_CREATE_FUNCTION(IDEEP, EventCreateCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_RECORD_FUNCTION(IDEEP, EventRecordCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_WAIT_FUNCTION(IDEEP, IDEEP, EventWaitCPUCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_WAIT_FUNCTION(IDEEP, CPU, EventWaitCPUCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_WAIT_FUNCTION(CPU, IDEEP, EventWaitCPUCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_FINISH_FUNCTION(IDEEP, EventFinishCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_QUERY_FUNCTION(IDEEP, EventQueryCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(IDEEP, EventErrorMessageCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_SET_FINISHED_FUNCTION(IDEEP, EventSetFinishedCPU);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_EVENT_RESET_FUNCTION(IDEEP, EventResetCPU);

} // namespace caffe2
