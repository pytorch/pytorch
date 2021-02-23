#include <caffe2/core/event_cpu.h>
#include <caffe2/core/operator.h>
#include <caffe2/proto/caffe2_pb.h>
#include <ideep/tensor.hpp>
#include "ideep_context.h"

namespace at {
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

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::IDEEP,
    DeviceType::CPU,
    CopyBytesWrapper);
REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CPU,
    DeviceType::IDEEP,
    CopyBytesWrapper);
REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::IDEEP,
    DeviceType::IDEEP,
    CopyBytesWrapper);
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(ideep::tensor);

C10_DEFINE_REGISTRY(
    IDEEPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

CAFFE_REGISTER_DEVICE_TYPE(DeviceType::IDEEP, IDEEPOperatorRegistry);

REGISTER_EVENT_CREATE_FUNCTION(IDEEP, EventCreateCPU);
REGISTER_EVENT_RECORD_FUNCTION(IDEEP, EventRecordCPU);
REGISTER_EVENT_WAIT_FUNCTION(IDEEP, IDEEP, EventWaitCPUCPU);
REGISTER_EVENT_WAIT_FUNCTION(IDEEP, CPU, EventWaitCPUCPU);
REGISTER_EVENT_WAIT_FUNCTION(CPU, IDEEP, EventWaitCPUCPU);
REGISTER_EVENT_FINISH_FUNCTION(IDEEP, EventFinishCPU);
REGISTER_EVENT_QUERY_FUNCTION(IDEEP, EventQueryCPU);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(IDEEP, EventErrorMessageCPU);
REGISTER_EVENT_SET_FINISHED_FUNCTION(IDEEP, EventSetFinishedCPU);
REGISTER_EVENT_RESET_FUNCTION(IDEEP, EventResetCPU);

} // namespace caffe2
