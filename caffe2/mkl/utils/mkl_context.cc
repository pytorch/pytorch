#include "caffe2/mkl/utils/mkl_context.h"

#include "caffe2/core/event.h"

namespace caffe2 {

// For MKLDNN devices, Event is essentially a no-op since they are all
// synchronous.
void EventCreateMKLDNN(const DeviceOption& /* unused */, Event* /* unused */) {}
void EventRecordMKLDNN(const void* /* unused */, Event* /* unused */) {}
void EventWaitMKLDNNMKLDNN(const Event* /* unused */, void* /* unused */) {}
void EventFinishMKLDNN(Event* /* unused */) {}

REGISTER_EVENT_CREATE_FUNCTION(MKLDNN, EventCreateMKLDNN);
REGISTER_EVENT_RECORD_FUNCTION(MKLDNN, EventRecordMKLDNN);
REGISTER_EVENT_WAIT_FUNCTION(MKLDNN, MKLDNN, EventWaitMKLDNNMKLDNN);
REGISTER_EVENT_FINISH_FUNCTION(MKLDNN, EventFinishMKLDNN);

} // namespace caffe2
