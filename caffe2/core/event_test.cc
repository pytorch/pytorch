#include <gtest/gtest.h>
#include "caffe2/core/context.h"
#include "caffe2/core/event.h"

namespace caffe2 {

// For the CPU event test, we only test if these functions are properly
// registered. Nothing special needs to be checked since CPU events are no-ops.
TEST(EventCPUTest, EventBasics) {
  DeviceOption device_option;
  device_option.set_device_type(CPU);
  Event event(device_option);
  CPUContext context;

  // Calling from Context
  context.WaitEvent(event);
  context.Record(&event);

  // Calling from Event
  event.Finish();
  event.Record(CPU, &context);
  event.Wait(CPU, &context);
}

} // namespace caffe2
