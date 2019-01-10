#include <gtest/gtest.h>
#include "caffe2/core/context.h"
#include "caffe2/core/event.h"

namespace caffe2 {

TEST(EventCPUTest, EventBasics) {
  DeviceOption device_option;
  device_option.set_device_type(CPU);
  Event event(device_option);
  CPUContext context;

  context.Record(&event);
  event.SetFinished();

  context.WaitEvent(event);
  event.Finish();

  event.Reset();
  event.Record(CPU, &context);
  event.SetFinished();
  event.Wait(CPU, &context);
}

} // namespace caffe2
