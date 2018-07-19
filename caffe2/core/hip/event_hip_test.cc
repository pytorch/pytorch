#include <gtest/gtest.h>
#include "caffe2/core/context.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/core/event.h"

namespace caffe2 {

TEST(EventHIPTest, EventBasics)
{
    if(!HasHipGPU())
        return;
    DeviceOption device_cpu;
    device_cpu.set_device_type(CPU);
    DeviceOption device_hip;
    device_hip.set_device_type(HIP);

    CPUContext context_cpu(device_cpu);
    HIPContext context_hip(device_hip);

    Event event_cpu(device_cpu);
    Event event_hip(device_hip);

    // CPU context and event interactions
    context_cpu.Record(&event_cpu);
    event_cpu.SetFinished();
    event_cpu.Finish();
    context_cpu.WaitEvent(event_cpu);

    event_cpu.Reset();
    event_cpu.Record(CPU, &context_cpu);
    event_cpu.SetFinished();
    event_cpu.Wait(CPU, &context_cpu);

    // HIP context and event interactions
    context_hip.SwitchToDevice();
    context_hip.Record(&event_hip);
    context_hip.WaitEvent(event_hip);
    event_hip.Finish();

    event_hip.Reset();
    event_hip.Record(HIP, &context_hip);
    event_hip.Wait(HIP, &context_hip);

    // CPU context waiting for HIP event
    context_cpu.WaitEvent(event_hip);

    // HIP context waiting for CPU event
    context_hip.WaitEvent(event_cpu);
}

} // namespace caffe2
