//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

// CHECK: OpEntryPoint

struct Globals_s
{
      float4 g_vSomeConstantA;
      float4 g_vTestFloat4;
      float4 g_vSomeConstantB;
};

typedef vk::BufferPointer<Globals_s> Globals_p;

struct TestPushConstant_t
{
      Globals_p m_nBufferDeviceAddress;
};

[[vk::push_constant]] TestPushConstant_t g_PushConstants;

[shader("fragment")]
float4 MainPs(void) : SV_Target0
{
      float4 vTest = float4(1.0,0.0,0.0,0.0);
      g_PushConstants.m_nBufferDeviceAddress.Get().g_vTestFloat4 = vTest;
      return vTest;
}