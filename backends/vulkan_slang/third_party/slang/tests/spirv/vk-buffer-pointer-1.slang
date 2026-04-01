//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -O0

// Note: spirv-opt will crash on the generated spirv, so use -O0 to disable it.

// CHECK: OpEntryPoint
// CHECK: OpTypeForwardPointer
typedef vk::BufferPointer<block_s> block_p;

struct block_s
{
      float4 x;
      block_p next;
};

struct TestPushConstant_t
{
      block_p root;
};

[[vk::push_constant]] TestPushConstant_t g_PushConstants;

[shader("fragment")]
float4 MainPs(void) : SV_Target0
{
#if __has_feature(hlsl_vk_buffer_pointer)
      block_p g_p = g_PushConstants.root;
      g_p = g_p.Get().next;
      if ((uint64_t)g_p == 0) // Null pointer test
          return float4(0.0, 0.0, 0.0, 0.0);
      return g_p.Get().x;
#else
    return float4(0.0);
#endif
}