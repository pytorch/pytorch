//TEST:SIMPLE(filecheck=CHECK): -target spirv

//CHECK: OpEntryPoint

struct TypeA {
  float4 placeholder;
};
struct TypeB {
  float4 placeholder;
};

[[vk::binding(0, 0)]]
StructuredBuffer<TypeA> a_buffers[] : register(t0, space0);
[[vk::binding(0, 1)]]
StructuredBuffer<TypeB> b_buffer : register(t0, space0);

struct VertexIn {
  int32_t vert_idx : SV_VertexID;
  [[KnownBuiltin("DrawIndex")]]
  uint32_t draw_idx : POSITION0;
};

[shader("vertex")]
float4 vert(VertexIn i) : SV_POSITION {
  // Proper usage of NonUniformResourceIndex works fine
  float4 a =
      a_buffers[NonUniformResourceIndex(i.draw_idx)][i.vert_idx].placeholder;
  // Incorrect usage but dxc compiles this with no warning  
  float4 b = b_buffer[NonUniformResourceIndex(i.draw_idx)].placeholder;
  return a + b;
}
