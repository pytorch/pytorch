//TEST:SIMPLE(filecheck=CHECK): -target spirv -fvk-use-entrypoint-name -emit-spirv-directly

// CHECK: OpEntryPoint

[shader("vertex")]
float4 vmain(uint vertex_id : SV_VertexID) : SV_Position {
  return float4(vertex_id, 0, 0, 1);
}

[shader("pixel")]
float4 pmain(float4 position : SV_Position) : SV_Target {
  return position;
}